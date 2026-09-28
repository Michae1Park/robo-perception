#!/usr/bin/env python3
"""Runs INSIDE the FoundationPose container, invoked by talk_and_pick.py (the host-side
orchestrator) via `docker exec`. Does stage (1) YOLOE TEXT-prompt detection (open-vocabulary --
no reference image, just a phrase) on the multi-object scene, and, if a mesh path is given and
something was detected, stage (2) single-frame FoundationPose registration using the mask stage
(1) produced.

Not meant to be run by hand -- talk_and_pick.py copies this file to third_party/FoundationPose/
and runs it there (cwd matters: FoundationPose's own `from estimater import ...` resolves via the
script's own directory, same convention 02_pose.py already relies on) with the right arguments.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = '/home/ai/workspace/prompt-pick-place'
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_perception'))
from ppp_perception.yoloe_prompt import build_text_prompt_model, detect  # noqa: E402

SCENE = os.path.join(REPO, 'perception_demo/pipeline_demo/multi_object_scene')
OUT = os.path.join(REPO, 'perception_demo/pipeline_demo/output')
DEPTH_SCALE_MM_PER_UNIT = 0.1  # this scene's BOP convention, see multi_object_scene/meta.json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phrase', required=True, help='open-vocabulary text prompt, e.g. "mustard sauce"')
    p.add_argument('--label', required=True, help='filesystem-safe tag for output filenames')
    p.add_argument('--mesh', default=None, help='assets/ycb/<name>/textured.obj -- if given, also '
                   'run stage (2) FoundationPose registration')
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--weights', default='yoloe-11l-seg.pt')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    scene_bgr = cv2.imread(os.path.join(SCENE, 'scene_rgb.png'), cv2.IMREAD_COLOR)
    print('text-prompt detecting: "%s"' % args.phrase)
    model = build_text_prompt_model(args.weights, [args.phrase], args.device)
    dets = [d for d in detect(model, scene_bgr, conf=args.conf, device=args.device) if d['mask'] is not None]

    result = {'phrase': args.phrase, 'label': args.label, 'detected': False, 'pose': None}
    if not dets:
        print('NOT DETECTED: "%s"' % args.phrase)
        with open(os.path.join(OUT, 'talk_%s_result.json' % args.label), 'w') as f:
            json.dump(result, f, indent=2)
        return

    best = max(dets, key=lambda d: d['score'])
    ov = scene_bgr.copy()
    ov[best['mask']] = (255, 0, 255)
    overlay = cv2.addWeighted(ov, 0.4, scene_bgr, 0.6, 0)
    x1, y1, x2, y2 = [int(v) for v in best['bbox']]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(overlay, '"%s" %.2f' % (args.phrase, best['score']), (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, 'talk_%s_detect.png' % args.label), overlay)
    mask_path = os.path.join(OUT, 'talk_%s_mask.png' % args.label)
    cv2.imwrite(mask_path, best['mask'].astype(np.uint8) * 255)
    result.update(detected=True, score=best['score'], bbox=best['bbox'], mask_path=mask_path)
    print('DETECTED "%s": score=%.3f bbox=%s' % (args.phrase, best['score'],
                                                  [round(v, 1) for v in best['bbox']]))

    if not args.mesh:
        with open(os.path.join(OUT, 'talk_%s_result.json' % args.label), 'w') as f:
            json.dump(result, f, indent=2)
        print('no mesh given -- stopping after detection')
        return

    # ---- stage (2): single-frame FoundationPose registration ----
    from estimater import (FoundationPose, PoseRefinePredictor, ScorePredictor, draw_posed_3d_box,
                           draw_xyz_axis, dr, set_logging_format, set_seed)
    import trimesh

    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(args.mesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox3d = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    debug_dir = os.path.join(OUT, 'talk_%s_fp_debug' % args.label)
    os.makedirs(debug_dir, exist_ok=True)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=1, glctx=glctx)

    K = np.loadtxt(os.path.join(SCENE, 'camera_K.txt'))
    color = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(os.path.join(SCENE, 'scene_depth.png'), cv2.IMREAD_UNCHANGED)
    depth = depth_raw.astype(np.float32) * DEPTH_SCALE_MM_PER_UNIT / 1000.0  # BOP units -> metres

    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=best['mask'], iteration=5)
    np.savetxt(os.path.join(OUT, 'talk_%s_pose_ob_in_cam.txt' % args.label), pose.reshape(4, 4))
    np.savetxt(os.path.join(OUT, 'talk_%s_cam_K.txt' % args.label), K)

    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox3d)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                        transparency=0, is_input_rgb=True)
    cv2.imwrite(os.path.join(OUT, 'talk_%s_pose_overlay.png' % args.label), vis[..., ::-1])

    result['pose'] = {'ob_in_cam': pose.reshape(4, 4).tolist(), 'mesh': args.mesh}
    with open(os.path.join(OUT, 'talk_%s_result.json' % args.label), 'w') as f:
        json.dump(result, f, indent=2)
    print('pose (ob_in_cam) ->')
    print(pose.reshape(4, 4))


if __name__ == '__main__':
    main()
