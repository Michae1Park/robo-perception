#!/usr/bin/env python3
"""Container half of talk_and_pick.py (not run by hand): text-prompt detection on the multi-object
scene, then - if --mesh is given and something was detected - FoundationPose registration.
Runs inside the `foundationpose` container. Writes output/talk_<label>_*.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config, pose  # noqa: E402
from vision.detect import build_text_prompt_model, detect  # noqa: E402
from vision.viz import draw_detection  # noqa: E402

SCENE = os.path.join(REPO, 'data', 'multi_object_scene')
OUT = os.path.join(REPO, 'output')
DEPTH_SCALE_MM_PER_UNIT = 0.1  # BOP convention, see data/multi_object_scene/meta.json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phrase', required=True, help='text prompt, e.g. "mustard bottle"')
    p.add_argument('--label', required=True, help='filesystem-safe tag for output filenames')
    p.add_argument('--mesh', default=None, help='object mesh (.obj); if given, also run stage 2')
    args = p.parse_args()
    cfg = load_config()
    dcfg = cfg['detect']
    os.makedirs(OUT, exist_ok=True)
    result_path = os.path.join(OUT, 'talk_%s_result.json' % args.label)

    scene_bgr = cv2.imread(os.path.join(SCENE, 'scene_rgb.png'), cv2.IMREAD_COLOR)
    model = build_text_prompt_model(dcfg['weights'], [args.phrase], dcfg['device'])
    dets = [d for d in detect(model, scene_bgr, conf=dcfg['conf'], device=dcfg['device'])
            if d['mask'] is not None]

    result = {'phrase': args.phrase, 'label': args.label, 'detected': False, 'pose': None}
    if not dets:
        print('NOT DETECTED: "%s"' % args.phrase)
    else:
        best = max(dets, key=lambda d: d['score'])
        cv2.imwrite(os.path.join(OUT, 'talk_%s_detect.png' % args.label),
                    draw_detection(scene_bgr, best, (255, 0, 255), '"%s" %.2f' % (args.phrase, best['score'])))
        result.update(detected=True, score=best['score'], bbox=best['bbox'])
        print('DETECTED "%s": score=%.3f bbox=%s' % (args.phrase, best['score'],
                                                      [round(v, 1) for v in best['bbox']]))

        if args.mesh:
            K = np.loadtxt(os.path.join(SCENE, 'camera_K.txt'))
            depth_raw = cv2.imread(os.path.join(SCENE, 'scene_depth.png'), cv2.IMREAD_UNCHANGED)
            depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE_MM_PER_UNIT / 1000.0
            ob_in_cam, vis = pose.register(args.mesh, K, cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB), depth_m,
                                           best['mask'], os.path.join(OUT, 'talk_%s_fp_debug' % args.label),
                                           cfg['pose']['iterations'])
            cv2.imwrite(os.path.join(OUT, 'talk_%s_pose_overlay.png' % args.label), vis[..., ::-1])
            result['pose'] = {'ob_in_cam': ob_in_cam.tolist(), 'mesh': args.mesh}
            print(ob_in_cam)

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
