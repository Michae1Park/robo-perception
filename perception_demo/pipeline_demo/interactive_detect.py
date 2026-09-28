#!/usr/bin/env python3
"""Interactive stage (1) demo: try your own visual-prompt detections (no ROS).

Register one or more objects from a reference image + box, then detect them in a scene image --
exactly the real pipeline's yoloe_prompt.build_visual_prompt_model()/detect(), reused directly
(no reimplementation).

Default scene: perception_demo/pipeline_demo/multi_object_scene/scene_rgb.png -- a real BOP
YCB-Video photo (scene 000059, frame 61) with 6 objects on a table and NO robot arm in frame.
Default ready-made prompts (from a DIFFERENT BOP scene, so this is genuine open-set matching, not
re-finding identical pixels): multi_object_scene/refs/005_tomato_soup_can.{png,json} and
multi_object_scene/refs/010_potted_meat_can.{png,json}.

Run inside the FoundationPose container (torch/ultralytics/opencv live in its `my` conda env --
call that env's python3 directly, plain python3 in a non-interactive `docker exec` resolves to the
system interpreter and lacks these packages):

  docker start foundationpose

  # use a ready-made prompt by name (looks up multi_object_scene/refs/<name>.{png,json})
  docker exec -it foundationpose bash -lc \
    'cd /home/ai/workspace/prompt-pick-place && /opt/conda/envs/my/bin/python3 \
     perception_demo/pipeline_demo/interactive_detect.py --object 005_tomato_soup_can'

  # register several objects at once, see YOLOE tell them apart in one cluttered scene
  docker exec -it foundationpose bash -lc \
    'cd /home/ai/workspace/prompt-pick-place && /opt/conda/envs/my/bin/python3 \
     perception_demo/pipeline_demo/interactive_detect.py \
     --object 005_tomato_soup_can --object 010_potted_meat_can'

  # your own reference image + box (x0,y0,x1,y1 in that image's pixel coordinates)
  docker exec -it foundationpose bash -lc \
    'cd /home/ai/workspace/prompt-pick-place && /opt/conda/envs/my/bin/python3 \
     perception_demo/pipeline_demo/interactive_detect.py \
     --prompt my_mug /path/to/reference.png 120,80,340,410 --scene /path/to/scene.png'
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_perception'))
from ppp_perception.yoloe_prompt import build_visual_prompt_model, detect, load_prompt  # noqa: E402

SCENE_DIR = os.path.join(REPO, 'perception_demo', 'pipeline_demo', 'multi_object_scene')
REFS_DIR = os.path.join(SCENE_DIR, 'refs')
DEFAULT_SCENE = os.path.join(SCENE_DIR, 'scene_rgb.png')
OUT = os.path.join(REPO, 'perception_demo', 'pipeline_demo', 'output')

_COLORS = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 255)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--object', action='append', default=[], metavar='NAME',
                   help='use the ready-made reference multi_object_scene/refs/NAME.{png,json} '
                        '(repeatable, e.g. --object 005_tomato_soup_can --object 010_potted_meat_can). '
                        'Available: ' + ', '.join(sorted(
                            os.path.splitext(os.path.basename(f))[0]
                            for f in glob.glob(os.path.join(REFS_DIR, '*.png')))))
    p.add_argument('--prompt', action='append', default=[], nargs=3,
                   metavar=('NAME', 'REF_IMAGE', 'X0,Y0,X1,Y1'),
                   help='register a custom object: name, path to a reference image, and a box '
                        'around it in that image (pixel xyxy, comma-separated). Repeatable.')
    p.add_argument('--scene', default=DEFAULT_SCENE, help='scene image to detect in (default: the '
                   'bundled multi-object BOP frame, no robot arm)')
    p.add_argument('--weights', default='yoloe-11l-seg.pt')
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=None, help='output filename (default: interactive_<objects>.png)')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.object and not args.prompt:
        raise SystemExit('give at least one --object NAME or --prompt NAME IMAGE BBOX')
    os.makedirs(OUT, exist_ok=True)

    prompts = []
    for name in args.object:
        ref_path = os.path.join(REFS_DIR, name + '.png')
        img, bbox = load_prompt(ref_path)
        prompts.append((name, img, bbox))
    for name, ref_image, bbox_str in args.prompt:
        img = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit('cannot read reference image: %s' % ref_image)
        bbox = [float(v) for v in bbox_str.split(',')]
        prompts.append((name, img, bbox))

    scene_bgr = cv2.imread(args.scene, cv2.IMREAD_COLOR)
    if scene_bgr is None:
        raise SystemExit('cannot read scene image: %s' % args.scene)

    print('registering %d object(s): %s' % (len(prompts), ', '.join(n for n, _, _ in prompts)))
    model = build_visual_prompt_model(args.weights, prompts, args.device)
    dets = detect(model, scene_bgr, conf=args.conf, device=args.device)

    overlay = scene_bgr.copy()
    found = {n: None for n, _, _ in prompts}
    for d in dets:
        if d['name'] not in found or d['mask'] is None:
            continue
        if found[d['name']] is None or d['score'] > found[d['name']]['score']:
            found[d['name']] = d

    for i, (name, _, _) in enumerate(prompts):
        d = found.get(name)
        color = _COLORS[i % len(_COLORS)]
        if d is None:
            print('  %-24s NOT DETECTED (raw detections in scene: %d)' % (name, len(dets)))
            continue
        ov = overlay.copy()
        ov[d['mask']] = color
        overlay = cv2.addWeighted(ov, 0.4, overlay, 0.6, 0)
        x1, y1, x2, y2 = [int(v) for v in d['bbox']]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, '%s %.2f' % (name, d['score']), (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        print('  %-24s score=%.3f bbox=%s' % (name, d['score'], [round(v, 1) for v in d['bbox']]))

    label = args.out or ('interactive_' + '_'.join(n for n, _, _ in prompts) + '.png')
    out_path = os.path.join(OUT, label)
    cv2.imwrite(out_path, overlay)
    print('total raw detections in scene: %d -> %s' % (len(dets), out_path))


if __name__ == '__main__':
    main()
