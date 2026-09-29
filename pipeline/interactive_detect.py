#!/usr/bin/env python3
"""Stage 1, interactive: try your own visual prompts on a multi-object scene.

Runs   : inside the `foundationpose` container (torch + ultralytics).
In     : data/multi_object_scene/{scene_rgb.png, refs/<name>.{png,json}}  (or your own images)
Out    : output/interactive_<names>.png, plus per-object score/bbox printed

  --object NAME                        ready-made reference from data/multi_object_scene/refs/ (repeatable)
  --prompt NAME REF_IMAGE X0,Y0,X1,Y1  your own reference image + box around the object (repeatable)
  --scene PATH                         scene image (default: the bundled BOP frame)
"""
import argparse
import glob
import os
import sys

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config  # noqa: E402
from vision.detect import build_visual_prompt_model, detect, load_prompt  # noqa: E402
from vision.viz import COLORS, draw_detection  # noqa: E402

SCENE_DIR = os.path.join(REPO, 'data', 'multi_object_scene')
REFS_DIR = os.path.join(SCENE_DIR, 'refs')
OUT = os.path.join(REPO, 'output')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--object', action='append', default=[], metavar='NAME',
                   help='available: ' + ', '.join(sorted(
                       os.path.splitext(os.path.basename(f))[0] for f in glob.glob(REFS_DIR + '/*.png'))))
    p.add_argument('--prompt', action='append', default=[], nargs=3,
                   metavar=('NAME', 'REF_IMAGE', 'X0,Y0,X1,Y1'))
    p.add_argument('--scene', default=os.path.join(SCENE_DIR, 'scene_rgb.png'))
    p.add_argument('--out', default=None, help='output filename (default: interactive_<objects>.png)')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()['detect']
    if not args.object and not args.prompt:
        raise SystemExit('give at least one --object NAME or --prompt NAME IMAGE BBOX')
    os.makedirs(OUT, exist_ok=True)

    prompts = []
    for name in args.object:
        img, bbox = load_prompt(os.path.join(REFS_DIR, name + '.png'))
        prompts.append((name, img, bbox))
    for name, ref_image, bbox_str in args.prompt:
        img = cv2.imread(ref_image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit('cannot read reference image: %s' % ref_image)
        prompts.append((name, img, [float(v) for v in bbox_str.split(',')]))
    scene = cv2.imread(args.scene, cv2.IMREAD_COLOR)
    if scene is None:
        raise SystemExit('cannot read scene image: %s' % args.scene)

    model = build_visual_prompt_model(cfg['weights'], prompts, cfg['device'])
    dets = detect(model, scene, conf=cfg['conf'], device=cfg['device'])

    best = {}
    for d in dets:
        if d['mask'] is not None and d['score'] > best.get(d['name'], {'score': -1})['score']:
            best[d['name']] = d
    overlay = scene
    for i, (name, _, _) in enumerate(prompts):
        d = best.get(name)
        if d is None:
            print('  %-24s NOT DETECTED' % name)
            continue
        overlay = draw_detection(overlay, d, COLORS[i % len(COLORS)])
        print('  %-24s score=%.3f bbox=%s' % (name, d['score'], [round(v, 1) for v in d['bbox']]))

    path = os.path.join(OUT, args.out or 'interactive_%s.png' % '_'.join(n for n, _, _ in prompts))
    cv2.imwrite(path, overlay)
    print('-> ' + path)


if __name__ == '__main__':
    main()
