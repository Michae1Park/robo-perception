#!/usr/bin/env python3
"""Stage 1 playground: prompts + scene image -> every YOLOE detection, timing, overlay image.
Guide: docs/STAGE1_DETECT.md. Live/slider version: notebooks/yoloe_playground.ipynb.

  --text "mustard bottle"             text prompt (repeat for more classes)
  --ref 006_mustard_bottle            visual prompt from data/multi_object_scene/refs/
  --prompt NAME REF.png X0,Y0,X1,Y1   your own visual prompt (image + box around the object)
  --prompt-free                       no prompts: *-seg-pf.pt weights, built-in 4585-class vocabulary

Defaults come from config.yaml `detect:`. Writes output/yoloe/<tag>.png and, when run from a
Cursor terminal, opens it as a tab.
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

os.environ['YOLO_VERBOSE'] = 'False'  # hide ultralytics' per-predict banner
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config  # noqa: E402
from vision.detect import (build_prompt_free_model, build_text_prompt_model, build_visual_prompt_model,  # noqa: E402
                           detect, load_prompt)
from vision.viz import COLORS, draw_detection  # noqa: E402

SCENE_DIR = os.path.join(REPO, 'data', 'multi_object_scene')


def parse_args():
    cfg = load_config()['detect']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--text', action='append', default=[])
    p.add_argument('--ref', action='append', default=[])
    p.add_argument('--prompt', action='append', default=[], nargs=3, metavar=('NAME', 'IMAGE', 'X0,Y0,X1,Y1'))
    p.add_argument('--prompt-free', action='store_true', help='use the -pf weights of the same size; no prompts')
    p.add_argument('--scene', default=os.path.join(SCENE_DIR, 'scene_rgb.png'))
    p.add_argument('--weights', default=cfg['weights'])
    p.add_argument('--conf', type=float, default=cfg['conf'])
    p.add_argument('--iou', type=float, default=cfg['iou'])
    p.add_argument('--imgsz', type=int, default=cfg['imgsz'])
    p.add_argument('--max-det', type=int, default=cfg['max_det'])
    p.add_argument('--half', action='store_true', default=cfg['half'])
    p.add_argument('--tag', default='play')
    a = p.parse_args()
    n_kinds = sum(map(bool, (a.text, a.ref or a.prompt, a.prompt_free)))
    if n_kinds != 1:
        p.error('give exactly one of: --text, --ref/--prompt (visual), --prompt-free')
    return a, cfg['device']


def main():
    a, device = parse_args()
    scene = cv2.imread(a.scene)

    # 1. build the model: its classes are the prompts
    t0 = time.perf_counter()
    if a.prompt_free:  # same size, -pf weights: the vocabulary is baked in
        a.weights = a.weights if a.weights.endswith('-pf.pt') else a.weights.replace('-seg.pt', '-seg-pf.pt')
        classes = None
        model = build_prompt_free_model(a.weights)
    elif a.text:
        classes = a.text
        model = build_text_prompt_model(a.weights, a.text, device)
    else:
        visual = [(n, *load_prompt(os.path.join(SCENE_DIR, 'refs', n + '.png'))) for n in a.ref]
        visual += [(n, cv2.imread(img), [float(v) for v in box.split(',')]) for n, img, box in a.prompt]
        classes = [v[0] for v in visual]
        model = build_visual_prompt_model(a.weights, visual, device)
    t_build = time.perf_counter() - t0

    # 2. detect: one warm-up run, then time 20
    kw = dict(conf=a.conf, iou=a.iou, imgsz=a.imgsz, max_det=a.max_det, half=a.half, device=device)
    detect(model, scene, **kw)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        dets = detect(model, scene, **kw)
        times.append((time.perf_counter() - t0) * 1000)

    # 3. report + draw
    print('\nprompts: %s' % ('none (prompt-free, %d-class vocabulary)' % len(model.names) if a.prompt_free else classes))
    print('weights=%s imgsz=%d conf=%.2f iou=%.2f half=%s' % (a.weights, a.imgsz, a.conf, a.iou, a.half))
    print('model build %.2f s | predict p50 %.1f ms' % (t_build, np.median(times)))
    print('\n%-3s %-22s %6s  %-26s %8s' % ('#', 'class', 'score', 'bbox xyxy', 'mask px'))
    overlay = scene
    classes = classes or sorted({d['name'] for d in dets})
    for i, d in enumerate(sorted(dets, key=lambda d: -d['score'])):
        print('%-3d %-22s %6.3f  %-26s %8d' % (i, d['name'], d['score'], [int(v) for v in d['bbox']], d['mask'].sum()))
        overlay = draw_detection(overlay, d, COLORS[classes.index(d['name']) % len(COLORS)])

    path = os.path.join(REPO, 'output', 'yoloe', a.tag + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, overlay)
    print('\n-> ' + path)
    if 'VSCODE_IPC_HOOK_CLI' in os.environ:  # set in Cursor / VS Code terminals
        subprocess.run(['cursor', '--reuse-window', path])


if __name__ == '__main__':
    main()
