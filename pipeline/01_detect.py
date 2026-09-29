#!/usr/bin/env python3
"""Stage 1 - detection: YOLOE visual-prompt segmentation on the mustard0 sequence.

Prompt : box around the mustard bottle in frame 0 (from the dataset's GT mask).
Scene  : frame 736 of the same sequence (a different frame, so it is real matching).
Runs   : inside the `foundationpose` container (needs torch + ultralytics).
In     : third_party/FoundationPose/demo_data/mustard0/
Out    : output/01_detection.png, 01_mask.png, 01_result.json
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config  # noqa: E402
from vision.detect import build_visual_prompt_model, detect  # noqa: E402
from vision.viz import draw_detection  # noqa: E402

DATA = os.path.join(REPO, 'third_party', 'FoundationPose', 'demo_data', 'mustard0')
OUT = os.path.join(REPO, 'output')
TARGET_FRAME = 736
OBJECT_NAME = '006_mustard_bottle'


def main():
    cfg = load_config()['detect']
    os.makedirs(OUT, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(DATA, 'rgb', '*.png')))
    frame_id = os.path.splitext(os.path.basename(frames[TARGET_FRAME]))[0]

    ref_bgr = cv2.imread(frames[0], cv2.IMREAD_COLOR)
    ref_mask = cv2.imread(frames[0].replace('rgb', 'masks'), cv2.IMREAD_UNCHANGED)
    ys, xs = np.where(ref_mask > 0)
    ref_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    scene_bgr = cv2.imread(frames[TARGET_FRAME], cv2.IMREAD_COLOR)

    model = build_visual_prompt_model(cfg['weights'], [(OBJECT_NAME, ref_bgr, ref_bbox)], cfg['device'])
    dets = detect(model, scene_bgr, conf=cfg['conf'], device=cfg['device'])
    cands = [d for d in dets if d['name'] == OBJECT_NAME and d['mask'] is not None]
    if not cands:
        raise SystemExit('no detection of %s in frame %s (%d raw detections)' %
                         (OBJECT_NAME, frame_id, len(dets)))
    best = max(cands, key=lambda d: d['score'])

    cv2.imwrite(os.path.join(OUT, '01_detection.png'), draw_detection(scene_bgr, best))
    cv2.imwrite(os.path.join(OUT, '01_mask.png'), best['mask'].astype(np.uint8) * 255)
    with open(os.path.join(OUT, '01_result.json'), 'w') as f:
        json.dump({'frame_id': frame_id, 'frame_index': TARGET_FRAME, 'bbox': best['bbox'],
                   'score': best['score'], 'num_raw_detections': len(dets)}, f, indent=2)
    print('detected %s score=%.3f bbox=%s' % (best['name'], best['score'], best['bbox']))


if __name__ == '__main__':
    main()
