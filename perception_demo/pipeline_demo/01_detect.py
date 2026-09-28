#!/usr/bin/env python3
"""Stage (1) demo: YOLOE-seg visual-prompt detection on a real RGB frame (no ROS).

Reference: mustard0 frame 0's ground-truth mask supplies a tight bbox around the mustard bottle
in that frame -- this is the "one example image" visual prompt used by the real pipeline's
build_visual_prompt_model(). Detection then runs on a DIFFERENT frame of the same sequence, so
this exercises actual open-set matching rather than re-finding the same pixels.

Run inside the FoundationPose container (torch/ultralytics/opencv already installed there):
  docker start foundationpose
  docker exec -it foundationpose bash -lc \
    'cd /home/ai/workspace/prompt-pick-place && python3 perception_demo/pipeline_demo/01_detect.py'
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_perception'))
from ppp_perception.yoloe_prompt import detect  # noqa: E402

DATA = os.path.join(REPO, 'third_party', 'FoundationPose', 'demo_data', 'mustard0')
OUT = os.path.join(REPO, 'perception_demo', 'pipeline_demo', 'output')
TARGET_FRAME = 736
OBJECT_NAME = '006_mustard_bottle'
WEIGHTS = 'yoloe-11l-seg.pt'
DEVICE = 'cuda:0'


def build_visual_prompt_model(weights, name, ref_img, ref_bbox, device):
    """Same result as yoloe_prompt.build_visual_prompt_model, but only the `refer_image` code
    path -- the repo's `try` branch passes `return_vpe=True` straight into model.predict(),
    which current ultralytics (>=8.3.150) rejects with SyntaxError (not TypeError/AttributeError,
    so the repo's except clause never catches it). Not editing the tracked yoloe_prompt.py for
    this demo, so this reimplements just the working half of _visual_pe/build_visual_prompt_model.
    """
    import torch.nn.functional as F
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    model = YOLOE(weights)
    vp = dict(bboxes=np.array([ref_bbox], dtype=np.float32), cls=np.array([0]))
    model.predict(ref_img, refer_image=ref_img, visual_prompts=vp, predictor=YOLOEVPSegPredictor,
                  device=device, verbose=False)
    vpe = model.model.pe.detach().clone().to('cpu').reshape(1, 1, -1)
    vpe = F.normalize(vpe, dim=-1, p=2)
    model.predictor = None
    model.set_classes([name], vpe)
    model.predictor = None
    return model


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(DATA, 'rgb', '*.png')))
    frame_id = os.path.splitext(os.path.basename(frames[TARGET_FRAME]))[0]

    ref_bgr = cv2.imread(frames[0], cv2.IMREAD_COLOR)
    ref_mask = cv2.imread(frames[0].replace('rgb', 'masks'), cv2.IMREAD_UNCHANGED)
    ys, xs = np.where(ref_mask > 0)
    ref_bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    scene_bgr = cv2.imread(frames[TARGET_FRAME], cv2.IMREAD_COLOR)

    model = build_visual_prompt_model(WEIGHTS, OBJECT_NAME, ref_bgr, ref_bbox, DEVICE)
    dets = detect(model, scene_bgr, conf=0.05, device=DEVICE)
    cands = [d for d in dets if d['name'] == OBJECT_NAME and d['mask'] is not None]
    if not cands:
        raise SystemExit('no detection of %s in frame %s (%d raw detections)' %
                         (OBJECT_NAME, frame_id, len(dets)))
    chosen = max(cands, key=lambda d: d['score'])

    overlay = scene_bgr.copy()
    ov = overlay.copy()
    ov[chosen['mask']] = (0, 255, 0)
    overlay = cv2.addWeighted(ov, 0.4, overlay, 0.6, 0)
    x1, y1, x2, y2 = [int(v) for v in chosen['bbox']]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, '%s %.2f' % (chosen['name'], chosen['score']), (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, '01_detection.png'), overlay)
    cv2.imwrite(os.path.join(OUT, '01_mask.png'), (chosen['mask'].astype(np.uint8) * 255))
    with open(os.path.join(OUT, '01_result.json'), 'w') as f:
        json.dump({'frame_id': frame_id, 'frame_index': TARGET_FRAME, 'bbox': chosen['bbox'],
                   'score': chosen['score'], 'num_raw_detections': len(dets)}, f, indent=2)
    print('detected %s score=%.3f bbox=%s -> %s' % (
        chosen['name'], chosen['score'], chosen['bbox'], os.path.join(OUT, '01_detection.png')))


if __name__ == '__main__':
    main()
