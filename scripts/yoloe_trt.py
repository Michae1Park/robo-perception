#!/usr/bin/env python3
"""[NOT YET RUN] YOLOE visual-prompt model -> TensorRT engine, and PyTorch vs TensorRT latency benchmark.

Run inside the `foundationpose` container (needs torch + ultralytics + TensorRT).
Prompts: every data/multi_object_scene/refs/<name>.png. Bench image default: data/multi_object_scene/scene_rgb.png.

  python scripts/yoloe_trt.py export --out models/yoloe_vp.engine
  python scripts/yoloe_trt.py bench  --engine models/yoloe_vp.engine [--images 'frames/*.png']

The visual-prompt embeddings of all objects are set as the model's classes before export, so the
engine detects exactly the registered objects with one forward pass.
"""
import argparse
import glob
import json
import os
import platform
import shutil
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vision import load_config  # noqa: E402
from vision.detect import build_visual_prompt_model, load_prompt  # noqa: E402

REFS = os.path.join(ROOT, 'data', 'multi_object_scene', 'refs')
SCENE = os.path.join(ROOT, 'data', 'multi_object_scene', 'scene_rgb.png')


def ref_names():
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(REFS + '/*.png'))


def vp_model(weights, device):
    prompts = []
    for n in ref_names():
        img, bbox = load_prompt(os.path.join(REFS, n + '.png'))
        prompts.append((n, img, bbox))
    return build_visual_prompt_model(weights, prompts, device)


def cmd_export(a):
    model = vp_model(a.weights, a.device)
    path = model.export(format='engine', half=not a.fp32, imgsz=a.imgsz, device=a.device, workspace=4)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    shutil.move(path, a.out)
    print('engine ->', a.out, '| classes:', ref_names())


def load_images(a):
    import cv2
    imgs = [cv2.imread(p) for p in (sorted(glob.glob(a.images)) if a.images else [SCENE])]
    return [i for i in imgs if i is not None]


def bench(model, imgs, device, half, warmup, iters):
    for i in range(warmup):
        model.predict(imgs[i % len(imgs)], device=device, half=half, retina_masks=True, verbose=False)
    total, infer = [], []
    for i in range(iters):
        t0 = time.perf_counter()
        r = model.predict(imgs[i % len(imgs)], device=device, half=half, retina_masks=True, verbose=False)[0]
        total.append((time.perf_counter() - t0) * 1000.0)
        infer.append(r.speed['inference'])

    def p(v, q):
        return float(np.percentile(v, q))
    return {'end_to_end_p50_ms': p(total, 50), 'end_to_end_p95_ms': p(total, 95),
            'inference_p50_ms': p(infer, 50), 'inference_mean_ms': statistics.mean(infer)}


def cmd_bench(a):
    import torch
    from ultralytics import YOLO
    imgs = load_images(a)
    rows = []
    model = vp_model(a.weights, a.device)
    rows.append(('PyTorch FP32', bench(model, imgs, a.device, False, a.warmup, a.iters)))
    rows.append(('PyTorch FP16', bench(model, imgs, a.device, True, a.warmup, a.iters)))
    if a.engine:
        trt = YOLO(a.engine, task='segment')
        rows.append(('TensorRT FP16', bench(trt, imgs, a.device, True, a.warmup, a.iters)))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    lines = ['YOLOE visual-prompt seg (%s, %d classes, imgsz %d) on %s (desktop GPU)'
             % (os.path.basename(a.weights), len(ref_names()), a.imgsz, gpu), '',
             '| Backend | inference p50 (ms) | end-to-end p50 (ms) | end-to-end p95 (ms) |',
             '|---|---|---|---|']
    for name, r in rows:
        lines.append('| %s | %.2f | %.2f | %.2f |' % (name, r['inference_p50_ms'], r['end_to_end_p50_ms'],
                                                      r['end_to_end_p95_ms']))
    md = '\n'.join(lines) + '\n'
    print(md)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump({'gpu': gpu, 'host': platform.node(), 'rows': dict(rows)}, f, indent=2)
        with open(os.path.splitext(a.out)[0] + '.md', 'w') as f:
            f.write(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default=load_config()['detect']['weights'])
    ap.add_argument('--device', default=load_config()['detect']['device'])
    ap.add_argument('--imgsz', type=int, default=640)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export')
    e.add_argument('--out', default=os.path.join(ROOT, 'models', 'yoloe_vp.engine'))
    e.add_argument('--fp32', action='store_true')
    b = sub.add_parser('bench')
    b.add_argument('--engine', default='')
    b.add_argument('--images', default='', help="glob of scene frames, e.g. 'frames/*.png'")
    b.add_argument('--warmup', type=int, default=20)
    b.add_argument('--iters', type=int, default=200)
    b.add_argument('--out', default=os.path.join(ROOT, 'output', 'yoloe_latency.json'))
    a = ap.parse_args()
    {'export': cmd_export, 'bench': cmd_bench}[a.cmd](a)


if __name__ == '__main__':
    main()
