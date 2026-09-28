#!/usr/bin/env python3
"""YOLOE visual-prompt model -> TensorRT engine, and PyTorch vs TensorRT latency benchmark.

  python3 scripts/yoloe_trt.py export --out models/yoloe_vp.engine
  python3 scripts/yoloe_trt.py bench  --engine models/yoloe_vp.engine [--images 'frames/*.png']

The visual-prompt embeddings of all objects (one example image each) are set as the model's
classes before export, so the engine detects exactly the registered objects with one forward pass.
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
for pkg in ('ppp_common', 'ppp_perception'):
    sys.path.insert(0, os.path.join(ROOT, 'ros2', pkg))
from ppp_common.config import SceneConfig  # noqa: E402
from ppp_perception.yoloe_prompt import build_visual_prompt_model, load_prompt  # noqa: E402


def vp_model(cfg, weights, device):
    prompts = []
    for n in cfg.object_names:
        img, bbox = load_prompt(cfg.prompt_path(n))
        prompts.append((n, img, bbox))
    return build_visual_prompt_model(weights, prompts, device)


def cmd_export(a, cfg):
    model = vp_model(cfg, a.weights, a.device)
    path = model.export(format='engine', half=not a.fp32, imgsz=a.imgsz, device=a.device, workspace=4)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    shutil.move(path, a.out)
    print('engine ->', a.out, '| classes:', cfg.object_names)


def load_images(a, cfg):
    import cv2
    paths = sorted(glob.glob(a.images)) if a.images else []
    imgs = [cv2.imread(p) for p in paths]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        # fall back to the prompt crops pasted on a 640x480 canvas (latency does not depend on content)
        for n in cfg.object_names:
            crop, _ = load_prompt(cfg.prompt_path(n))
            canvas = np.full((cfg.camera['height'], cfg.camera['width'], 3), 127, np.uint8)
            h, w = min(crop.shape[0], canvas.shape[0]), min(crop.shape[1], canvas.shape[1])
            canvas[:h, :w] = crop[:h, :w]
            imgs.append(canvas)
    return imgs


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


def cmd_bench(a, cfg):
    import torch
    from ultralytics import YOLO
    imgs = load_images(a, cfg)
    rows = []
    model = vp_model(cfg, a.weights, a.device)
    rows.append(('PyTorch FP32', bench(model, imgs, a.device, False, a.warmup, a.iters)))
    rows.append(('PyTorch FP16', bench(model, imgs, a.device, True, a.warmup, a.iters)))
    if a.engine:
        trt = YOLO(a.engine, task='segment')
        rows.append(('TensorRT FP16', bench(trt, imgs, a.device, True, a.warmup, a.iters)))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    lines = ['YOLOE visual-prompt seg (%s, %d classes, imgsz %d) on %s (desktop GPU)'
             % (os.path.basename(a.weights), len(cfg.object_names), a.imgsz, gpu), '',
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
    ap.add_argument('--assets', default='')
    ap.add_argument('--weights', default='yoloe-11l-seg.pt')
    ap.add_argument('--device', default='cuda:0')
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
    b.add_argument('--out', default=os.path.join(ROOT, 'results', 'yoloe_latency.json'))
    a = ap.parse_args()
    cfg = SceneConfig(assets=a.assets)
    {'export': cmd_export, 'bench': cmd_bench}[a.cmd](a, cfg)


if __name__ == '__main__':
    main()
