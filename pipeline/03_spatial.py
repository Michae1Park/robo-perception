#!/usr/bin/env python3
"""Stage 3 playground: depth -> point cloud -> table plane (RANSAC) -> occupancy grid, on the stage 1 image.
Guide: docs/STAGE3_SPATIAL.md. Host .venv (numpy + OpenCV only).

  python pipeline/03_spatial.py                         # config.yaml `spatial:` defaults
  python pipeline/03_spatial.py --resolution 0.02 --height-thresh 0.03 --tag coarse

Writes output/spatial/<tag>.png (2x2 panels) and opens it in Cursor.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config, spatial  # noqa: E402

SCENE = os.path.join(REPO, 'data', 'multi_object_scene')
COLORS = {spatial.FREE: (113, 204, 46), spatial.OCCUPIED: (60, 76, 231), spatial.UNKNOWN: (187, 187, 187)}  # BGR


def label(img, text):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (0, 0), (w + 12, h + 14), (255, 255, 255), -1)
    cv2.putText(img, text, (6, h + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def to_px(K, T_cam_world, pts_world):
    cam = pts_world @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:]


def main():
    cfg = load_config()['spatial']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('ransac_dist', 'resolution', 'height_thresh', 'max_height'):
        p.add_argument('--' + k.replace('_', '-'), type=float, default=cfg[k])
    p.add_argument('--ransac-iters', type=int, default=cfg['ransac_iters'])
    p.add_argument('--pose', default='play', help='stage 2 result output/pose/<TAG>.json to place on the table')
    p.add_argument('--tag', default='play')
    a = p.parse_args()
    cfg.update({k: getattr(a, k) for k in cfg if hasattr(a, k)})

    bgr = cv2.imread(os.path.join(SCENE, 'scene_rgb.png'))
    K = np.loadtxt(os.path.join(SCENE, 'camera_K.txt'))
    depth = cv2.imread(os.path.join(SCENE, 'scene_depth.png'), cv2.IMREAD_UNCHANGED) * 0.1 / 1000  # -> metres
    valid = depth > 0  # depth_to_points keeps exactly these pixels, in this order

    # 1. the whole stage (timed): points -> RANSAC plane -> table frame -> grid -> clearance
    t0 = time.perf_counter()
    s = spatial.analyze_scene(depth, K, cfg)
    t_ms = (time.perf_counter() - t0) * 1000
    T_wc = s['T_world_cam']
    T_cw = np.linalg.inv(T_wc)
    pts_world = spatial.depth_to_points(depth, K) @ T_wc[:3, :3].T + T_wc[:3, 3]
    height = np.zeros(depth.shape)
    height[valid] = pts_world[:, 2]
    grid, ((x0, x1), (y0, y1)), r = s['grid'], s['zone'], cfg['resolution']

    # 2. report
    n = s['plane_cam'][:3]
    print('\npoints %d (%d px without depth) | analyze_scene %.0f ms' % (s['n_points'], (~valid).sum(), t_ms))
    print('table plane (camera frame): %.3fx %+.3fy %+.3fz %+.3f = 0' % tuple(s['plane_cam']))
    print('  inliers %d (%.0f%% of points) within %.0f mm' % (s['inliers'].sum(), 100 * s['inliers'].mean(),
                                                              cfg['ransac_dist'] * 1000))
    print('  camera is %.3f m above the table, looking %.0f deg down from horizontal'
          % (s['plane_cam'][3], np.degrees(np.arcsin(abs(n[2])))))
    print('grid %d x %d cells of %.0f mm over %.2f x %.2f m | free %d | occupied %d | unknown %d'
          % (grid.shape[1], grid.shape[0], r * 1000, x1 - x0, y1 - y0, (grid == spatial.FREE).sum(),
             (grid == spatial.OCCUPIED).sum(), (grid == spatial.UNKNOWN).sum()))
    print('max clearance %.0f mm (largest free circle radius)' % (s['clearance'].max() * 1000))

    pose_path = os.path.join(REPO, 'output', 'pose', a.pose + '.json')
    obj = None
    if os.path.exists(pose_path):  # stage 2 pose -> where the bottle stands in the table frame
        T_wo = T_wc @ np.array(json.load(open(pose_path))['ob_in_cam'])
        obj = T_wo[:3, 3]
        tilt = np.degrees(np.arccos(abs(T_wo[2, 2])))
        print('mustard bottle (stage 2 %s): x=%.3f y=%.3f m, centre %.0f mm above the table, axis %.0f deg from vertical'
              % (a.pose, obj[0], obj[1], obj[2] * 1000, tilt))

    # 3. panels
    table = bgr.copy()  # A: RANSAC inliers
    inl = np.zeros(depth.shape, bool)
    inl[valid] = s['inliers']
    table[inl] = table[inl] // 2 + np.array([0, 127, 0], np.uint8)

    h = np.clip(height / 0.25, 0, 1)  # B: height above the table, 0..25 cm
    heat = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[np.abs(height) < cfg['height_thresh']] = 60
    heat[~valid] = 0

    proj = bgr.copy()  # C: grid cells drawn on the table in the image
    for iy, ix in np.ndindex(grid.shape):
        c = np.array([[x0 + ix * r, y0 + iy * r, 0], [x0 + (ix + 1) * r, y0 + iy * r, 0],
                      [x0 + (ix + 1) * r, y0 + (iy + 1) * r, 0], [x0 + ix * r, y0 + (iy + 1) * r, 0]])
        cv2.fillPoly(proj, [np.round(to_px(K, T_cw, c)).astype(np.int32)], COLORS[grid[iy, ix]])
    proj = cv2.addWeighted(proj, 0.45, bgr, 0.55, 0)

    top = np.zeros(grid.shape + (3,), np.uint8)  # D: top-down view (far side of the table at the top)
    for v, col in COLORS.items():
        top[grid == v] = col
    top = top[::-1]
    scale = min(640 // grid.shape[1], 480 // grid.shape[0]) or 1
    top = cv2.resize(top, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if obj is not None:
        cx, cy = int((obj[0] - x0) / r * scale), int((y1 - obj[1]) / r * scale)
        cv2.drawMarker(top, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(proj, tuple(np.round(to_px(K, T_cw, obj[None])[0]).astype(int)), (0, 0, 0),
                       cv2.MARKER_CROSS, 20, 2)
    canvas = np.full((480, 640, 3), 255, np.uint8)
    canvas[:top.shape[0], :top.shape[1]] = top[:480, :640]

    out = np.vstack([np.hstack([label(table, 'A  table plane (RANSAC inliers)'),
                                label(heat, 'B  height above table (0 - 25 cm)')]),
                     np.hstack([label(proj, 'C  grid on the image (+ = bottle)'),
                                label(canvas, 'D  grid from above: free / occupied / unknown')])])
    path = os.path.join(REPO, 'output', 'spatial', a.tag + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)
    print('\n-> ' + path)
    if 'VSCODE_IPC_HOOK_CLI' in os.environ:  # set in Cursor / VS Code terminals
        subprocess.run(['cursor', '--reuse-window', path])


if __name__ == '__main__':
    main()
