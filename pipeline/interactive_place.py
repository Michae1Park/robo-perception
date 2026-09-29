#!/usr/bin/env python3
"""Stage 5 playground: wrist-camera RGB-D of the shelf -> horizontal supports -> free space -> where the object fits.
Guide: docs/STAGE5_PLACE.md. Host .venv (numpy + OpenCV only). Data: data/sim/shelf/ from sim/scene.py.

  python pipeline/interactive_place.py                          # place the mustard bottle, config.yaml `place:`
  python pipeline/interactive_place.py --object 005_tomato_soup_can --tag soup
  python pipeline/interactive_place.py --hand-clearance 0.15 --tag bighand

Writes output/place/<tag>.png and opens it in Cursor.
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
from vision import grasp as g, load_config, place  # noqa: E402

DATA = os.path.join(REPO, 'data', 'sim', 'shelf')
PALETTE = [(46, 204, 113), (231, 76, 60), (52, 152, 219), (241, 196, 15), (155, 89, 182), (26, 188, 156)]  # RGB
CELL = {place.FREE: (113, 204, 46), place.OCCUPIED: (60, 76, 231), place.UNKNOWN: (170, 170, 170)}  # BGR


def label(img, text):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (0, 0), (w + 12, h + 14), (255, 255, 255), -1)
    cv2.putText(img, text, (6, h + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def text(img, s, xy, color=(0, 0, 0)):
    for c, w in (((255, 255, 255), 3), (color, 1)):
        cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, w, cv2.LINE_AA)


def to_px(K, T_cam_base, pts):
    c = pts @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    uv = c @ K.T
    return np.round(uv[:, :2] / uv[:, 2:]).astype(np.int32)


def main():
    cfg = load_config()['place']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--object', default='006_mustard_bottle', help='YCB name in assets/ycb (placed upright)')
    for k in ('ransac_dist', 'resolution', 'height_thresh', 'margin', 'hand_clearance', 'max_reach'):
        p.add_argument('--' + k.replace('_', '-'), type=float, default=cfg[k])
    p.add_argument('--frame', default='000000')
    p.add_argument('--tag', default='play')
    a = p.parse_args()
    cfg.update({k: getattr(a, k) for k in cfg if hasattr(a, k)})

    bgr = cv2.imread(os.path.join(DATA, 'rgb', a.frame + '.png'))
    depth = cv2.imread(os.path.join(DATA, 'depth', a.frame + '.png'), cv2.IMREAD_UNCHANGED) / 1000.0
    K = np.loadtxt(os.path.join(DATA, 'cam_K.txt'))
    T_bc = np.loadtxt(os.path.join(DATA, 'T_base_cam.txt'))   # from the arm's joint angles: which way is up
    T_cb = np.linalg.inv(T_bc)
    verts = g.read_obj_vertices(os.path.join(REPO, 'assets', 'ycb', a.object, 'textured.obj'))
    radius = np.linalg.norm(verts[:, :2], axis=1).max() + cfg['margin']   # upright, any yaw
    height = np.ptp(verts[:, 2])
    need = height + cfg['hand_clearance']

    # 1. points + normals -> 2. horizontal planes -> 3. supports -> 4. free space -> 5. candidates
    t0 = time.perf_counter()
    P, n, valid, ok = place.points_and_normals(depth, K, T_bc)
    planes = place.find_planes(P, n, ok, cfg)
    grid = place.Grid(P[valid][:, :2], cfg['resolution'])
    supports = place.find_supports(P, planes, grid, cfg)
    for s in supports:
        place.analyze_support(s, supports, P, valid, grid, cfg)
        s['reachable'] = place.reachable(s, grid, cfg)
    cands = [(i, c) for i, s in enumerate(supports) for c in place.candidates(s, grid, radius, need, cfg)]
    cands.sort(key=lambda ic: -ic[1]['clearance'])
    t_ms = (time.perf_counter() - t0) * 1000

    # report
    gt = json.load(open(os.path.join(DATA, 'shelf.json')))
    print('\npoints %d | horizontal-facing %d | %d planes -> %d supports | %.0f ms'
          % (valid.sum(), (ok & (np.abs(n[..., 2]) > np.cos(np.radians(cfg['max_tilt_deg'])))).sum(),
             len(planes), len(supports), t_ms))
    print('object %s upright: footprint radius %.0f mm (incl. %.0f margin), height %.0f mm -> needs %.0f mm headroom'
          % (a.object, radius * 1000, cfg['margin'] * 1000, height * 1000, need * 1000))
    print('\nsupports (S = support id; GT z = nearest shelf board height from shelf.json, - = not a shelf board)')
    print('%-3s %8s %6s %10s %6s %6s %6s %10s %10s  %s' % ('S', 'z (m)', 'GT z', 'area', 'free', 'occ', 'unk',
                                                        'headroom', 'max clear', 'candidates'))
    for i, s in enumerate(supports):
        gz = min(gt['levels_z'], key=lambda z: abs(z - s['z']))
        hr = s['ceiling_z'] - s['z']
        k = sum(1 for j, _ in cands if j == i)
        print('%-3d %8.3f %6s %7.2f m2 %6d %6d %6d %10s %7.0f mm  %s'
              % (i, s['z'], ('%.2f' % gz) if abs(gz - s['z']) < 0.03 else '-', s['hull'].sum() * grid.res ** 2,
                 (s['grid'] == place.FREE).sum(), (s['grid'] == place.OCCUPIED).sum(),
                 ((s['grid'] == place.UNKNOWN) & s['hull']).sum(),
                 'open' if np.isinf(hr) else '%.0f mm' % (hr * 1000), s['clearance'].max() * 1000,
                 k if s['reachable'] else 'out of reach'))
    print('\ncandidates, best first (place point = object origin standing upright on the support)')
    print('%-3s %-3s %24s %10s %10s %8s  %s' % ('#', 'S', 'place point, base (m)', 'clearance', 'headroom',
                                                  'reach', 'pose in camera (m)'))
    for r, (i, c) in enumerate(cands):
        pt = np.array([*c['xy'], c['z'] - verts[:, 2].min()])   # object origin when standing on the support
        cam = T_cb[:3, :3] @ pt + T_cb[:3, 3]
        print('%-3d %-3d %8.3f %7.3f %7.3f %7.0f mm %10s %5.2f m  %6.3f %6.3f %6.3f'
              % (r, i, *pt, c['clearance'] * 1000, 'open' if np.isinf(c['headroom']) else '%.0f mm' % (c['headroom'] * 1000),
                 c['reach'], *cam))

    # draw: A supports, B free space + candidates on the image, C top-down per support
    A = bgr.copy()
    for i, s in enumerate(supports):
        col = PALETTE[i % len(PALETTE)][::-1]
        A[s['pixels']] = (A[s['pixels']] * 0.35 + np.array(col) * 0.65).astype(np.uint8)
        v, u = np.argwhere(s['pixels']).mean(axis=0).astype(int)
        text(A, 'S%d z=%.2f' % (i, s['z']), (u - 40, v))
    B = bgr.copy()
    for s in supports:
        if not s['reachable']:
            continue
        for iy, ix in np.argwhere(s['hull']):
            c = grid.centre(iy, ix)
            sq = np.array([[c[0] - grid.res / 2, c[1] - grid.res / 2], [c[0] + grid.res / 2, c[1] - grid.res / 2],
                           [c[0] + grid.res / 2, c[1] + grid.res / 2], [c[0] - grid.res / 2, c[1] + grid.res / 2]])
            cv2.fillPoly(B, [to_px(K, T_cb, np.c_[sq, np.full(4, s['z'])])], CELL[s['grid'][iy, ix]])
    B = cv2.addWeighted(B, 0.45, bgr, 0.55, 0)
    ring = np.c_[np.cos(np.linspace(0, 2 * np.pi, 40)), np.sin(np.linspace(0, 2 * np.pi, 40))]
    for r, (i, c) in reversed(list(enumerate(cands))):
        col = (255, 0, 255) if r == 0 else (255, 255, 0)
        foot = to_px(K, T_cb, np.c_[c['xy'] + radius * ring, np.full(40, c['z'])])   # footprint on the support
        cv2.polylines(B, [foot], True, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(B, [foot], True, col, 2, cv2.LINE_AA)
        u, v = to_px(K, T_cb, np.array([[*c['xy'], c['z']]]))[0]
        text(B, str(r), (int(u) - 4, int(v) + 5), col)

    tiles = []
    for i, s in enumerate(supports):
        if not s['reachable']:
            continue
        ys, xs = np.nonzero(s['hull'])
        crop = s['grid'][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        img = np.zeros(crop.shape + (3,), np.uint8)
        for v, col in CELL.items():
            img[crop == v] = col
        sc = max(1, min(8, 300 // max(crop.shape)))
        img = cv2.resize(img[:, ::-1], None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)  # as the camera sees it
        for r, (j, c) in enumerate(cands):
            if j == i:
                iy, ix = grid.cells(c['xy'][None])
                cx, cy = (xs.max() - ix[0] + 0.5) * sc, (iy[0] - ys.min() + 0.5) * sc
                cv2.circle(img, (int(cx), int(cy)), int(radius / grid.res * sc), (0, 0, 0), 1, cv2.LINE_AA)
                text(img, str(r), (int(cx) - 4, int(cy) + 4))
        tile = np.full((img.shape[0] + 24, max(img.shape[1], 110), 3), 255, np.uint8)
        tile[24:, :img.shape[1]] = img
        cv2.putText(tile, 'S%d z=%.2f' % (i, s['z']), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(tile)
    h = max(t.shape[0] for t in tiles) + 36
    C = np.full((h, 1280, 3), 255, np.uint8)
    x = 0
    for t in tiles:  # one row, left to right
        C[32:32 + t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + 16
    out = np.vstack([np.hstack([label(A, 'A  horizontal supports (RANSAC planes, split into pieces)'),
                                label(B, 'B  free / occupied / unknown + where %s fits' % a.object[4:])]),
                     label(C, 'C  each reachable support from above, as the camera faces it (back at the top): green free, red occupied, grey unknown')])
    path = os.path.join(REPO, 'output', 'place', a.tag + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)
    print('\n-> ' + path)
    if 'VSCODE_IPC_HOOK_CLI' in os.environ:  # set in Cursor / VS Code terminals
        subprocess.run(['cursor', '--reuse-window', path])


if __name__ == '__main__':
    main()
