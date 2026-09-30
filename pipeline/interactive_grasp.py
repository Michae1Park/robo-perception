#!/usr/bin/env python3
"""Stage 4 playground: stage 2 pose + stage 3 table -> parallel-jaw grasp candidates -> filter, on the stage 1 image.
Guide: docs/STAGE4_GRASP.md. Host .venv (numpy + OpenCV only).

  python pipeline/interactive_grasp.py                        # config.yaml `gripper:` + `grasp:` defaults
  python pipeline/interactive_grasp.py --max-tilt 100 --tag side

Needs output/pose/<--pose>.json from interactive_pose.py. Writes output/grasp/<tag>.png and opens it in Cursor.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import grasp as g, load_config, read_obj_vertices, spatial, transforms as tf  # noqa: E402

SCENE = os.path.join(REPO, 'data', 'multi_object_scene')
FINGER_LEN, FINGER_THICK = 0.045, g.FINGER_THICK  # stock Franka fingers: TCP -> palm, pad thickness
TIP_TO_GRIP = tf.make_T(t=[0, 0, -FINGER_LEN / 2])  # fingertips (TCP) -> halfway up the fingers ("grip" frame)


def label(img, text):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (0, 0), (w + 12, h + 14), (255, 255, 255), -1)
    cv2.putText(img, text, (6, h + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def gripper(img, K, T_cam_world, T_world_tcp, width, open_w, color, thick=2):
    """Draw a parallel-jaw gripper: two fingers (closed to `width`), palm (open width), wrist line."""
    y0, y1 = width / 2 + FINGER_THICK / 2, open_w / 2 + FINGER_THICK / 2
    pts = np.array([[0, y0, 0], [0, y0, -FINGER_LEN], [0, -y0, 0], [0, -y0, -FINGER_LEN],   # fingers
                    [0, y1, -FINGER_LEN], [0, -y1, -FINGER_LEN], [0, 0, -FINGER_LEN], [0, 0, -0.1]])  # palm, wrist
    uv, z = tf.project(K, T_cam_world, tf.transform_points(T_world_tcp, pts))
    p = [tuple(int(v) for v in q) for q in uv]
    for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (1, 4), (3, 5)):
        cv2.line(img, p[a], p[b], (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.line(img, p[a], p[b], color, thick, cv2.LINE_AA)
    return p[7]  # wrist end, for numbering


def rpy(R):
    """Roll, pitch, yaw (deg): R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])


def pose_str(T):
    return 'xyz %7.1f %7.1f %7.1f mm | rpy %7.1f %7.1f %7.1f deg' % (*T[:3, 3] * 1000, *rpy(T[:3, :3]))


def axes(img, K, T_cam_x, length, name):
    """Draw a frame's x (red) / y (green) / z (blue) axes + its name; T_cam_x = that frame in the camera frame."""
    cv2.drawFrameAxes(img, K, None, cv2.Rodrigues(T_cam_x[:3, :3])[0], T_cam_x[:3, 3], length, 3)
    u, v = (K @ T_cam_x[:3, 3] / T_cam_x[2, 3])[:2].astype(int)
    for color, w in (((255, 255, 255), 4), ((0, 0, 0), 1)):
        cv2.putText(img, name, (u - 70, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, w, cv2.LINE_AA)


def box_edges(img, K, T_cam_world, T_world_obj, lo, hi, color):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    uv, _ = tf.project(K, T_cam_world, tf.transform_points(T_world_obj, c))
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:  # corners differ in exactly one axis = an edge
                cv2.line(img, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), color, 1, cv2.LINE_AA)


def main():
    cfg = load_config()
    gr, gc = cfg['gripper'], cfg['grasp']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--max-opening', type=float, default=gr['max_opening'])
    p.add_argument('--finger-depth', type=float, default=gr['finger_depth'])
    p.add_argument('--max-tilt', type=float, default=gc['max_approach_tilt_deg'])
    p.add_argument('--width-margin', type=float, default=gc['width_margin'])
    p.add_argument('--clearance', type=float, default=gc['table_clearance'])
    p.add_argument('--pose', default='play', help='stage 2 result output/pose/<TAG>.json')
    p.add_argument('--tag', default='play')
    a = p.parse_args()

    # inputs: image + K, stage 3 table frame (re-computed, ~0.6 s), stage 2 pose, mesh bounding box
    bgr = cv2.imread(os.path.join(SCENE, 'scene_rgb.png'))
    K = np.loadtxt(os.path.join(SCENE, 'camera_K.txt'))
    depth = cv2.imread(os.path.join(SCENE, 'scene_depth.png'), cv2.IMREAD_UNCHANGED) * 0.1 / 1000
    sc = cfg['spatial']
    T_wc = spatial.analyze_scene(depth, K, sc)['T_world_cam']
    T_cw = tf.invert(T_wc)
    res = json.load(open(os.path.join(REPO, 'output', 'pose', a.pose + '.json')))
    T_wo = T_wc @ np.array(res['ob_in_cam'])
    verts = read_obj_vertices(res['mesh'])
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ext = hi - lo

    # obstacles = scene points that RANSAC did not call table and that are not the bottle (its box + 1 cm)
    pts = tf.transform_points(T_wc, spatial.depth_to_points(depth, K))
    above = (pts[:, 2] > sc['height_thresh']) & (pts[:, 2] < sc['max_height'])
    in_obj = tf.transform_points(tf.invert(T_wo), pts)
    own = np.all((in_obj > lo - 0.01) & (in_obj < hi + 0.01), axis=1)
    obstacles = pts[above & ~own]
    print('\nscene points %d: table %d | bottle %d | obstacles %d (everything else above the table)'
          % (len(pts), (~above & (np.abs(pts[:, 2]) < sc['height_thresh'])).sum(), (above & own).sum(),
             len(obstacles)))

    # 1. candidates: 6 box faces x closing directions that fit in the gripper
    cands = g.generate_candidates(lo, hi, a.max_opening, a.width_margin, a.finger_depth)
    print('mesh box (object frame) %.0f x %.0f x %.0f mm | gripper opens %.0f mm, needs %.0f mm margin'
          % (*ext * 1000, a.max_opening * 1000, a.width_margin * 1000))
    for i, e in enumerate(ext):
        print('  close along %s (%.0f mm): %s' % ('xyz'[i], e * 1000, 'fits' if e + a.width_margin <= a.max_opening
                                                 else 'too wide -> no candidates'))

    # 2. filter each candidate against the table (stage 3 frame: table z = 0, up = +z)
    rows = []
    for c in cands:
        T, tilt, lowest, hits, why = g.evaluate(c, T_wo, 0.0, a.max_tilt, a.clearance, obstacles, a.max_opening,
                                                FINGER_LEN)
        rows.append((c, T, np.degrees(tilt), lowest, hits, why))
    rows.sort(key=lambda r: (bool(r[5]), r[2]))  # feasible first, then least tilted
    n_ok = sum(not r[5] for r in rows)
    print('\n%d candidates, %d feasible (max tilt %.0f deg, clearance %.0f mm, >= 10 scene points = hit)'
          % (len(rows), n_ok, a.max_tilt, a.clearance * 1000))
    print('%-3s %-7s %9s %9s %12s %6s  %-24s  %s' % ('#', 'face', 'width', 'tilt', 'lowest pt', 'hits', 'result',
                                                       'TCP xyz, table frame (mm)'))
    for i, (c, T, tilt, lowest, hits, why) in enumerate(rows):
        print('%-3d %-7s %6.0f mm %5.1f deg %9.0f mm %6d  %-24s  %6.0f %6.0f %6.0f'
              % (i, c.face, c.width * 1000, tilt, lowest * 1000, hits, why or ('BEST' if i == 0 else 'ok'),
                 *T[:3, 3] * 1000))

    # 3. poses: stage 2 gives object-in-camera; stage 3 gives table-in-camera; everything else is a product
    print('\nposes (camera frame: x right, y down, z forward | table frame: origin on the table, z up)')
    print('  object   in camera  %s' % pose_str(T_cw @ T_wo))
    print('  object   in table   %s' % pose_str(T_wo))
    if n_ok:
        T_best = rows[0][1] @ TIP_TO_GRIP
        print('  gripper  in camera  %s' % pose_str(T_cw @ T_best))
        print('  gripper  in table   %s' % pose_str(T_best))
        print('  gripper  in object  %s   (the candidate itself, before placing)'
              % pose_str(rows[0][0].T_obj_tcp @ TIP_TO_GRIP))
        print('  gripper frame: between the fingers, %.1f mm back from the tips | z = approach, y = finger closing'
              % (FINGER_LEN / 2 * 1000))

    # 4. draw: A = every candidate, numbered; B = best grasp + the box it was made from
    allc = bgr.copy()
    for i, (c, T, _, _, _, why) in reversed(list(enumerate(rows))):
        color = (0, 0, 255) if why else ((255, 0, 255) if i == 0 else (255, 255, 0))
        end = gripper(allc, K, T_cw, T, c.width, a.max_opening, color)
        cv2.putText(allc, str(i), (end[0] + 4, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    best = bgr.copy()
    box_edges(best, K, T_cw, T_wo, lo, hi, (0, 255, 0))
    if n_ok:
        c, T = rows[0][0], rows[0][1]
        gripper(best, K, T_cw, T, c.width, a.max_opening, (255, 0, 255), 3)
    tf_ = best.copy()
    axes(tf_, K, T_cw @ T_wo, 0.03, 'object')  # object frame at the mesh origin (box centre)
    if n_ok:
        axes(tf_, K, T_cw @ rows[0][1] @ TIP_TO_GRIP, 0.03, 'gripper')  # between the fingers
    out = np.hstack([label(allc, 'A  all %d candidates (magenta best, cyan ok, red rejected)' % len(rows)),
                     label(tf_, 'B  best grasp + mesh box + object & gripper axes')])
    path = os.path.join(REPO, 'output', 'grasp', a.tag + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out)
    print('\n-> ' + path)
    if 'VSCODE_IPC_HOOK_CLI' in os.environ:  # set in Cursor / VS Code terminals
        subprocess.run(['cursor', '--reuse-window', path])


if __name__ == '__main__':
    main()
