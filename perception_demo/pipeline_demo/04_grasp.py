#!/usr/bin/env python3
"""Stage (4) demo: box-face grasp candidates from the mesh -> world (pseudo-world from stage 3)
via the stage (2) pose -> table/tilt filter, using this repo's actual core/grasp.py (pure numpy).

LIMITATIONS vs. the real grasp_node.py (documented, not hidden):
  1. Mesh mismatch: stage (2)'s pose was estimated against FoundationPose's OWN mustard-bottle
     mesh (demo_data/mustard0/mesh/textured_simple.obj), not this repo's assets/ycb/006_mustard_
     bottle mesh -- those are two different scans/origins. Using assets/ycb's box here would be
     geometrically inconsistent with the estimated pose's frame, so this script uses the SAME
     mesh FoundationPose registered against for the box-face candidates. In the real pipeline,
     pose and grasp geometry always come from the one YCB mesh fed to FoundationPose.
  2. No MoveIt / IK reachability filtering (no robot description wired up yet) -- candidates are
     only geometrically filtered (tilt, table clearance), not reachability-checked, so this stops
     one step short of grasp_node.py's final "best reachable" selection.
  3. table_z = 0.0 exactly, by construction of stage (3)'s plane-aligned pseudo-world frame.

Run on the host (pure numpy, no GPU/torch needed):
  .venv/bin/python perception_demo/pipeline_demo/04_grasp.py
"""
import json
import os
import sys

import cv2
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_common'))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_manipulation'))
from ppp_common import transforms as tf  # noqa: E402
from ppp_common.mesh import read_obj_vertices  # noqa: E402
from ppp_manipulation.core import grasp as g  # noqa: E402

DATA = os.path.join(REPO, 'third_party/FoundationPose/demo_data/mustard0')
OUT = os.path.join(REPO, 'perception_demo/pipeline_demo/output')
MESH = os.path.join(DATA, 'mesh/textured_simple.obj')


def main():
    with open(os.path.join(OUT, '02_result.json')) as f:
        info = json.load(f)
    cfg = yaml.safe_load(open(os.path.join(REPO, 'ros2/ppp_bringup/config/scene.yaml')))
    robot, gcfg = cfg['robot'], cfg['grasp']

    T_cam_obj = np.loadtxt(os.path.join(OUT, '02_pose_ob_in_cam.txt'))
    T_world_cam = np.loadtxt(os.path.join(OUT, '03_T_world_cam.txt'))
    T_world_obj = T_world_cam @ T_cam_obj
    table_z = 0.0  # by construction of stage 3's plane-aligned pseudo-world frame

    verts = read_obj_vertices(MESH)
    lo, hi = verts.min(axis=0), verts.max(axis=0)

    cands = g.generate_candidates(lo, hi, robot['max_opening'], gcfg['width_margin'], robot['finger_depth'])
    feasible = g.filter_candidates(cands, T_world_obj, table_z, gcfg['max_approach_tilt_deg'],
                                   gcfg['table_clearance'])

    # Per-candidate tilt/clearance, independent of pass/fail, for full transparency either way.
    DOWN = np.array([0.0, 0.0, -1.0])
    all_rows = []
    for c in cands:
        T = T_world_obj @ c.T_obj_tcp
        tilt = tf.angle_between(T[:3, 2], DOWN)
        all_rows.append({'face': c.face, 'tilt_deg': float(np.degrees(tilt)), 'width': float(c.width),
                         'T_world_tcp': T})
    all_rows.sort(key=lambda r: r['tilt_deg'])

    print('mesh extents (m):', (hi - lo).round(3))
    print('%d raw candidates, %d pass table/tilt filter (max_approach_tilt_deg=%s)' %
         (len(cands), len(feasible), gcfg['max_approach_tilt_deg']))
    for r in all_rows:
        print('  face=%-8s tilt=%6.1f deg  width=%.3f m' % (r['face'], r['tilt_deg'], r['width']))

    with open(os.path.join(OUT, '04_result.json'), 'w') as f:
        json.dump({'num_candidates': len(cands), 'num_feasible': len(feasible),
                   'max_approach_tilt_deg': gcfg['max_approach_tilt_deg'],
                   'all_candidates_by_tilt': [{'face': r['face'], 'tilt_deg': r['tilt_deg'],
                                               'width': r['width']} for r in all_rows]}, f, indent=2)

    K = np.loadtxt(os.path.join(OUT, '02_cam_K.txt'))
    T_cam_world = tf.invert(T_world_cam)
    rgb = cv2.imread(os.path.join(DATA, 'rgb', info['frame_id'] + '.png'))
    img = rgb.copy()
    arrow_len = 0.06

    def draw_grasp(T_world_tcp, width, color):
        # Two-finger parallel-jaw gripper: approach arrow (TCP's +z, into the object) plus the two
        # actual finger contact points, offset +/-width/2 along the TCP's closing axis (its y-axis,
        # per core/grasp.py's generate_candidates convention: R = [x, y, z], y = closing direction).
        tip_world = T_world_tcp[:3, 3]
        tail_world = tip_world - T_world_tcp[:3, 2] * arrow_len
        closing_world = T_world_tcp[:3, 1]
        finger0_world = tip_world + closing_world * (width / 2.0)
        finger1_world = tip_world - closing_world * (width / 2.0)
        pts_cam = tf.transform_points(T_cam_world,
                                      np.stack([tail_world, tip_world, finger0_world, finger1_world]))
        uv, z = tf.project(K, np.eye(4), pts_cam)
        if (z <= 0).any():
            return
        p0, p1, f0, f1 = uv.astype(int)
        # black outline first so the shapes read against any background colour, then the fill on top
        cv2.arrowedLine(img, tuple(p0), tuple(p1), (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(img, tuple(p0), tuple(p1), color, 3, cv2.LINE_AA, tipLength=0.35)
        cv2.line(img, tuple(f0), tuple(f1), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, tuple(f0), tuple(f1), color, 2, cv2.LINE_AA)
        for p in (f0, f1):
            cv2.circle(img, tuple(p), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 4, color, -1, cv2.LINE_AA)

    if feasible:
        for i, c in enumerate(feasible):
            draw_grasp(c.T_world_tcp, c.width, (255, 0, 255) if i == 0 else (255, 255, 0))
        label = 'grasp candidates: magenta=best, cyan=other feasible (%d/%d)' % (len(feasible), len(cands))
    else:
        # Nothing passed -- still show what was considered (red = rejected), and why.
        for r in all_rows:
            draw_grasp(r['T_world_tcp'], r['width'], (0, 0, 255))
        label = ('0/%d passed tilt<=%s deg filter (best=%.0f deg) -- object is lying on its side '
                 'in this frame, no safe top-down approach exists' %
                 (len(cands), gcfg['max_approach_tilt_deg'], all_rows[0]['tilt_deg']))
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, '04_grasp_candidates.png'), img)
    print('-> ' + os.path.join(OUT, '04_grasp_candidates.png'))


if __name__ == '__main__':
    main()
