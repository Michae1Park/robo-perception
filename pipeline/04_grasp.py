#!/usr/bin/env python3
"""Stage 4 - grasp: box-face parallel-jaw candidates -> geometric filter (tilt + table clearance).

Runs   : on the host (numpy only), .venv/bin/python
In     : output/02_pose_ob_in_cam.txt, 02_cam_K.txt, 03_T_world_cam.txt; the mustard0 mesh
Out    : output/04_grasp_candidates.png, 04_result.json
Note   : uses the mesh FoundationPose registered against (not assets/ycb) so pose and geometry agree.
         No IK/reachability check - there is no robot in this repo.
"""
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import grasp as g, load_config, read_obj_vertices, viz  # noqa: E402

DATA = os.path.join(REPO, 'third_party', 'FoundationPose', 'demo_data', 'mustard0')
MESH = os.path.join(DATA, 'mesh', 'textured_simple.obj')
OUT = os.path.join(REPO, 'output')


def main():
    cfg = load_config()
    gripper, gcfg = cfg['gripper'], cfg['grasp']
    with open(os.path.join(OUT, '02_result.json')) as f:
        frame_id = json.load(f)['frame_id']
    K = np.loadtxt(os.path.join(OUT, '02_cam_K.txt'))
    T_world_cam = np.loadtxt(os.path.join(OUT, '03_T_world_cam.txt'))
    T_world_obj = T_world_cam @ np.loadtxt(os.path.join(OUT, '02_pose_ob_in_cam.txt'))

    verts = read_obj_vertices(MESH)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    cands = g.generate_candidates(lo, hi, gripper['max_opening'], gcfg['width_margin'], gripper['finger_depth'])
    feasible = g.filter_candidates(cands, T_world_obj, 0.0, gcfg['max_approach_tilt_deg'],  # table_z = 0
                                   gcfg['table_clearance'])
    rows = viz.grasp_rows(cands, T_world_obj)

    print('mesh extents (m):', (hi - lo).round(3))
    print('%d candidates, %d feasible (max tilt %s deg)' % (len(cands), len(feasible),
                                                           gcfg['max_approach_tilt_deg']))
    for r in rows:
        print('  face=%-8s tilt=%6.1f deg  width=%.3f m' % (r['face'], r['tilt_deg'], r['width']))
    with open(os.path.join(OUT, '04_result.json'), 'w') as f:
        json.dump({'num_candidates': len(cands), 'num_feasible': len(feasible),
                   'max_approach_tilt_deg': gcfg['max_approach_tilt_deg'],
                   'all_candidates_by_tilt': [{k: r[k] for k in ('face', 'tilt_deg', 'width')} for r in rows]},
                  f, indent=2)

    rgb = cv2.imread(os.path.join(DATA, 'rgb', frame_id + '.png'))
    cv2.imwrite(os.path.join(OUT, '04_grasp_candidates.png'),
                viz.draw_grasps(rgb, K, T_world_cam, feasible, rows))


if __name__ == '__main__':
    main()
