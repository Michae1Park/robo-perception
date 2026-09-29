#!/usr/bin/env python3
"""Stage 3 - spatial: depth -> point cloud -> table-plane RANSAC -> place-zone occupancy grid.

Runs   : on the host (numpy only), .venv/bin/python
In     : output/02_result.json + 02_cam_K.txt; mustard0 depth in third_party/FoundationPose/demo_data/
Out    : output/03_occupancy.png, 03_T_world_cam.txt (camera -> plane-aligned frame), 03_result.json
Note   : the "world" frame is built from the table plane itself (normal = +Z, table height = 0).
"""
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config, spatial, viz  # noqa: E402

DATA = os.path.join(REPO, 'third_party', 'FoundationPose', 'demo_data', 'mustard0')
OUT = os.path.join(REPO, 'output')


def main():
    with open(os.path.join(OUT, '02_result.json')) as f:
        frame_id = json.load(f)['frame_id']
    K = np.loadtxt(os.path.join(OUT, '02_cam_K.txt'))
    depth = cv2.imread(os.path.join(DATA, 'depth', frame_id + '.png'), cv2.IMREAD_UNCHANGED) / 1e3
    rgb = cv2.imread(os.path.join(DATA, 'rgb', frame_id + '.png'))

    scene = spatial.analyze_scene(depth, K, load_config()['spatial'])
    grid = scene['grid']
    np.savetxt(os.path.join(OUT, '03_T_world_cam.txt'), scene['T_world_cam'])
    viz.plot_occupancy(rgb, scene, os.path.join(OUT, '03_occupancy.png'), 'scene frame %s' % frame_id)
    with open(os.path.join(OUT, '03_result.json'), 'w') as f:
        json.dump({'plane_cam': scene['plane_cam'].tolist(), 'zone': scene['zone'],
                   'grid_shape': list(grid.shape), 'num_inliers': int(scene['inliers'].sum())}, f, indent=2)

    print('table plane (camera frame) [a,b,c,d]:', scene['plane_cam'])
    print('inliers: %d / %d points' % (scene['inliers'].sum(), scene['n_points']))
    print('grid %s: free=%d occupied=%d unknown=%d' % (grid.shape, (grid == spatial.FREE).sum(),
          (grid == spatial.OCCUPIED).sum(), (grid == spatial.UNKNOWN).sum()))
    print('max clearance (m): %.3f' % scene['clearance'].max())


if __name__ == '__main__':
    main()
