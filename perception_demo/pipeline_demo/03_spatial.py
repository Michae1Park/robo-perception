#!/usr/bin/env python3
"""Stage (3) demo: depth -> point cloud -> table-plane RANSAC -> place-zone occupancy grid,
using this repo's actual core/plane.py and core/free_space.py (pure numpy, no ROS).

SIMPLIFICATION vs. the real pipeline: the real scene_node.py works in the robot's calibrated
world frame (camera -> robot-base extrinsics from scene.yaml). We have no such calibration for
this stock dataset frame, so instead we build a "plane-aligned pseudo-world" frame here: fit the
dominant plane in raw camera coordinates, then re-express points with the plane normal as +Z and
two arbitrary in-plane axes as X/Y. table_z is therefore always 0 by construction in this frame --
that's a property of this demo's frame choice, not of the real pipeline. The 4x4 camera->pseudo
world transform is saved (03_T_world_cam.txt) so stage (4) can put the stage (2) object pose into
the same frame.

Run on the host (pure numpy/opencv/matplotlib, no GPU/torch needed):
  .venv/bin/python perception_demo/pipeline_demo/03_spatial.py
"""
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_perception'))
from ppp_perception.core.plane import fit_plane_ransac  # noqa: E402
from ppp_perception.core import free_space as fs  # noqa: E402

DATA = os.path.join(REPO, 'third_party/FoundationPose/demo_data/mustard0')
OUT = os.path.join(REPO, 'perception_demo/pipeline_demo/output')
RESOLUTION = 0.01


def depth_to_points(depth, K):
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth > 0.001
    z = depth[valid]
    x = (us[valid] - cx) * z / fx
    y = (vs[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def main():
    with open(os.path.join(OUT, '02_result.json')) as f:
        info = json.load(f)
    frame_id = info['frame_id']
    K = np.loadtxt(os.path.join(OUT, '02_cam_K.txt'))
    depth = cv2.imread(os.path.join(DATA, 'depth', frame_id + '.png'), cv2.IMREAD_UNCHANGED) / 1e3

    pts_cam = depth_to_points(depth, K)

    # Dominant plane in raw camera frame: disable the up/tilt filter (max_tilt_deg=180 makes the
    # `|dot(normal,up)| >= cos(max_tilt_deg)` check always pass) since we don't know the camera's
    # orientation relative to gravity for this stock frame.
    plane_cam, inliers = fit_plane_ransac(pts_cam, dist_thresh=0.006, iters=300,
                                          up=(0.0, 0.0, 1.0), max_tilt_deg=180, min_inliers=500)
    if plane_cam is None:
        raise SystemExit('RANSAC found no dominant plane')
    # Orient so the camera origin (height_above(plane, [0,0,0]) == plane[3]) is on the positive side.
    if plane_cam[3] < 0:
        plane_cam = -plane_cam

    n = plane_cam[:3]
    p0 = -plane_cam[3] * n  # a point on the plane
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    R = np.stack([u, v, n], axis=0)  # world_from_cam rotation
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R
    T_world_cam[:3, 3] = -R @ p0
    np.savetxt(os.path.join(OUT, '03_T_world_cam.txt'), T_world_cam)

    pts_world = (R @ (pts_cam - p0).T).T
    plane_world = np.array([0.0, 0.0, 1.0, 0.0])  # by construction

    table_pts = pts_world[np.abs(pts_world[:, 2]) < 0.01]
    if len(table_pts) < 100:
        raise SystemExit('too few table-height points to size a zone automatically')
    x0, x1 = np.percentile(table_pts[:, 0], [2, 98])
    y0, y1 = np.percentile(table_pts[:, 1], [2, 98])
    zone = ((float(x0), float(x1)), (float(y0), float(y1)))

    grid = fs.occupancy_grid(pts_world, plane_world, zone, resolution=RESOLUTION,
                             height_thresh=0.012, max_height=0.6)
    clearance = fs.clearance_map(grid, zone, RESOLUTION)

    with open(os.path.join(OUT, '03_result.json'), 'w') as f:
        json.dump({'plane_cam': plane_cam.tolist(), 'zone': zone, 'resolution': RESOLUTION,
                   'grid_shape': list(grid.shape), 'num_inliers_cam_frame': int(inliers.sum())},
                  f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rgb = cv2.cvtColor(cv2.imread(os.path.join(DATA, 'rgb', frame_id + '.png')), cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb)
    axes[0].set_title('scene frame %s' % frame_id)
    axes[0].axis('off')

    cmap = matplotlib.colors.ListedColormap(['#bbbbbb', '#2ecc71', '#e74c3c'])  # unknown/free/occupied
    disp = np.select([grid == fs.UNKNOWN, grid == fs.FREE, grid == fs.OCCUPIED], [0, 1, 2])
    axes[1].imshow(disp, cmap=cmap, origin='lower', extent=[x0, x1, y0, y1], vmin=0, vmax=2)
    axes[1].set_title('place-zone occupancy (grey=unknown, green=free, red=occupied)')
    axes[1].set_xlabel('x (pseudo-world, m)')
    axes[1].set_ylabel('y (pseudo-world, m)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '03_occupancy.png'), dpi=130)

    print('table plane (camera frame) [a,b,c,d]:', plane_cam)
    print('inliers: %d / %d points' % (int(inliers.sum()), len(pts_cam)))
    print('zone (pseudo-world m):', zone)
    print('grid shape:', grid.shape, ' free=%d occupied=%d unknown=%d' % (
        int((grid == fs.FREE).sum()), int((grid == fs.OCCUPIED).sum()), int((grid == fs.UNKNOWN).sum())))
    print('max clearance (m):', float(clearance.max()))
    print('-> ' + os.path.join(OUT, '03_occupancy.png'))


if __name__ == '__main__':
    main()
