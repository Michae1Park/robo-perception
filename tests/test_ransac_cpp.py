"""D-019: the C++ RANSAC (ppp_geometry, through its pybind11 module) matches vision/'s numpy version on recorded sim
frames: plane normal within 0.5 deg, offset within 2 mm, inlier count within 1%. Different random samples, so
results are compared, not bits.

Needs the ROS workspace built and sourced (source ros2/install/setup.bash) and a sim capture in data/sim/
(sim/scene.py); skipped otherwise.
"""
import os

import cv2
import numpy as np
import pytest

from vision import REPO, load_config, place, spatial

g = pytest.importorskip('ppp_geometry_py', reason='build + source ros2/ (ppp_geometry)')
SIM = os.path.join(REPO, 'data', 'sim')


def frame(sub):
    d = os.path.join(SIM, sub)
    if not os.path.exists(os.path.join(d, 'depth', '000000.png')):
        pytest.skip('no sim capture in data/sim/%s (run sim/scene.py)' % sub)
    depth = cv2.imread(os.path.join(d, 'depth', '000000.png'), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    return depth, np.loadtxt(os.path.join(d, 'cam_K.txt')), np.loadtxt(os.path.join(d, 'T_base_cam.txt'))


def assert_same(p_np, n_np, p_cpp, n_cpp):
    angle = np.degrees(np.arccos(np.clip(np.dot(p_np[:3], p_cpp[:3]), -1, 1)))
    assert angle < 0.5, 'normals differ by %.2f deg' % angle
    assert abs(p_np[3] - p_cpp[3]) < 0.002, 'offsets differ by %.1f mm' % (1000 * abs(p_np[3] - p_cpp[3]))
    assert abs(n_np - n_cpp) <= 0.01 * n_np, 'inliers %d vs %d' % (n_np, n_cpp)


def test_table_plane_matches_numpy():
    """Stage 3: the table in the fixed camera's view, in the base frame (z up)."""
    depth, K, T = frame('scene')
    cfg = load_config()['spatial']
    pts = spatial.depth_to_points(depth, K) @ T[:3, :3].T + T[:3, 3]
    kw = dict(dist_thresh=cfg['ransac_dist'], iters=cfg['ransac_iters'], up=(0, 0, 1), max_tilt_deg=10.0,
              min_inliers=cfg['min_inliers'])
    p_np, in_np = spatial.fit_plane_ransac(pts, seed=0, **kw)
    p_cpp, in_cpp = g.fit_plane_ransac(pts, seed=0, **{**kw, 'up': np.array([0.0, 0.0, 1.0])})
    assert p_np is not None and p_cpp is not None
    assert_same(p_np, in_np.sum(), p_cpp, in_cpp.sum())
    assert abs(p_cpp[3]) < 0.003   # the table top is z = 0 in the sim


def test_shelf_supports_match_numpy():
    """Stage 5: sequential RANSAC over the upward-facing points of the wrist camera's shelf shot."""
    depth, K, T = frame('shelf')
    cfg = load_config()['place']
    P, n, _, ok = place.points_and_normals(depth, K, T)
    planes_np = place.find_planes(P, n, ok, cfg)
    horiz = ok & (np.abs(n[..., 2]) > np.cos(np.radians(cfg['max_tilt_deg'])))
    pts = P[horiz]
    planes_cpp = g.find_planes(pts, dist_thresh=cfg['ransac_dist'], iters=cfg['ransac_iters'],
                               up=np.array([0.0, 0.0, 1.0]), max_tilt_deg=cfg['max_tilt_deg'],
                               min_inliers=cfg['min_inliers'], max_planes=cfg['max_planes'])
    z_np = sorted(z for z, _ in planes_np)
    z_cpp = sorted(float(pts[m][:, 2].mean()) for _, m in planes_cpp)
    assert len(z_np) == len(z_cpp), (z_np, z_cpp)
    counts_np = sorted(int(m.sum()) for _, m in planes_np)
    counts_cpp = sorted(int(m.sum()) for _, m in planes_cpp)
    for a, b in zip(z_np, z_cpp):
        assert abs(a - b) < 0.002, (z_np, z_cpp)
    for a, b in zip(counts_np, counts_cpp):
        assert abs(a - b) <= 0.01 * a, (counts_np, counts_cpp)
