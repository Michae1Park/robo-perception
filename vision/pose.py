"""Stage 2: 6-DoF pose with NVIDIA FoundationPose (single-frame registration).

Runs ONLY inside the `foundationpose` Docker container (needs its CUDA extensions).
FoundationPose's own code lives in third_party/FoundationPose and expects to run from that directory.
"""
import os

import numpy as np

from . import REPO

FP_DIR = os.path.join(REPO, 'third_party', 'FoundationPose')


def _fp():
    """Import FoundationPose's estimater module (it must be imported from its own directory)."""
    import sys
    sys.path.insert(0, FP_DIR)
    os.chdir(FP_DIR)
    import estimater
    return estimater


def build_estimator(mesh_path, debug_dir, n_views=40, inplane_step=60, quiet=False):
    """Load the mesh + refiner + scorer networks. Rotation hypotheses = n_views icosphere viewpoints
    x (360 / inplane_step) in-plane rotations, then clustered (30 deg). Returns (est, mesh)."""
    import logging

    import trimesh
    fp = _fp()
    fp.set_logging_format()
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    fp.set_seed(0)
    mesh = trimesh.load(mesh_path)
    os.makedirs(debug_dir, exist_ok=True)
    est = fp.FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                            scorer=fp.ScorePredictor(), refiner=fp.PoseRefinePredictor(), debug_dir=debug_dir,
                            debug=1, glctx=fp.dr.RasterizeCudaContext())
    if (n_views, inplane_step) != (40, 60):  # the constructor already built the default grid
        est.make_rotation_grid(min_n_views=n_views, inplane_step=inplane_step)
    return est, mesh


def draw_pose(K, rgb, ob_in_cam, mesh):
    """Overlay the mesh's oriented 3D box + xyz axes (x red, y green, z blue) on an RGB image."""
    import trimesh
    fp = _fp()
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    center_pose = ob_in_cam @ np.linalg.inv(to_origin)
    vis = fp.draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
    return fp.draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                            transparency=0, is_input_rgb=True)


def register(mesh_path, K, rgb, depth_m, mask, debug_dir, iterations=5):
    """rgb: HxWx3 RGB uint8, depth_m: HxW metres, mask: HxW bool.
    Returns (ob_in_cam 4x4, overlay RGB image with 3D box + axes)."""
    est, mesh = build_estimator(mesh_path, debug_dir)
    pose = est.register(K=K, rgb=rgb, depth=depth_m, ob_mask=mask, iteration=iterations).reshape(4, 4)
    return pose, draw_pose(K, rgb, pose, mesh)
