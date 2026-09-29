"""Stage 2: 6-DoF pose with NVIDIA FoundationPose (single-frame registration).

Runs ONLY inside the `foundationpose` Docker container (needs its CUDA extensions).
FoundationPose's own code lives in third_party/FoundationPose and expects to run from that directory.
"""
import os

import numpy as np

from . import REPO

FP_DIR = os.path.join(REPO, 'third_party', 'FoundationPose')


def register(mesh_path, K, rgb, depth_m, mask, debug_dir, iterations=5):
    """rgb: HxWx3 RGB uint8, depth_m: HxW metres, mask: HxW bool.
    Returns (ob_in_cam 4x4, overlay RGB image with 3D box + axes)."""
    import sys
    sys.path.insert(0, FP_DIR)
    os.chdir(FP_DIR)
    import trimesh
    from estimater import (FoundationPose, PoseRefinePredictor, ScorePredictor, draw_posed_3d_box,
                           draw_xyz_axis, dr, set_logging_format, set_seed)

    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    os.makedirs(debug_dir, exist_ok=True)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=debug_dir,
                         debug=1, glctx=dr.RasterizeCudaContext())
    pose = est.register(K=K, rgb=rgb, depth=depth_m, ob_mask=mask, iteration=iterations).reshape(4, 4)

    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                        transparency=0, is_input_rgb=True)
    return pose, vis
