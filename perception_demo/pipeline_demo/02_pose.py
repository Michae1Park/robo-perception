#!/usr/bin/env python3
"""Stage (2) demo: FoundationPose 6D pose using the MASK STAGE (1) PRODUCED (not the dataset's
ground-truth mask) -- this is what makes (1)->(2) a real chain, not two disconnected demos.

Must run inside the FoundationPose container, cwd == this file's dir (same convention as their
own run_demo.py, so relative asset/extension loading behaves identically):
  docker start foundationpose
  docker exec -it foundationpose bash -lc \
    'cd /home/ai/workspace/prompt-pick-place/third_party/FoundationPose && python ppp_02_pose.py'
"""
import json
import os

import cv2
import numpy as np
import trimesh

from estimater import (FoundationPose, PoseRefinePredictor, ScorePredictor, draw_posed_3d_box,
                       draw_xyz_axis, dr, set_logging_format, set_seed)
from datareader import YcbineoatReader

REPO = '/home/ai/workspace/prompt-pick-place'
DATA = os.path.join(REPO, 'third_party/FoundationPose/demo_data/mustard0')
OUT = os.path.join(REPO, 'perception_demo/pipeline_demo/output')


def main():
    with open(os.path.join(OUT, '01_result.json')) as f:
        det = json.load(f)
    frame_idx = det['frame_index']
    mask = cv2.imread(os.path.join(OUT, '01_mask.png'), cv2.IMREAD_UNCHANGED) > 0

    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(os.path.join(DATA, 'mesh/textured_simple.obj'))
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    debug_dir = os.path.join(OUT, '02_fp_debug')
    os.makedirs(debug_dir, exist_ok=True)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=1, glctx=glctx)

    reader = YcbineoatReader(video_dir=DATA, shorter_side=None, zfar=np.inf)
    color = reader.get_color(frame_idx)
    depth = reader.get_depth(frame_idx)
    if mask.shape != (reader.H, reader.W):
        mask = cv2.resize(mask.astype(np.uint8), (reader.W, reader.H),
                          interpolation=cv2.INTER_NEAREST) > 0

    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
    np.savetxt(os.path.join(OUT, '02_pose_ob_in_cam.txt'), pose.reshape(4, 4))
    np.savetxt(os.path.join(OUT, '02_cam_K.txt'), reader.K)
    with open(os.path.join(OUT, '02_result.json'), 'w') as f:
        json.dump({'frame_index': frame_idx, 'frame_id': det['frame_id'],
                   'mesh_file': os.path.join(DATA, 'mesh/textured_simple.obj')}, f, indent=2)

    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3,
                        transparency=0, is_input_rgb=True)
    cv2.imwrite(os.path.join(OUT, '02_pose_overlay.png'), vis[..., ::-1])
    print('pose (ob_in_cam) ->', os.path.join(OUT, '02_pose_ob_in_cam.txt'))
    print(pose.reshape(4, 4))
    print('overlay ->', os.path.join(OUT, '02_pose_overlay.png'))


if __name__ == '__main__':
    main()
