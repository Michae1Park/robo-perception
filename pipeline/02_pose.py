#!/usr/bin/env python3
"""Stage 2 - 6-DoF pose: FoundationPose registration using stage 1's mask.

Runs   : inside the `foundationpose` container (needs FoundationPose's CUDA extensions).
In     : output/01_result.json + 01_mask.png; mustard0 RGB-D + mesh in third_party/FoundationPose/demo_data/
Out    : output/02_pose_ob_in_cam.txt (4x4), 02_cam_K.txt, 02_pose_overlay.png, 02_result.json
"""
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config, pose  # noqa: E402

DATA = os.path.join(REPO, 'third_party', 'FoundationPose', 'demo_data', 'mustard0')
MESH = os.path.join(DATA, 'mesh', 'textured_simple.obj')
OUT = os.path.join(REPO, 'output')


def main():
    with open(os.path.join(OUT, '01_result.json')) as f:
        det = json.load(f)
    mask = cv2.imread(os.path.join(OUT, '01_mask.png'), cv2.IMREAD_UNCHANGED) > 0

    sys.path.insert(0, pose.FP_DIR)
    from datareader import YcbineoatReader  # FoundationPose's own dataset reader
    reader = YcbineoatReader(video_dir=DATA, shorter_side=None, zfar=np.inf)
    color, depth = reader.get_color(det['frame_index']), reader.get_depth(det['frame_index'])
    if mask.shape != (reader.H, reader.W):
        mask = cv2.resize(mask.astype(np.uint8), (reader.W, reader.H), interpolation=cv2.INTER_NEAREST) > 0

    ob_in_cam, vis = pose.register(MESH, reader.K, color, depth, mask,
                                   os.path.join(OUT, '02_fp_debug'), load_config()['pose']['iterations'])
    np.savetxt(os.path.join(OUT, '02_pose_ob_in_cam.txt'), ob_in_cam)
    np.savetxt(os.path.join(OUT, '02_cam_K.txt'), reader.K)
    cv2.imwrite(os.path.join(OUT, '02_pose_overlay.png'), vis[..., ::-1])
    with open(os.path.join(OUT, '02_result.json'), 'w') as f:
        json.dump({'frame_index': det['frame_index'], 'frame_id': det['frame_id'], 'mesh_file': MESH}, f, indent=2)
    print(ob_in_cam)


if __name__ == '__main__':
    main()
