"""Render one visual-prompt image per object: the object alone, seen from a different viewpoint
(side view, closer, different yaw) than the task camera, so the prompt is a genuine "example image"
rather than a crop of the test scene.

  ./python.sh isaac_sim/capture_prompts.py [--assets DIR]

Writes assets/prompts/<name>.png (crop) and <name>.json ({"bbox": [x1, y1, x2, y2]} inside the crop).
Any photo + box in the same format can replace these files.
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--scene-config', default='')
ap.add_argument('--assets', default='')
ap.add_argument('--margin', type=int, default=24, help='context pixels around the object box')
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_common import SimScene  # noqa: E402

from ppp_common import transforms as tf  # noqa: E402
from ppp_common.config import SceneConfig  # noqa: E402
from ppp_common.mesh import read_obj_vertices, subsample  # noqa: E402

W, H, F = 640, 480, 615.0


def main():
    cfg = SceneConfig(args.scene_config, args.assets)
    out_dir = os.path.join(cfg.assets, 'prompts')
    os.makedirs(out_dir, exist_ok=True)
    scene = SimScene(cfg, with_robot=False)
    center = np.array([0.52, 0.0])
    z_table = cfg.table_height
    eye = [center[0] - 0.05, center[1] + 0.40, z_table + 0.30]     # side view, ~0.5 m away
    target = [center[0], center[1], z_table + 0.04]
    cam = SimScene.add_camera('/World/PromptCamera', eye, target, W, H)
    scene.world.reset()
    cam.initialize()
    SimScene.configure_intrinsics(cam, W, H, F, F, W / 2.0, H / 2.0)
    K = tf.intrinsics_matrix(F, F, W / 2.0, H / 2.0)
    T_cam_world = tf.invert(tf.make_T(tf.look_at_ros(eye, target), eye))
    rng = np.random.default_rng(7)

    for name in cfg.object_names:
        scene.park_all()
        prim = scene.objects[name]
        SimScene.set_pose(prim, [center[0], center[1], z_table + scene.half_height[name] + 0.005],
                          rng.uniform(-np.pi, np.pi))
        for _ in range(90):              # settle + let the renderer converge
            scene.world.step(render=True)
        rgb = np.asarray(cam.get_rgba())[..., :3].astype(np.uint8)
        pts = subsample(read_obj_vertices(cfg.mesh_path(name)), 4000)
        uv, _ = tf.project(K, T_cam_world, tf.transform_points(scene.object_pose(name), pts))
        x1, y1 = np.floor(uv.min(axis=0)).astype(int)
        x2, y2 = np.ceil(uv.max(axis=0)).astype(int)
        cx1, cy1 = max(0, x1 - args.margin), max(0, y1 - args.margin)
        cx2, cy2 = min(W, x2 + args.margin), min(H, y2 + args.margin)
        crop = rgb[cy1:cy2, cx1:cx2]
        Image.fromarray(crop).save(os.path.join(out_dir, name + '.png'))
        bbox = [float(max(0, x1 - cx1)), float(max(0, y1 - cy1)),
                float(min(cx2, x2) - cx1), float(min(cy2, y2) - cy1)]
        with open(os.path.join(out_dir, name + '.json'), 'w') as f:
            json.dump({'bbox': bbox, 'source': 'isaac_sim side view'}, f)
        print('[ppp] prompt %s: crop %dx%d bbox %s' % (name, crop.shape[1], crop.shape[0], bbox))
    app.close()


if __name__ == '__main__':
    main()
