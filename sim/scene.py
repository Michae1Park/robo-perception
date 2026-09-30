#!/usr/bin/env python3
"""Isaac Sim cell: table + YCB objects + Franka + shelf + RGB-D camera. Writes one dataset per target object in
the FoundationPose demo_data layout (same as mustard0), plus ground-truth poses and camera extrinsics.

  .venv-sim/bin/python sim/scene.py                   # 1 frame  -> data/sim/
  .venv-sim/bin/python sim/scene.py --frames 10       # 10 frames, new random yaws each
  .venv-sim/bin/python sim/scene.py --livestream      # view only (no capture): WebRTC Streaming Client on the laptop

Output (<out>/, default data/sim/):
  scene/     rgb/000000.png  depth/000000.png (uint16 mm)  cam_K.txt  T_base_cam.txt (GT extrinsics)
  shelf/     same, from the wrist camera with the arm in its shelf-looking pose (stage 5), + shelf.json (GT board heights)
  <target>/  masks/000000.png (visible pixels = 255)  ob_in_cam/000000.txt (GT 4x4 pose, OpenCV camera frame)
             mesh -> assets/ycb/<target>/ (the OBJ FoundationPose uses); rgb, depth, cam_K.txt -> ../scene/

Scene parameters: config.yaml `sim:`. The same cell live over ROS 2: sim/ros_cell.py.
"""
import argparse
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--frames', type=int, default=1, help='frames to capture (new random yaws each)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=None, help='default: data/sim')
ap.add_argument('--livestream', action='store_true', help='view only: stream the viewport over WebRTC, no capture')
ap.add_argument('--overview', metavar='PNG', help='render one high-res still of the whole cell (config.yaml sim.overview; '
                'arm in its shelf-looking pose) to PNG and exit')
ap.add_argument('--arm', choices=['home', 'look'], default='home',
                help='--livestream only: arm pose to show (look = the wrist camera looking into the shelf)')
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (must be created before any other omni/isaacsim import)

app = SimulationApp({'headless': True, 'hide_ui': not args.livestream, 'width': 1280, 'height': 720})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'sim'))
from cell import (YCB, add_camera, add_camera_rig, add_wrist_camera, build_scene, drop_all,  # noqa: E402
                  move_arm, place_on_shelf, reset_robot, set_viewport, settle, start_livestream, world_pose_cv)
from pxr import Gf  # noqa: E402
from vision import load_config  # noqa: E402
from vision import transforms as tf  # noqa: E402


def prepare_shelf_output(root, K, shelf_cfg):
    """Fresh <root>/shelf/: the wrist camera's images + GT shelf geometry (to score stage 5). Extrinsics per frame."""
    d = os.path.join(root, 'shelf')
    shutil.rmtree(d, ignore_errors=True)
    for sub in ('rgb', 'depth'):
        os.makedirs(os.path.join(d, sub))
    np.savetxt(os.path.join(d, 'cam_K.txt'), K)
    with open(os.path.join(d, 'shelf.json'), 'w') as f:
        json.dump({k: shelf_cfg[k] for k in ('center_xy', 'size', 'levels_z', 'board')}, f, indent=2)
    return d


def prepare_output(root, targets, K, T_world_cam):
    """Fresh <root>/scene/ plus one FoundationPose-layout folder per target, sharing the scene's images."""
    scene = os.path.join(root, 'scene')
    for d in [scene] + [os.path.join(root, n) for n in targets]:
        shutil.rmtree(d, ignore_errors=True)
    for d in ('rgb', 'depth'):
        os.makedirs(os.path.join(scene, d))
    np.savetxt(os.path.join(scene, 'cam_K.txt'), K)
    # ground-truth extrinsics: camera (OpenCV frame) in the robot base frame; base = world origin
    np.savetxt(os.path.join(scene, 'T_base_cam.txt'), T_world_cam)
    for n in targets:
        d = os.path.join(root, n)
        os.makedirs(os.path.join(d, 'masks'))
        os.makedirs(os.path.join(d, 'ob_in_cam'))
        for f in ('rgb', 'depth', 'cam_K.txt', 'T_base_cam.txt'):
            os.symlink(os.path.join('..', 'scene', f), os.path.join(d, f))
        os.symlink(os.path.relpath(os.path.join(YCB, n), d), os.path.join(d, 'mesh'))
    return scene


def render_overview(world, franka, objects, shelf_objects, cfg, rng, look):
    """One high-res still of the cell for the docs: items dropped, arm looking at the shelf (wrist D455 visible)."""
    o = cfg['overview']
    w, h = o['size']
    fx = w / 2 / np.tan(np.radians(o['hfov_deg']) / 2)
    path, _, _ = add_camera({'width': w, 'height': h, 'fx': fx, 'eye': o['eye'], 'target': o['target']},
                            '/World/OverviewCamera')
    annot = rep.AnnotatorRegistry.get_annotator('rgb')
    annot.attach(rep.create.render_product(path, (w, h)))
    drop_all(objects, rng)
    place_on_shelf(shelf_objects, cfg['shelf'])
    move_arm(franka, np.array(look))
    settle(world, cfg['settle_steps'] + o.get('extra_steps', 0))   # extra frames let the renderer converge
    cv2.imwrite(args.overview, np.asarray(annot.get_data())[..., 2::-1])
    print('[sim] overview -> %s (%dx%d)' % (args.overview, w, h))


def stream(world, cfg):
    """View-only: the viewport (not the RGB-D sensor) is what the WebRTC client shows. Replicator annotators
    never return data while streaming, so --livestream does not capture (I-002)."""
    set_viewport(cfg)
    print('[sim] streaming on TCP 49100 / UDP 47998 - connect the Isaac Sim WebRTC Streaming Client '
          'to this machine\'s IP. Ctrl+C to quit.', flush=True)
    while app.is_running():
        world.step(render=True)


def main():
    if args.livestream:
        start_livestream()
    full = load_config()
    cfg = full['sim']
    world, franka, objects, shelf_objects = build_scene(cfg, full['gripper'])
    targets = [o['name'] for o, _, _ in objects if o.get('target')]
    cam_cfg = cfg['camera']
    cam_path, K, T_world_cam = add_camera(cam_cfg)
    add_camera_rig(T_world_cam, -cfg['table_size'][2], cam_cfg['rig'])

    wcfg = cfg['wrist_camera']
    shelf_path = add_wrist_camera(cam_cfg, wcfg)   # on the robot's wrist (stage 5); follows the arm

    if not (args.livestream or args.overview):
        root = args.out or os.path.join(REPO, 'data', 'sim')
        scene = prepare_output(root, targets, K, T_world_cam)
        rp = rep.create.render_product(cam_path, (cam_cfg['width'], cam_cfg['height']))
        annot = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ('rgb', 'distance_to_image_plane')}
        annot['seg'] = rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False})
        for a in annot.values():
            a.attach(rp)
        shelf_dir = prepare_shelf_output(root, K, cfg['shelf'])
        rp2 = rep.create.render_product(shelf_path, (cam_cfg['width'], cam_cfg['height']))
        annot2 = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ('rgb', 'distance_to_image_plane')}
        for a in annot2.values():
            a.attach(rp2)

    robot = full['robot']
    reset_robot(world, franka, robot['home_joints'])
    rng = np.random.default_rng(args.seed)

    if args.overview:
        render_overview(world, franka, objects, shelf_objects, cfg, rng, robot['look_joints'])
        app.close()
        return

    if args.livestream:
        drop_all(objects, rng)
        place_on_shelf(shelf_objects, cfg['shelf'])
        if args.arm == 'look':
            move_arm(franka, np.array(robot['look_joints']))
        stream(world, cfg)
        app.close()
        return

    prims = {o['name']: prim for o, prim, _ in objects}
    T_cam_world = tf.invert(T_world_cam)
    for i in range(args.frames):
        move_arm(franka, np.array(robot['home_joints']))   # table shot: arm at home
        yaws = drop_all(objects, rng)
        place_on_shelf(shelf_objects, cfg['shelf'])
        settle(world, cfg['settle_steps'])

        fid = '%06d' % i
        rgb = np.asarray(annot['rgb'].get_data())[..., :3]
        depth = np.asarray(annot['distance_to_image_plane'].get_data(), dtype=np.float32)
        depth[~np.isfinite(depth)] = 0
        cv2.imwrite(os.path.join(scene, 'rgb', fid + '.png'), rgb[..., ::-1])
        cv2.imwrite(os.path.join(scene, 'depth', fid + '.png'), np.round(depth * 1000).astype(np.uint16))

        seg = annot['seg'].get_data()
        labels = seg['info']['idToLabels']
        for n in targets:
            ids = [int(k) for k, v in labels.items() if v.get('class') == n]
            mask = np.isin(seg['data'], ids)
            p, q = prims[n].get_world_pose()
            ob_in_cam = T_cam_world @ tf.make_T(np.array(Gf.Matrix3d(Gf.Quatd(*map(float, q)))).T, p)
            cv2.imwrite(os.path.join(root, n, 'masks', fid + '.png'), mask.astype(np.uint8) * 255)
            np.savetxt(os.path.join(root, n, 'ob_in_cam', fid + '.txt'), ob_in_cam)
            print('[sim] frame %s %-20s yaw %6.1f deg, %.3f m from camera, mask %5d px'
                  % (fid, n, yaws[n], np.linalg.norm(ob_in_cam[:3, 3]), mask.sum()))

        move_arm(franka, np.array(robot['look_joints']))   # shelf shot: arm in its looking pose
        settle(world, cfg['settle_steps'])
        T_base_wrist = world_pose_cv(shelf_path)            # base = world origin
        np.savetxt(os.path.join(shelf_dir, 'T_base_cam.txt'), T_base_wrist)
        print('[sim] frame %s wrist camera at %s, looking %s' % (fid, np.round(T_base_wrist[:3, 3], 3),
                                                                 np.round(T_base_wrist[:3, 2], 2)))
        rgb2 = np.asarray(annot2['rgb'].get_data())[..., :3]
        depth2 = np.asarray(annot2['distance_to_image_plane'].get_data(), dtype=np.float32)
        depth2[~np.isfinite(depth2)] = 0
        cv2.imwrite(os.path.join(shelf_dir, 'rgb', fid + '.png'), rgb2[..., ::-1])
        cv2.imwrite(os.path.join(shelf_dir, 'depth', fid + '.png'), np.round(depth2 * 1000).astype(np.uint16))

    print('[sim] wrote %s: scene/ + shelf/ + %s' % (root, ', '.join(targets)))
    app.close()


if __name__ == '__main__':
    main()
