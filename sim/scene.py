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

Scene parameters: config.yaml `sim:`. No ROS here: the ROS 2 bridge comes later.
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
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (must be created before any other omni/isaacsim import)

app = SimulationApp({'headless': True, 'hide_ui': not args.livestream, 'width': 1280, 'height': 720})
if args.livestream:
    # WebRTC stream of the viewport (signal TCP 49100, media UDP 47998)
    import carb
    from isaacsim.core.utils.extensions import enable_extension
    # no NvStreamer-*.etli event-trace files in the working directory
    carb.settings.get_settings().set('/exts/omni.kit.livestream.app/primaryStream/enableEventTracing', False)
    enable_extension('omni.kit.livestream.app')
    print('[sim] livestream extension enabled', flush=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCylinder, VisualSphere  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Semantics, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'sim'))
from vision import load_config  # noqa: E402
from vision import transforms as tf  # noqa: E402
from vision.grasp import read_obj_vertices  # noqa: E402
from ycb_usd import YCB, ensure_object_usd, prim_name  # noqa: E402

# USD cameras look down -z with +y up; OpenCV/ROS optical frames look down +z with +y down.
USD_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])
FRANKA_USD = '/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'   # under the NVIDIA asset root
RS_USD = '/Isaac/Sensors/Intel/RealSense/rsd455.usd'   # RealSense D455 model (body only; its own sensors unused)
# D455 asset frame: +x forward (front glass at x = 0), +y left, +z up; colour lens at y = -11.5 mm.
# T_cam_rs places it so the colour lens sits on our (OpenCV) camera's optical centre.
T_CAM_RS = np.array([[0.0, -1.0, 0.0, -0.0115], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
RS_MOUNT = np.array([-0.013, 0.0, -0.0145])   # 1/4" tripod thread under the body (asset frame)
FRANKA_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])  # standard home pose, gripper open


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World pose (4x4) of an OpenCV camera at `eye` looking at `target`."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return tf.make_T(np.stack([x, np.cross(z, x), z], axis=1), eye)


def quat_wxyz(R):
    q = Gf.Matrix3d(*R.T.flatten()).ExtractRotation().GetQuat()   # Gf is row-vector: pass R^T
    return np.array([q.GetReal(), *q.GetImaginary()])


def linear(srgb):
    """config.yaml colours are sRGB (as in a colour picker); USD material colours are linear."""
    return np.asarray(srgb, float) ** 2.2


def box(path, center, size, color):
    return FixedCuboid(path, position=np.array(center, float), scale=np.array(size, float), size=1.0,
                       color=linear(color))


def add_shelf(cfg, floor_z, color):
    """Open shelf: side panels + back panel + one board per level, opening facing the robot (+y)."""
    (cx, cy), (w, d), t = cfg['center_xy'], cfg['size'], cfg['board']
    top = max(cfg['levels_z'])
    h = top - floor_z
    for side, x in (('left', cx - w / 2 + t / 2), ('right', cx + w / 2 - t / 2)):
        box('/World/Shelf/' + side, [x, cy, floor_z + h / 2], [t, d, h], color)
    box('/World/Shelf/back', [cx, cy - d / 2 + t / 2, floor_z + h / 2], [w, t, h], color)
    for k, z in enumerate(cfg['levels_z']):
        box('/World/Shelf/board_%d' % k, [cx, cy, z - t / 2], [w, d, t], color)


def add_ycb(world, name, label, path=None):
    """YCB object as a dynamic rigid body with a convex-hull collider. -> (SingleRigidPrim, mesh vertices)."""
    stage = get_current_stage()
    path = path or '/World/Objects/' + prim_name(name)
    add_reference_to_stage(ensure_object_usd(name), path)
    root = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.3)
    geom = stage.GetPrimAtPath(path + '/geom')
    UsdPhysics.CollisionAPI.Apply(geom)
    UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr('convexHull')
    sem = Semantics.SemanticsAPI.Apply(root, 'Semantics')
    sem.CreateSemanticTypeAttr('class')
    sem.CreateSemanticDataAttr(label)
    verts = read_obj_vertices(os.path.join(YCB, name, 'textured.obj'))
    return world.scene.add(SingleRigidPrim(path, name=label)), verts


def build_scene(cfg):
    world = World(stage_units_in_meters=1.0, physics_dt=cfg['physics_dt'], rendering_dt=cfg['physics_dt'])
    stage = get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    col = cfg['colors']
    tw, tl, th = cfg['table_size']
    GroundPlane('/World/Ground', z_position=-th, size=20.0, color=linear(col['floor']))
    box('/World/Table', [*cfg['table_center'], -th / 2], [tw, tl, th], col['table'])
    box('/World/Pedestal', [0.0, 0.0, -th / 2], [0.2, 0.2, th], col['pedestal'])
    add_shelf(cfg['shelf'], -th, col['shelf'])

    dome = UsdLux.DomeLight.Define(stage, '/World/Lights/dome')
    dome.CreateIntensityAttr(250.0)
    dome.CreateColorAttr(Gf.Vec3f(*linear(col['backdrop'])))
    sun = UsdLux.DistantLight.Define(stage, '/World/Lights/sun')
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 20.0, 0.0))

    add_reference_to_stage(get_assets_root_path() + FRANKA_USD, '/World/Franka')
    franka = world.scene.add(SingleArticulation('/World/Franka', name='franka'))

    objects = [(o, *add_ycb(world, o['name'], o['name'])) for o in cfg['objects']]   # (config, prim, vertices)
    shelf_objects = [(o, *add_ycb(world, o['name'], 'shelf_' + o['name'], '/World/ShelfObjects/' + prim_name(o['name'])))
                     for o in cfg.get('shelf_objects', [])]
    return world, franka, objects, shelf_objects


def quat_z_to(v):
    """wxyz quaternion rotating +z onto direction v (cylinders are built along z)."""
    z = np.asarray(v, float) / np.linalg.norm(v)
    x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return quat_wxyz(np.stack([x, np.cross(z, x), z], axis=1))


def rod(path, a, b, radius, color):
    """Visual-only cylinder from point a to point b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return VisualCylinder(path, position=(a + b) / 2, orientation=quat_z_to(b - a), radius=radius,
                          height=float(np.linalg.norm(b - a)), color=linear(color))


def add_camera_rig(T_world_cam, floor_z, rig_cfg):
    """RealSense D455 body on a ball head + floor tripod, all visual only (no collisions, no physics).
    Everything sits behind / below the optical centre, so the camera never sees its own rig."""
    stage = get_current_stage()
    T_world_rs = T_world_cam @ T_CAM_RS
    add_reference_to_stage(get_assets_root_path() + RS_USD, '/World/CameraRig/D455')
    body = stage.GetPrimAtPath('/World/CameraRig/D455/RSD455')
    UsdPhysics.RigidBodyAPI(body).CreateRigidBodyEnabledAttr(False)    # the asset is a rigid body; keep it put
    xf = UsdGeom.Xformable(stage.GetPrimAtPath('/World/CameraRig/D455'))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*T_world_rs.T.flatten()))

    black, metal = rig_cfg['color'], rig_cfg['metal_color']
    mount = tf.transform_points(T_world_rs, RS_MOUNT[None])[0]
    down = -T_world_rs[:3, 2]                                          # the body's "down" (tilted with the camera)
    plate = mount + 0.012 * down
    rod('/World/CameraRig/plate', mount, plate, 0.018, black)          # quick-release plate
    ball = plate + np.array([0.0, 0.0, -0.022])
    VisualSphere('/World/CameraRig/ball_head', position=ball, radius=0.022, color=linear(black))
    hub = np.array([ball[0], ball[1], floor_z + rig_cfg['hub_height']])
    rod('/World/CameraRig/column', ball, hub, 0.011, metal)            # centre column
    rod('/World/CameraRig/collar', hub + [0, 0, -0.03], hub + [0, 0, 0.05], 0.022, black)
    for k, ang in enumerate(np.deg2rad(rig_cfg['leg_angles_deg'])):
        foot = hub + [rig_cfg['leg_spread'] * np.cos(ang), rig_cfg['leg_spread'] * np.sin(ang), 0.0]
        foot[2] = floor_z + 0.015
        rod('/World/CameraRig/leg_%d' % k, hub, foot, 0.010, black)
        rod('/World/CameraRig/foot_%d' % k, foot + [0, 0, -0.015], foot + [0, 0, 0.005], 0.016, black)


def add_camera(cam_cfg, path='/World/Camera'):
    """USD camera with pinhole intrinsics K and the configured pose. -> (prim path, K, T_world_cam)."""
    w, h, f = cam_cfg['width'], cam_cfg['height'], cam_cfg['fx']
    cam = UsdGeom.Camera.Define(get_current_stage(), path)
    ha = 20.955                                   # horizontal aperture (mm); focal length follows from fx
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha * h / w)
    cam.CreateFocalLengthAttr(f * ha / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    T_world_cam = look_at(cam_cfg['eye'], cam_cfg['target'])
    T_usd = T_world_cam @ USD_FROM_CV
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*T_usd.T.flatten()))
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    return path, K, T_world_cam


def settle(world, steps):
    for _ in range(steps):                           # annotators update on every rendered step; the object is at
        world.step(render=True)                      # rest by the end


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def drop(obj, xy, verts, R, gap):
    """Place `obj` with orientation R, its lowest point `gap` above the table top, at rest."""
    z = -(verts @ R.T)[:, 2].min() + gap
    obj.set_world_pose(position=np.array([*xy, z]), orientation=quat_wxyz(R))
    obj.set_linear_velocity(np.zeros(3))
    obj.set_angular_velocity(np.zeros(3))


def drop_all(objects, rng):
    """Every object at its spot. yaw_deg [lo, hi] draws a new yaw per call. `tilt_deg` tips the item about its own
    x axis first (90 = lying on its side); tipped items are set down gently (1 mm) so they don't roll.
    -> {name: yaw in degrees}"""
    yaws = {}
    for o, prim, verts in objects:
        y = o.get('yaw_deg', 0.0)
        yaws[o['name']] = rng.uniform(*y) if isinstance(y, list) else float(y)
        tilt = np.deg2rad(o.get('tilt_deg', 0.0))
        drop(prim, o['xy'], verts, tf.rot_z(np.deg2rad(yaws[o['name']])) @ rot_x(tilt), 0.001 if tilt else 0.01)
    return yaws


def add_wrist_camera(cam_cfg, mount_xyz):
    """Camera on the Franka hand (moves with the arm), looking along the hand's +z. -> prim path."""
    w, h, f = cam_cfg['width'], cam_cfg['height'], cam_cfg['fx']
    path = '/World/Franka/panda_hand/WristCamera'
    cam = UsdGeom.Camera.Define(get_current_stage(), path)
    ha = 20.955
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha * h / w)
    cam.CreateFocalLengthAttr(f * ha / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    T_hand_usd = tf.make_T(t=mount_xyz) @ USD_FROM_CV     # OpenCV camera axes = hand axes
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*T_hand_usd.T.flatten()))
    return path


def world_pose_cv(path):
    """Current world pose of a USD camera prim, as an OpenCV camera (4x4)."""
    m = UsdGeom.Xformable(get_current_stage().GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m).T @ USD_FROM_CV


def move_arm(franka, q7):
    q = np.concatenate([q7, FRANKA_READY[7:]])
    franka.set_joint_positions(q)
    franka.apply_action(ArticulationAction(joint_positions=q))


def place_on_shelf(shelf_objects, shelf_cfg):
    """Stand each shelf item upright on its board, at x along the shelf width, centred front-to-back."""
    (cx, cy) = shelf_cfg['center_xy']
    for o, prim, verts in shelf_objects:
        R = tf.rot_z(np.deg2rad(o.get('yaw_deg', 0.0)))
        z = shelf_cfg['levels_z'][o['level']] - (verts @ R.T)[:, 2].min() + 0.005
        prim.set_world_pose(position=np.array([cx + o['x'], cy, z]), orientation=quat_wxyz(R))
        prim.set_linear_velocity(np.zeros(3))
        prim.set_angular_velocity(np.zeros(3))


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


def stream(world):
    """View-only: the viewport (not the RGB-D sensor) is what the WebRTC client shows. Replicator annotators
    never return data while streaming, so --livestream does not capture."""
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array([1.95, -0.25, 1.30]), target=np.array([0.30, -0.08, 0.10]))   # table, robot, shelf, rig
    print('[sim] streaming on TCP 49100 / UDP 47998 - connect the Isaac Sim WebRTC Streaming Client '
          'to this machine\'s IP. Ctrl+C to quit.', flush=True)
    while app.is_running():
        world.step(render=True)


def main():
    cfg = load_config()['sim']
    world, franka, objects, shelf_objects = build_scene(cfg)
    targets = [o['name'] for o, _, _ in objects if o.get('target')]
    cam_cfg = cfg['camera']
    cam_path, K, T_world_cam = add_camera(cam_cfg)
    add_camera_rig(T_world_cam, -cfg['table_size'][2], cam_cfg['rig'])

    if not args.livestream:
        root = args.out or os.path.join(REPO, 'data', 'sim')
        scene = prepare_output(root, targets, K, T_world_cam)
        rp = rep.create.render_product(cam_path, (cam_cfg['width'], cam_cfg['height']))
        annot = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ('rgb', 'distance_to_image_plane')}
        annot['seg'] = rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False})
        for a in annot.values():
            a.attach(rp)
        # 2nd camera: on the robot's wrist (stage 5). Same intrinsics; its pose follows the arm
        wcfg = cfg['wrist_camera']
        shelf_path = add_wrist_camera(cam_cfg, wcfg['mount_xyz'])
        shelf_dir = prepare_shelf_output(root, K, cfg['shelf'])
        rp2 = rep.create.render_product(shelf_path, (cam_cfg['width'], cam_cfg['height']))
        annot2 = {k: rep.AnnotatorRegistry.get_annotator(k) for k in ('rgb', 'distance_to_image_plane')}
        for a in annot2.values():
            a.attach(rp2)

    world.reset()
    franka.set_joints_default_state(positions=FRANKA_READY)
    franka.set_joint_positions(FRANKA_READY)
    franka.apply_action(ArticulationAction(joint_positions=FRANKA_READY))   # drive targets, or it sags back to 0
    rng = np.random.default_rng(args.seed)

    if args.livestream:
        drop_all(objects, rng)
        place_on_shelf(shelf_objects, cfg['shelf'])
        stream(world)
        app.close()
        return

    prims = {o['name']: prim for o, prim, _ in objects}
    T_cam_world = tf.invert(T_world_cam)
    for i in range(args.frames):
        move_arm(franka, FRANKA_READY[:7])                 # table shot: arm at home
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

        move_arm(franka, np.array(wcfg['look_joints']))    # shelf shot: arm in its looking pose
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
