"""The Isaac Sim cell, shared by scene.py (dataset capture) and ros_cell.py (live over ROS 2): table + YCB
objects + shelf + Franka + fixed and wrist RGB-D cameras. Scene parameters: config.yaml `sim:`.

Import only after SimulationApp has been created (omni / isaacsim modules need the running app).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCylinder, VisualSphere
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Semantics, Usd, UsdGeom, UsdLux, UsdPhysics

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'sim'))
from vision import transforms as tf
from vision.grasp import read_obj_vertices
from ycb_usd import YCB, ensure_object_usd, prim_name

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
    set_drives(stage, cfg['arm_drive'], cfg['gripper'])
    franka = world.scene.add(SingleArticulation('/World/Franka', name='franka'))

    objects = [(o, *add_ycb(world, o['name'], o['name'])) for o in cfg['objects']]   # (config, prim, vertices)
    shelf_objects = [(o, *add_ycb(world, o['name'], 'shelf_' + o['name'], '/World/ShelfObjects/' + prim_name(o['name'])))
                     for o in cfg.get('shelf_objects', [])]
    return world, franka, objects, shelf_objects


def set_drives(stage, arm, g):
    """Joint drives closer to a real Franka than the asset's. Arm: franka.usd's damping (80 N m s/deg against a
    stiffness of 400 N m/deg, a 0.2 s time constant) makes the arm trail a moving trajectory by centimetres.
    Hand: the finger drive is capped at 7.2 N (a real Franka Hand squeezes with 70 N continuous), too weak to hold
    a 0.3 kg bottle while the arm moves; finger pads get rubber-like friction. panda_finger_joint2 mimics joint1,
    so one drive moves both fingers."""
    from pxr import PhysxSchema, UsdShade
    for k in range(1, 8):
        path = '/World/Franka/panda_link%d/panda_joint%d' % (k - 1, k)
        d = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(path), 'angular')
        d.GetStiffnessAttr().Set(arm['stiffness'])
        d.GetDampingAttr().Set(arm['damping'])
    drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath('/World/Franka/panda_hand/panda_finger_joint1'), 'linear')
    drive.GetStiffnessAttr().Set(g['stiffness'])
    drive.GetDampingAttr().Set(g['damping'])
    drive.GetMaxForceAttr().Set(g['max_force'])
    mat = UsdShade.Material.Define(stage, '/World/Physics/finger_pads')
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(g['friction'])
    api.CreateDynamicFrictionAttr(g['friction'])
    PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('max')
    for finger in ('panda_leftfinger', 'panda_rightfinger'):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/Franka/' + finger)).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, 'physics')


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


def add_wrist_camera(cam_cfg, wcfg):
    """RealSense D455 on the Franka hand (moves with the arm), looking along the hand's +z. -> camera prim path."""
    w, h, f = cam_cfg['width'], cam_cfg['height'], cam_cfg['fx']
    path = '/World/Franka/panda_hand/WristCamera'
    cam = UsdGeom.Camera.Define(get_current_stage(), path)
    ha = 20.955
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha * h / w)
    cam.CreateFocalLengthAttr(f * ha / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    T_hand_cam = tf.make_T(tf.rot_z(np.deg2rad(wcfg['mount_yaw_deg']))[:3, :3], wcfg['mount_xyz'])
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*(T_hand_cam @ USD_FROM_CV).T.flatten()))

    # the camera prim renders nothing: add the D455 model, placed like the tripod one (colour lens on the optical
    # centre), visual only - no rigid body, no colliders, so it can't disturb the arm
    stage = get_current_stage()
    body_path = '/World/Franka/panda_hand/WristD455'
    add_reference_to_stage(get_assets_root_path() + RS_USD, body_path)
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(body_path + '/RSD455')).CreateRigidBodyEnabledAttr(False)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(body_path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy():
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(body_path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*(T_hand_cam @ T_CAM_RS).T.flatten()))
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
