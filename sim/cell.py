"""The Isaac Sim cell, shared by scene.py (dataset capture) and ros_cell.py (live over ROS 2): table + YCB
objects + shelf + Franka (longer fingers) + fixed and wrist RGB-D cameras. Scene parameters: config.yaml `sim:`.

Import only after SimulationApp has been created (omni / isaacsim modules need the running app).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PreviewSurface
from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCylinder, VisualSphere
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Semantics, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'sim'))
from vision import read_obj_vertices, shelf_boxes
from vision import transforms as tf
from ycb_usd import YCB, ensure_object_usd, prim_name

# USD cameras look down -z with +y up; OpenCV/ROS optical frames look down +z with +y down.
USD_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])
FRANKA_USD = '/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'   # under the NVIDIA asset root
RS_USD = '/Isaac/Sensors/Intel/RealSense/rsd455.usd'   # RealSense D455 model (body only; its own sensors unused)
# D455 asset frame: +x forward (front glass at x = 0), +y left, +z up; colour lens at y = -11.5 mm.
# T_cam_rs places it so the colour lens sits on our (OpenCV) camera's optical centre.
T_CAM_RS = np.array([[0.0, -1.0, 0.0, -0.0115], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
RS_MOUNT = np.array([-0.013, 0.0, -0.0145])   # 1/4" tripod thread under the body (asset frame)
FINGER_TIP = 0.0539   # stock finger tip, finger link frame (+z towards the tip; pad face at y = 0)


def quat_wxyz(R):
    q = Gf.Matrix3d(*R.T.flatten()).ExtractRotation().GetQuat()   # Gf is row-vector: pass R^T
    return np.array([q.GetReal(), *q.GetImaginary()])


def linear(srgb):
    """config.yaml colours are sRGB (as in a colour picker); USD material colours are linear."""
    return np.asarray(srgb, float) ** 2.2


def set_pose(prim, T):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*T.T.flatten()))


def box(path, center, size, color, material=None):
    return FixedCuboid(path, position=np.array(center, float), scale=np.array(size, float), size=1.0,
                       color=None if material else linear(color), visual_material=material)


def add_shelf(cfg, floor_z, color):
    """Metal pantry: corner posts + one board per level, open on every side (vision.shelf_boxes)."""
    steel = PreviewSurface('/World/Looks/steel', color=linear(color), roughness=0.4, metallic=0.7)
    for name, center, size in shelf_boxes(cfg, floor_z):
        box('/World/Shelf/' + name, center, size, color, steel)


def add_ycb(world, name, label, path=None):
    """YCB object as a dynamic rigid body with a convex-hull collider. -> (SingleRigidPrim, mesh vertices)."""
    stage = get_current_stage()
    path = path or '/World/Objects/' + prim_name(name)
    add_reference_to_stage(ensure_object_usd(name), path)
    root = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.3)
    # more solver iterations than the default 4: squeezed by the gripper, a light body otherwise sinks into the pads
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateSolverPositionIterationCountAttr(32)
    geom = stage.GetPrimAtPath(path + '/geom')
    UsdPhysics.CollisionAPI.Apply(geom)
    UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr('convexHull')
    sem = Semantics.SemanticsAPI.Apply(root, 'Semantics')
    sem.CreateSemanticTypeAttr('class')
    sem.CreateSemanticDataAttr(label)
    verts = read_obj_vertices(os.path.join(YCB, name, 'textured.obj'))
    return world.scene.add(SingleRigidPrim(path, name=label)), verts


def build_scene(cfg, gripper_cfg):
    """cfg: config.yaml `sim:`, gripper_cfg: `gripper:`. -> world, franka, objects, shelf_objects."""
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
    add_finger_extensions(stage, gripper_cfg['finger_extension'], col['fingers'])
    # the finger mimic joint drifts under the 140 N squeeze with the default 32 iterations
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath('/World/Franka')).CreateSolverPositionIterationCountAttr(64)
    franka = world.scene.add(SingleArticulation('/World/Franka', name='franka'))

    objects = [(o, *add_ycb(world, o['name'], o['name'])) for o in cfg['objects']]   # (config, prim, vertices)
    shelf_objects = [(o, *add_ycb(world, o['name'], 'shelf_' + o['name'], '/World/ShelfObjects/' + prim_name(o['name'])))
                     for o in cfg.get('shelf_objects', [])]
    return world, franka, objects, shelf_objects


def set_drives(stage, arm, g):
    """Joint drives closer to a real Franka than the asset's (D-026, D-035). Arm: franka.usd's damping makes it trail
    a moving trajectory by centimetres. Hand: the asset caps the finger drive at 7.2 N; a real Franka Hand squeezes
    with 70 N continuous / 140 N peak. The drive pushes with stiffness x (commanded - actual opening), capped at
    max_force: closing (command 0) on a 4-8 cm object saturates it. panda_finger_joint2 mimics joint1."""
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


def add_finger_extensions(stage, length, color):
    """A rigid box on each finger continuing it `length` past the stock tip, pad face flush with the stock pad
    (y = 0), so the fingers reach deeper and the hand stays further from the object. It is part of the finger link
    and gets the finger-pad material. MoveIt gets the same boxes (ppp_bringup launch/common.py)."""
    if length <= 0:
        return
    z0, z1 = FINGER_TIP - 0.009, FINGER_TIP + length    # overlaps the last 9 mm of the stock finger
    for finger, side in (('panda_leftfinger', 1.0), ('panda_rightfinger', -1.0)):
        cube = UsdGeom.Cube.Define(stage, '/World/Franka/%s/extension' % finger)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*linear(color))])
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, side * 0.006, (z0 + z1) / 2))
        xf.AddScaleOp().Set(Gf.Vec3d(0.0176, 0.012, z1 - z0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.01)


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


def add_d455(path, T_parent_rs):
    """NVIDIA's RealSense D455 model, visual only (no rigid body, no colliders), posed in its parent prim's frame."""
    stage = get_current_stage()
    add_reference_to_stage(get_assets_root_path() + RS_USD, path)
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path + '/RSD455')).CreateRigidBodyEnabledAttr(False)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy():
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    set_pose(stage.GetPrimAtPath(path), T_parent_rs)


def add_camera_rig(T_world_cam, floor_z, rig_cfg):
    """D455 body on a ball head + floor tripod, all visual only. Everything sits behind / below the optical centre,
    so the camera never sees its own rig."""
    T_world_rs = T_world_cam @ T_CAM_RS
    add_d455('/World/CameraRig/D455', T_world_rs)
    black, metal = rig_cfg['color'], rig_cfg['metal_color']
    mount = tf.transform_points(T_world_rs, RS_MOUNT[None])[0]
    plate = mount - 0.012 * T_world_rs[:3, 2]                          # along the body's "down"
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


def define_camera(path, width, height, fx):
    """Pinhole USD camera (square pixels, centred principal point). -> K."""
    cam = UsdGeom.Camera.Define(get_current_stage(), path)
    ha = 20.955                                   # horizontal aperture (mm); focal length follows from fx
    cam.CreateHorizontalApertureAttr(ha)
    cam.CreateVerticalApertureAttr(ha * height / width)
    cam.CreateFocalLengthAttr(fx * ha / width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


def add_camera(cam_cfg, path='/World/Camera'):
    """Fixed camera at cam_cfg eye/target. -> (prim path, K, T_world_cam)."""
    K = define_camera(path, cam_cfg['width'], cam_cfg['height'], cam_cfg['fx'])
    T_world_cam = tf.look_at(cam_cfg['eye'], cam_cfg['target'])
    set_pose(get_current_stage().GetPrimAtPath(path), T_world_cam @ USD_FROM_CV)
    return path, K, T_world_cam


def add_wrist_camera(cam_cfg, wcfg):
    """D455 on the Franka hand (moves with the arm), looking along the hand's +z. -> camera prim path."""
    path = '/World/Franka/panda_hand/WristCamera'
    define_camera(path, cam_cfg['width'], cam_cfg['height'], cam_cfg['fx'])
    T_hand_cam = tf.make_T(tf.rot_z(np.deg2rad(wcfg['mount_yaw_deg'])), wcfg['mount_xyz'])
    set_pose(get_current_stage().GetPrimAtPath(path), T_hand_cam @ USD_FROM_CV)
    add_d455('/World/Franka/panda_hand/WristD455', T_hand_cam @ T_CAM_RS)   # the camera prim itself renders nothing
    return path


def world_pose_cv(path):
    """Current world pose of a USD camera prim, as an OpenCV camera (4x4)."""
    m = UsdGeom.Xformable(get_current_stage().GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m).T @ USD_FROM_CV


def start_livestream():
    """WebRTC stream of the viewport (signalling TCP 49100, video UDP 47998) for the Isaac Sim Streaming Client."""
    import carb
    from isaacsim.core.utils.extensions import enable_extension
    # no NvStreamer-*.etli event-trace files in the working directory
    carb.settings.get_settings().set('/exts/omni.kit.livestream.app/primaryStream/enableEventTracing', False)
    enable_extension('omni.kit.livestream.app')


def set_viewport(cfg):
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=np.array(cfg['viewport']['eye']), target=np.array(cfg['viewport']['target']))


def settle(world, steps):
    for _ in range(steps):                           # annotators update on every rendered step
        world.step(render=True)


def move_arm(franka, q7, gripper=0.04):
    """Set joint positions AND drive targets (else the arm sags back to the drives' targets)."""
    q = np.concatenate([q7, [gripper, gripper]])
    franka.set_joint_positions(q)
    franka.apply_action(ArticulationAction(joint_positions=q))


def reset_robot(world, franka, home):
    """Arm at `home` (config.yaml robot.home_joints), gripper open."""
    world.reset()
    franka.set_joints_default_state(positions=np.r_[home, 0.04, 0.04])
    move_arm(franka, np.asarray(home))


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def place_at(prim, verts, xy, R, surface_z, gap):
    """Set `prim` down with orientation R, its lowest point `gap` above `surface_z`, at rest."""
    z = surface_z - (verts @ R.T)[:, 2].min() + gap
    prim.set_world_pose(position=np.array([*xy, z]), orientation=quat_wxyz(R))
    prim.set_linear_velocity(np.zeros(3))
    prim.set_angular_velocity(np.zeros(3))


def drop_all(objects, rng):
    """Every table object at its spot. yaw_deg [lo, hi] draws a new yaw per call. `tilt_deg` tips the item about its
    own x axis first (90 = lying on its side); tipped items are set down gently (1 mm) so they don't roll.
    -> {name: yaw in degrees}"""
    yaws = {}
    for o, prim, verts in objects:
        y = o.get('yaw_deg', 0.0)
        yaws[o['name']] = rng.uniform(*y) if isinstance(y, list) else float(y)
        tilt = np.deg2rad(o.get('tilt_deg', 0.0))
        R = tf.rot_z(np.deg2rad(yaws[o['name']])) @ rot_x(tilt)
        place_at(prim, verts, o['xy'], R, 0.0, 0.001 if tilt else 0.01)
    return yaws


def place_on_shelf(shelf_objects, shelf_cfg):
    """Stand each shelf item upright on its board, at x along the shelf width, centred front-to-back."""
    cx, cy = shelf_cfg['center_xy']
    for o, prim, verts in shelf_objects:
        R = tf.rot_z(np.deg2rad(o.get('yaw_deg', 0.0)))
        place_at(prim, verts, [cx + o['x'], cy], R, shelf_cfg['levels_z'][o['level']], 0.005)
