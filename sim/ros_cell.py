#!/usr/bin/env python3
"""The Isaac Sim cell, live over ROS 2 Jazzy: what a real cell would give the pipeline, plus ground truth for
evaluation only (D-010). Same scene as scene.py (sim/cell.py), but it keeps running.

  .venv-sim/bin/python sim/ros_cell.py                 # re-runs itself with /opt/ros/jazzy sourced
  .venv-sim/bin/python sim/ros_cell.py --seed 3 --rtf 0   # other object yaws; as fast as possible
  .venv-sim/bin/python sim/ros_cell.py --livestream    # also stream the viewport to the WebRTC client (D-032)

Published (pipeline-facing, RealSense-style names; depth 16UC1 mm, aligned to colour):
  /camera/camera/color/image_raw, .../color/camera_info,
  /camera/camera/aligned_depth_to_color/image_raw, .../aligned_depth_to_color/camera_info   (fixed camera)
  /wrist_camera/camera/...                                                                   (wrist camera, same set)
  /clock                           sim time (every node runs with use_sim_time)
  /joint_states, /panda_arm_controller/..., /panda_hand_controller/...
                                   ros2_control inside Isaac (controllers spawned by ppp_bringup sim.launch.py)
Evaluation only - the pipeline must never use these:
  /sim/gt/<object>/pose            e.g. /sim/gt/mustard_bottle/pose: PoseStamped in panda_link0 (= world), every camera frame
  /sim/gt/camera_pose, /sim/gt/wrist_camera_pose   true camera extrinsics (OpenCV optical frame)
  /sim/reset (std_srvs/Trigger)    re-drop the table objects with new random yaws, restock the shelf
"""
import argparse
import os
import sys
import sysconfig
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLERS = os.path.join(REPO, 'ros2', 'src', 'ppp_bringup', 'config', 'ros2_controllers.yaml')

# ros2_control inside Isaac links the system's controller_manager and controllers: needs Jazzy sourced before
# the process starts (the loader reads LD_LIBRARY_PATH only then), so re-exec through bash once. Isaac's own
# ROS libraries go after the system ones: pluginlib also finds plugins in Isaac's bundled prefix (e.g. its
# sdformat URDF parser), whose dependencies (libtinyxml2.so.9) exist only there.
EXTS = os.path.join(sysconfig.get_paths()['purelib'], 'isaacsim', 'exts')
if not os.environ.get('PPP_ROS_CELL_ENV'):
    isaac_libs = ':'.join(os.path.join(EXTS, e, 'jazzy', 'lib') for e in ('isaacsim.ros2.control', 'isaacsim.ros2.core'))
    cmd = 'source /opt/ros/jazzy/setup.bash && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s PPP_ROS_CELL_ENV=1 && exec "$0" "$@"' % isaac_libs
    os.execv('/bin/bash', ['bash', '-c', cmd, sys.executable, os.path.abspath(__file__), *sys.argv[1:]])

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--seed', type=int, default=0, help='object yaws (each /sim/reset draws new ones)')
ap.add_argument('--rtf', type=float, default=1.0, help='real-time factor cap (0 = as fast as possible)')
ap.add_argument('--camera-hz', type=float, default=10.0, help='camera publish rate (sim time)')
ap.add_argument('--livestream', action='store_true',
                help='also stream the viewport to the Isaac Sim WebRTC Streaming Client (D-032)')
args = ap.parse_args()

os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'hide_ui': not args.livestream, 'width': 1280, 'height': 720})
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

if args.livestream:
    # WebRTC stream of the viewport (signal TCP 49100, media UDP 47998), set up as in scene.py
    import carb  # noqa: E402
    carb.settings.get_settings().set('/exts/omni.kit.livestream.app/primaryStream/enableEventTracing', False)
    enable_extension('omni.kit.livestream.app')

for ext in ('isaacsim.ros2.bridge', 'isaacsim.ros2.control'):
    enable_extension(ext)
app.update()

import numpy as np  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import rclpy  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from isaacsim.ros2.control import Ros2ControlManager  # noqa: E402  (teardown)
from pxr import Gf  # noqa: E402
from rclpy.qos import QoSProfile  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'sim'))
from cell import (FRANKA_READY, ArticulationAction, add_camera, add_camera_rig, add_wrist_camera,  # noqa: E402
                  build_scene, drop_all, place_on_shelf, quat_wxyz, world_pose_cv)
from vision import load_config  # noqa: E402
from vision import transforms as tf  # noqa: E402

BASE = 'panda_link0'   # robot base frame = sim world origin


def ros_name(ycb):
    """'006_mustard_bottle' -> 'mustard_bottle' (ROS name tokens can't start with a digit)."""
    return ycb.split('_', 1)[1] if ycb[:1].isdigit() else ycb


def add_clock_graph():
    og.Controller.edit(
        {'graph_path': '/World/ROS2Clock', 'evaluator_name': 'execution'},
        {og.Controller.Keys.CREATE_NODES: [
            ('Tick', 'omni.graph.action.OnPlaybackTick'),
            ('SimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
            ('Clock', 'isaacsim.ros2.bridge.ROS2PublishClock')],
         og.Controller.Keys.CONNECT: [
            ('Tick.outputs:tick', 'Clock.inputs:execIn'),
            ('SimTime.outputs:simulationTime', 'Clock.inputs:timeStamp')]})


def stamp(t):
    return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)) % 1000000000)


def pose_msg(T, t):
    m = PoseStamped()
    m.header.frame_id, m.header.stamp = BASE, stamp(t)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, T[:3, 3])
    w, x, y, z = quat_wxyz(T[:3, :3])
    m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = map(float, (w, x, y, z))
    return m


class RGBDPublisher:
    """One camera: colour + aligned depth (16UC1 mm) + camera_info, all with the same stamp (one render)."""

    def __init__(self, node, prim_path, ns, frame_id, K, w, h):
        self.frame_id, self.K, self.w, self.h = frame_id, K, w, h
        rp = rep.create.render_product(prim_path, (w, h))
        self.rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        self.depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
        self.rgb.attach(rp)
        self.depth.attach(rp)
        q = QoSProfile(depth=5)
        self.pub = {k: node.create_publisher(typ, '%s/camera/%s' % (ns, k), q) for k, typ in (
            ('color/image_raw', Image), ('color/camera_info', CameraInfo),
            ('aligned_depth_to_color/image_raw', Image), ('aligned_depth_to_color/camera_info', CameraInfo))}

    def info(self, t):
        m = CameraInfo()
        m.header.frame_id, m.header.stamp = self.frame_id, stamp(t)
        m.width, m.height, m.distortion_model = self.w, self.h, 'plumb_bob'
        m.d = [0.0] * 5
        m.k = self.K.flatten().tolist()
        m.r = np.eye(3).flatten().tolist()
        m.p = np.c_[self.K, np.zeros(3)].flatten().tolist()
        return m

    def image(self, t, arr, encoding, bpp):
        m = Image()
        m.header.frame_id, m.header.stamp = self.frame_id, stamp(t)
        m.height, m.width, m.encoding, m.is_bigendian = arr.shape[0], arr.shape[1], encoding, 0
        m.step = arr.shape[1] * bpp
        m.data = np.ascontiguousarray(arr).tobytes()
        return m

    def publish(self, t):
        rgb = np.asarray(self.rgb.get_data())
        depth = np.asarray(self.depth.get_data(), dtype=np.float32)
        if rgb.size == 0 or depth.size == 0:   # the first frames after start have no data yet
            return
        depth[~np.isfinite(depth)] = 0
        depth_mm = np.clip(np.round(depth * 1000), 0, 65535).astype(np.uint16)
        info = self.info(t)
        self.pub['color/image_raw'].publish(self.image(t, rgb[..., :3], 'rgb8', 3))
        self.pub['aligned_depth_to_color/image_raw'].publish(self.image(t, depth_mm, '16UC1', 2))
        self.pub['color/camera_info'].publish(info)
        self.pub['aligned_depth_to_color/camera_info'].publish(info)


def fixed_joint_patch(stage, root_path, urdf_xml):
    """Isaac 6.1's URDF export drops fixed joints between links (the Franka's panda_link7 -> panda_hand), which
    leaves two roots and the controller_manager refuses the URDF. Re-add each missing one, posed from the USD."""
    import xml.etree.ElementTree as ET
    from pxr import Usd, UsdGeom, UsdPhysics
    root = ET.fromstring(urdf_xml)
    links = {l.get('name') for l in root.findall('link')}
    children = {j.find('child').get('link') for j in root.findall('joint')}
    world_T = lambda path: np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())).T
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        j = UsdPhysics.Joint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not (b0 and b1):
            continue
        parent, child = b0[0].name, b1[0].name
        if parent not in links or child not in links or child in children:
            continue
        T = tf.invert(world_T(b0[0])) @ world_T(b1[0])
        R = T[:3, :3]
        rpy = (np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0]))
        el = ET.SubElement(root, 'joint', {'name': prim.GetName(), 'type': 'fixed'})
        ET.SubElement(el, 'origin', {'xyz': ' '.join('%.9g' % v for v in T[:3, 3]),
                                     'rpy': ' '.join('%.9g' % v for v in rpy)})
        ET.SubElement(el, 'parent', {'link': parent})
        ET.SubElement(el, 'child', {'link': child})
        print('[ros_cell] URDF patch: fixed joint %s (%s -> %s)' % (prim.GetName(), parent, child), flush=True)
    return ET.tostring(root, encoding='unicode')


def start_ros2_control(world, prim_path='/World/Franka'):
    """Physics must have run before the articulation can be wrapped; the manager is set up while paused (as in
    Isaac's own tests). Same steps as Ros2ControlManager.setup(), plus the fixed-joint patch."""
    import omni.usd
    from isaacsim.ros2.control.bindings import _isaacsim_ros2_control as backend
    from isaacsim.ros2.control.urdf_synth import build_full_urdf
    for _ in range(600):
        world.step(render=False)
        if backend.is_ready():
            break
    else:
        raise RuntimeError('isaacsim.ros2.control backend never became ready')
    world.pause()
    for _ in range(2):
        app.update()
    stage = omni.usd.get_context().get_stage()
    urdf = fixed_joint_patch(stage, prim_path, build_full_urdf(stage, prim_path))
    rc = backend.setup_cm(articulation_path=prim_path, urdf_xml=urdf, controller_yaml_path=CONTROLLERS, ns_name='',
                          publish_robot_description=False, use_sim_time=True)
    if rc != 0:
        raise RuntimeError('controller_manager setup failed (%d)' % rc)
    world.play()
    print('[ros_cell] controller_manager up (%s)' % os.path.relpath(CONTROLLERS, REPO), flush=True)


def refuse_second_clock(node, wait=2.0):
    """Two sims both publishing /clock make sim time jump back and forth (TF_OLD_DATA everywhere, I-030)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:                     # give DDS discovery a moment
        rclpy.spin_once(node, timeout_sec=0.1)
    n = node.count_publishers('/clock')
    if n:
        print('[ros_cell] /clock already has %d publisher(s): another sim is running. Stop it first '
              '(pgrep -af ros_cell.py), then restart step 2 too.' % n, flush=True)
        app.close()
        sys.exit(1)                                          # Kit's exit hook takes a status code only


def main():
    rclpy.init()
    node = rclpy.create_node('isaac_sim_cell')
    refuse_second_clock(node)
    cfg = load_config()['sim']
    world, franka, objects, shelf_objects = build_scene(cfg)
    cam_cfg = cfg['camera']
    cam_path, K, T_world_cam = add_camera(cam_cfg)
    add_camera_rig(T_world_cam, -cfg['table_size'][2], cam_cfg['rig'])
    wrist_path = add_wrist_camera(cam_cfg, cfg['wrist_camera'])
    add_clock_graph()

    w, h = cam_cfg['width'], cam_cfg['height']
    cams = [RGBDPublisher(node, cam_path, '/camera', 'camera_color_optical_frame', K, w, h),
            RGBDPublisher(node, wrist_path, '/wrist_camera', 'wrist_camera_color_optical_frame', K, w, h)]
    targets = [(o['name'], prim) for o, prim, _ in objects if o.get('target')]
    gt_pub = {n: node.create_publisher(PoseStamped, '/sim/gt/%s/pose' % ros_name(n), 5) for n, _ in targets}
    gt_cam = node.create_publisher(PoseStamped, '/sim/gt/camera_pose', 5)
    gt_wrist = node.create_publisher(PoseStamped, '/sim/gt/wrist_camera_pose', 5)

    rng = np.random.default_rng(args.seed)
    world.reset()
    franka.set_joints_default_state(positions=FRANKA_READY)
    franka.set_joint_positions(FRANKA_READY)
    franka.apply_action(ArticulationAction(joint_positions=FRANKA_READY))
    drop_all(objects, rng)
    place_on_shelf(shelf_objects, cfg['shelf'])

    reset_requested = []

    def on_reset(_, res):
        reset_requested.append(True)
        res.success, res.message = True, 'objects re-dropped on the next step'
        return res
    node.create_service(Trigger, '/sim/reset', on_reset)

    start_ros2_control(world)
    dt = cfg['physics_dt']
    every = max(1, int(round(1.0 / (args.camera_hz * dt))))
    if args.livestream:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([1.95, -0.25, 1.30]), target=np.array([0.30, -0.08, 0.10]))  # as scene.py
        print('[ros_cell] streaming on TCP 49100 / UDP 47998 - connect the Isaac Sim WebRTC Streaming Client '
              'to this machine\'s IP', flush=True)
    print('[ros_cell] running: cameras at %.1f Hz sim time, rtf cap %s. Ctrl+C to quit.'
          % (1.0 / (every * dt), args.rtf or 'none'), flush=True)
    step, wall0, sim0 = 0, time.monotonic(), world.current_time
    try:
        while app.is_running():
            world.step(render=True)
            step += 1
            rclpy.spin_once(node, timeout_sec=0)
            if reset_requested:
                reset_requested.clear()
                yaws = drop_all(objects, rng)
                place_on_shelf(shelf_objects, cfg['shelf'])
                print('[ros_cell] reset: ' + ', '.join('%s %.0f deg' % (ros_name(n), yaws[n]) for n, _ in targets),
                      flush=True)
            t = world.current_time
            if step % every == 0:
                for c in cams:
                    c.publish(t)
                for n, prim in targets:
                    p, q = prim.get_world_pose()
                    R = np.array(Gf.Matrix3d(Gf.Quatd(*map(float, q)))).T
                    gt_pub[n].publish(pose_msg(tf.make_T(R, p), t))
                gt_cam.publish(pose_msg(T_world_cam, t))
                gt_wrist.publish(pose_msg(world_pose_cv(wrist_path), t))
            if args.rtf > 0:   # don't run ahead of wall time (x rtf)
                ahead = (t - sim0) / args.rtf - (time.monotonic() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
    except KeyboardInterrupt:
        pass
    Ros2ControlManager.teardown_all()
    node.destroy_node()
    rclpy.shutdown()
    app.close()


if __name__ == '__main__':
    main()
