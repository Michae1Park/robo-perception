"""Isaac Sim cell for prompt-pick-place: table + YCB objects + RGB-D camera + Franka, bridged to ROS 2.

  ./python.sh isaac_sim/scene.py [--livestream] [--seed 0]

ROS 2 interface (rclpy on Isaac Sim's bundled Humble libraries):
  pub  /camera/color/image_raw (rgb8), /camera/depth/image_raw (32FC1, m), /camera/color/camera_info
  pub  /joint_states, /tf (gt/<object> ground-truth frames), /tf_static (camera extrinsics)
  pub  /sim/ready   std_msgs/String  {"seed", "objects", "clutter"}   (latched)
  sub  /joint_command  sensor_msgs/JointState  position targets (from executor_node)
  sub  /sim/reset      std_msgs/String  {"seed": int[, "force": bool]}
All stamps use the wall clock (use_sim_time = false on the ROS side).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--scene-config', default='')
ap.add_argument('--assets', default='')
ap.add_argument('--seed', type=int, default=0, help='layout of the first episode')
ap.add_argument('--gui', action='store_true', help='open the Isaac Sim window (default headless)')
ap.add_argument('--livestream', action='store_true', help='WebRTC streaming for headless servers')
ap.add_argument('--render-every', type=int, default=2, help='render every N physics steps (60 Hz physics)')
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': not args.gui, 'width': 1280, 'height': 720})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

if args.livestream:
    app.set_setting('/app/window/drawMouse', True)
    enable_extension('omni.kit.livestream.webrtc')
for ext in ('isaacsim.ros2.bridge', 'omni.isaac.ros2_bridge'):
    try:
        enable_extension(ext)
        break
    except Exception:  # noqa: BLE001
        continue
app.update()

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_common import SimScene  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

from ppp_common import transforms as tf  # noqa: E402
from ppp_common.config import SceneConfig  # noqa: E402


def transform_msg(T, parent, child, stamp):
    m = TransformStamped()
    m.header.frame_id = parent
    m.header.stamp = stamp
    m.child_frame_id = child
    p, q = tf.pq_from_T(T)
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = [float(v) for v in p]
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = \
        [float(v) for v in q]
    return m


def image_msg(arr, encoding, stamp, frame_id):
    arr = np.ascontiguousarray(arr)
    m = Image()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height, m.width = arr.shape[:2]
    m.encoding = encoding
    m.step = arr.strides[0]
    m.data = arr.tobytes()
    return m


class SimBridge:
    def __init__(self, cfg):
        self.cfg = cfg
        cam = cfg.camera
        self.scene = SimScene(cfg)
        self.world = self.scene.world
        self.robot = self.scene.robot
        self.camera = SimScene.add_camera('/World/RGBD', cam['eye'], cam['target'], cam['width'], cam['height'])
        self.world.reset()
        self.camera.initialize()
        SimScene.configure_intrinsics(self.camera, cam['width'], cam['height'], cam['fx'], cam['fy'],
                                      cam['cx'], cam['cy'])
        self.camera.add_distance_to_image_plane_to_frame()
        print('[ppp] camera intrinsics (sim):\n', self.camera.get_intrinsics_matrix())

        self.dof_names = list(self.robot.dof_names)
        r = cfg.robot
        self.home = np.zeros(len(self.dof_names))
        for n, q in zip(r['arm_joints'], r['home']):
            self.home[self.dof_names.index(n)] = q
        for n in r['finger_joints']:
            self.home[self.dof_names.index(n)] = 0.04
        self.targets = self.home.copy()

        rclpy.init()
        self.node = rclpy.create_node('isaac_sim')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_rgb = self.node.create_publisher(Image, '/camera/color/image_raw', 2)
        self.pub_depth = self.node.create_publisher(Image, '/camera/depth/image_raw', 2)
        self.pub_info = self.node.create_publisher(CameraInfo, '/camera/color/camera_info', 2)
        self.pub_js = self.node.create_publisher(JointState, '/joint_states', 10)
        self.pub_tf = self.node.create_publisher(TFMessage, '/tf', 10)
        self.pub_tf_static = self.node.create_publisher(TFMessage, '/tf_static', latched)
        self.pub_ready = self.node.create_publisher(String, '/sim/ready', latched)
        self.node.create_subscription(JointState, '/joint_command', self.on_command, 10)
        self.node.create_subscription(String, '/sim/reset', self.on_reset, 10)

        self.pending_reset = {'seed': args.seed, 'force': True}
        self.last_seed = None
        self.episode = {'objects': [], 'clutter': []}
        self.rng = np.random.default_rng(12345)
        self.T_world_cam = cfg.T_world_camera()
        self.pub_tf_static.publish(TFMessage(transforms=[transform_msg(
            self.T_world_cam, cfg.world_frame, cam['frame_id'], self.node.get_clock().now().to_msg())]))

    # ---- ROS callbacks ---------------------------------------------------------------------
    def on_command(self, msg):
        for n, q in zip(msg.name, msg.position):
            if n in self.dof_names:
                self.targets[self.dof_names.index(n)] = q

    def on_reset(self, msg):
        try:
            req = json.loads(msg.data)
        except ValueError:
            req = {'seed': int(msg.data)}
        if req.get('seed') == self.last_seed and not req.get('force'):
            return  # duplicate request for the episode we already set up
        self.pending_reset = req

    # ---- episode ---------------------------------------------------------------------------
    def reset(self, req):
        seed = int(req['seed'])
        self.targets = self.home.copy()
        self.robot.set_joint_positions(self.home)
        self.robot.set_joint_velocities(np.zeros(len(self.home)))
        objects, clutter = self.scene.layout(seed)
        settle = int(self.cfg.raw['spawn']['settle_time'] / self.cfg.raw['physics']['dt'])
        for _ in range(settle):
            self.robot.apply_action(ArticulationAction(joint_positions=self.targets))
            self.world.step(render=False)
            self.publish_joint_states()
        self.world.render()
        self.episode = {'seed': seed, 'objects': objects, 'clutter': clutter}
        self.last_seed = seed
        self.pub_ready.publish(String(data=json.dumps(self.episode)))
        print('[ppp] episode seed=%d objects=%s clutter=%d' % (seed, objects, len(clutter)))

    # ---- publishing ------------------------------------------------------------------------
    def publish_joint_states(self):
        js = JointState()
        js.header.stamp = self.node.get_clock().now().to_msg()
        js.name = self.dof_names
        js.position = [float(v) for v in self.robot.get_joint_positions()]
        js.velocity = [float(v) for v in self.robot.get_joint_velocities()]
        self.pub_js.publish(js)

    def publish_camera(self):
        rgba = self.camera.get_rgba()
        frame = self.camera.get_current_frame()
        depth = frame.get('distance_to_image_plane') if frame else None
        cam = self.cfg.camera
        if rgba is None or depth is None or rgba.size == 0 or depth.size == 0:
            return
        rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
        depth = np.asarray(depth, dtype=np.float32).copy()
        valid = np.isfinite(depth) & (depth > cam['depth_min']) & (depth < cam['depth_max'])
        if cam.get('depth_noise_std', 0.0) > 0:
            depth[valid] += self.rng.normal(0.0, cam['depth_noise_std'], int(valid.sum())).astype(np.float32)
        depth[~valid] = 0.0
        stamp = self.node.get_clock().now().to_msg()
        fid = cam['frame_id']
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, fid
        info.width, info.height = cam['width'], cam['height']
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [float(v) for v in self.cfg.camera_K().flatten()]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [cam['fx'], 0.0, cam['cx'], 0.0, 0.0, cam['fy'], cam['cy'], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)
        self.pub_depth.publish(image_msg(depth, '32FC1', stamp, fid))
        self.pub_rgb.publish(image_msg(rgb, 'rgb8', stamp, fid))
        gt = [transform_msg(self.scene.object_pose(n), self.cfg.world_frame, 'gt/' + n, stamp)
              for n in self.episode['objects'] + self.episode['clutter']]
        if gt:
            self.pub_tf.publish(TFMessage(transforms=gt))

    def run(self):
        dt = float(self.cfg.raw['physics']['dt'])
        render_every = max(1, args.render_every)
        cam_every = max(1, int(round(1.0 / (self.cfg.camera['rate_hz'] * dt))))
        cam_every = max(render_every, cam_every // render_every * render_every)  # camera only on render steps
        step = 0
        while app.is_running():
            t0 = time.monotonic()
            rclpy.spin_once(self.node, timeout_sec=0.0)
            if self.pending_reset is not None:
                req, self.pending_reset = self.pending_reset, None
                self.reset(req)
            self.robot.apply_action(ArticulationAction(joint_positions=self.targets))
            self.world.step(render=(step % render_every == 0))
            self.publish_joint_states()
            if step % cam_every == 0:
                self.publish_camera()
            step += 1
            # keep roughly real time so trajectory timing on the ROS side matches physics
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    cfg = SceneConfig(args.scene_config, args.assets)
    SimBridge(cfg).run()
    app.close()


if __name__ == '__main__':
    main()
