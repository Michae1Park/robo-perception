"""Automated evaluation: N randomized episodes, scored against Isaac Sim ground truth.

Per episode: /sim/reset (seed) -> wait /sim/ready -> random target -> /pick_place -> compare with GT.
GT object poses come from TF frames `gt/<object>` published by the simulator.
Outputs <output_dir>/trials.jsonl, summary.json, summary.md
"""
import json
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from ppp_common import transforms as tf
from ppp_common.config import SceneConfig
from ppp_common.mesh import read_obj_vertices, subsample
from ppp_common.ros_conv import pose_to_T, transform_to_T
from ppp_common.ros_utils import CallError, run_action
from ppp_interfaces.action import PickPlace

from .core import metrics as m


class EvalNode(Node):
    def __init__(self):
        super().__init__('evaluator')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('assets_dir', '')
        self.declare_parameter('num_trials', 50)
        self.declare_parameter('seed', 0)
        self.declare_parameter('output_dir', 'eval_results')
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('pose_trans_threshold_mm', 10.0)
        self.declare_parameter('pose_rot_threshold_deg', 10.0)
        self.declare_parameter('place_threshold_mm', 30.0)
        self.declare_parameter('place_tilt_threshold_deg', 15.0)
        self.declare_parameter('disturb_threshold_mm', 20.0)
        self.declare_parameter('settle_after', 2.0)
        self.declare_parameter('episode_timeout', 180.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.p = p
        self.cfg = SceneConfig(p('scene_config'), p('assets_dir'))
        self.K = self.cfg.camera_K()
        self.T_cam_world = tf.invert(self.cfg.T_world_camera())
        self.points = {}
        for name in self.cfg.object_names:
            try:
                self.points[name] = subsample(read_obj_vertices(self.cfg.mesh_path(name)), 1000)
            except OSError:
                self.get_logger().warn('mesh missing for %s' % name)

        cb = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.reset_pub = self.create_publisher(String, '/sim/reset', qos)
        self.ready = None
        self.ready_event = threading.Event()
        self.create_subscription(String, '/sim/ready', self.on_ready, qos, callback_group=cb)
        self.client = ActionClient(self, PickPlace, '/pick_place', callback_group=cb)
        self.lifted_z = None

    def on_ready(self, msg):
        self.ready = json.loads(msg.data)
        self.ready_event.set()

    def gt(self, name):
        try:
            t = self.tf_buffer.lookup_transform(self.cfg.world_frame, 'gt/' + name, rclpy.time.Time())
            return transform_to_T(t.transform)
        except Exception:  # noqa: BLE001
            return None

    def reset(self, seed, timeout=60.0):
        """Ask the sim for episode `seed`; resend (without force, so it is not re-run) until ready."""
        self.ready_event.clear()
        self.ready = None
        self.reset_pub.publish(String(data=json.dumps({'seed': seed, 'force': True})))
        t0 = last = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.ready_event.wait(0.5) and self.ready and self.ready.get('seed') == seed:
                return self.ready
            self.ready_event.clear()
            if time.monotonic() - last > 5.0:
                self.reset_pub.publish(String(data=json.dumps({'seed': seed})))
                last = time.monotonic()
        raise CallError('simulator did not report ready for seed %d' % seed)

    def run(self):
        out_dir = os.path.abspath(os.path.expanduser(self.p('output_dir')))
        os.makedirs(out_dir, exist_ok=True)
        trials_path = os.path.join(out_dir, 'trials.jsonl')
        n, seed0 = int(self.p('num_trials')), int(self.p('seed'))
        trials = []
        with open(trials_path, 'w') as f:
            for i in range(n):
                seed = seed0 + i
                try:
                    trial = self.episode(seed)
                except CallError as e:
                    self.get_logger().error('episode %d: %s' % (seed, e))
                    trial = {'seed': seed, 'task_success': False, 'failed_stage': 'harness', 'message': str(e)}
                trials.append(trial)
                f.write(json.dumps(trial) + '\n')
                f.flush()
                ok = sum(1 for t in trials if t.get('task_success'))
                self.get_logger().info('[%d/%d] seed %d %s: %s (running success %d/%d)' % (
                    i + 1, n, seed, trial.get('target'), 'OK' if trial.get('task_success')
                    else 'FAIL@%s' % trial.get('failed_stage'), ok, len(trials)))
        summary = m.summarize(trials)
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        md = m.to_markdown(summary)
        with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
            f.write(md)
        self.get_logger().info('evaluation done -> %s\n%s' % (out_dir, md))

    def episode(self, seed):
        ready = self.reset(seed)
        objects = ready['objects']
        rng = np.random.default_rng(seed)
        target = str(rng.choice(objects))
        time.sleep(0.5)  # let TF catch up with the settled scene
        T0 = {name: self.gt(name) for name in objects + ready.get('clutter', [])}
        if T0.get(target) is None:
            raise CallError('no GT pose for %s' % target)
        trial = {'seed': seed, 'target': target, 'num_objects': len(objects)}

        self.lifted_z = None

        def feedback(msg):
            if msg.feedback.stage == 'lifted':
                T = self.gt(target)
                self.lifted_z = None if T is None else float(T[2, 3])

        goal = PickPlace.Goal(target=target)
        res = run_action(self.client, goal, timeout=float(self.p('episode_timeout')), wait_ready=30.0,
                         feedback_cb=feedback)
        time.sleep(float(self.p('settle_after')))
        trial.update({'task_ok_reported': res.success, 'failed_stage': res.failed_stage, 'message': res.message,
                      'latency_ms': dict(zip(res.stage_names, [float(v) for v in res.stage_latency_ms]))})
        self.score(trial, res, target, T0)
        return trial

    def score(self, trial, res, target, T0):
        cam = self.cfg.camera
        T_gt = T0[target]
        pts = self.points.get(target)
        detected = res.failed_stage not in ('detect', 'camera', 'busy')
        # detection: IoU between the predicted box and the projected GT mesh box
        if detected and pts is not None:
            gt_box = m.projected_bbox(self.K, self.T_cam_world, T_gt, pts, cam['width'], cam['height'])
            trial['iou'] = m.bbox_iou(list(res.bbox), gt_box) if gt_box else 0.0
            trial['detection_success'] = trial['iou'] >= self.p('iou_threshold')
        else:
            trial['detection_success'] = False
        # 6-DoF pose
        if detected and res.failed_stage != 'pose' and pts is not None:
            T_est = pose_to_T(res.estimated_pose.pose)
            sym = self.cfg.symmetry(target)
            trial['trans_err_mm'] = m.translation_error_mm(T_est, T_gt)
            trial['rot_err_deg'] = m.rotation_error_deg(T_est[:3, :3], T_gt[:3, :3], sym)
            trial['add_mm'] = m.add_mm(T_est, T_gt, pts)
            trial['adds_mm'] = m.adds_mm(T_est, T_gt, pts)
            trial['pose_success'] = (trial['trans_err_mm'] <= self.p('pose_trans_threshold_mm')
                                     and trial['rot_err_deg'] <= self.p('pose_rot_threshold_deg'))
        else:
            trial['pose_success'] = False
        # grasp: object followed the gripper up
        lift = self.cfg.robot['lift_height']
        trial['grasp_success'] = self.lifted_z is not None and self.lifted_z > T_gt[2, 3] + 0.5 * lift
        # place: final GT pose vs. planned placement
        T_final = self.gt(target)
        trial['place_success'] = False
        if res.success and T_final is not None:
            T_plan = pose_to_T(res.planned_place.pose)
            trial['place_err_mm'] = float(np.linalg.norm(T_final[:2, 3] - T_plan[:2, 3]) * 1000.0)
            tilt_change = abs(m.tilt_deg(T_final[:3, :3]) - m.tilt_deg(T_gt[:3, :3]))
            dz = abs(T_final[2, 3] - (T_plan[2, 3] - self.cfg.raw['place']['drop_height']))
            trial['place_tilt_change_deg'] = tilt_change
            trial['place_success'] = bool(trial['place_err_mm'] <= self.p('place_threshold_mm')
                                          and tilt_change <= self.p('place_tilt_threshold_deg') and dz < 0.03)
        # side effects: anything else moved?
        moved = []
        for name, T in T0.items():
            if name == target or T is None:
                continue
            T1 = self.gt(name)
            if T1 is not None and np.linalg.norm(T1[:2, 3] - T[:2, 3]) * 1000.0 > self.p('disturb_threshold_mm'):
                moved.append(name)
        trial['disturbed'] = moved
        trial['disturbed_scene'] = bool(moved)
        trial['task_success'] = bool(res.success and trial['grasp_success'] and trial['place_success'])
        if not trial['task_success'] and not trial['failed_stage']:
            trial['failed_stage'] = 'grasp_check' if not trial['grasp_success'] else 'place_check'


def main():
    rclpy.init()
    node = EvalNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
