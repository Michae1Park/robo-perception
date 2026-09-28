"""⑥ Task level: visual prompt -> detect -> 6-DoF pose -> scene -> grasp -> place -> MoveIt execution.

Action  /pick_place  ppp_interfaces/PickPlace
Uses    /perception/{detect, estimate_pose, analyze_scene}, /manipulation/{plan_grasp, plan_place},
        MoveIt 2 move_group (plan only) and the sim executor.
"""
import threading
import time
from collections import OrderedDict

import numpy as np
import rclpy
from message_filters import Subscriber, TimeSynchronizer
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

from ppp_common import transforms as tf
from ppp_common.config import SceneConfig
from ppp_common.ros_conv import pose_to_T, stamp_to_sec, transform_to_T
from ppp_common.ros_utils import CallError, Stopwatch, call_service
from ppp_interfaces.action import PickPlace
from ppp_interfaces.srv import AnalyzeScene, DetectTarget, EstimatePose, PlanGrasp, PlanPlace

from .moveit_client import MoveItClient


class StageFailed(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


class TaskManager(Node):
    def __init__(self):
        super().__init__('task_manager')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('velocity_scaling', 0.3)
        self.declare_parameter('grip_effort', 40.0)
        self.declare_parameter('frame_timeout', 3.0)
        self.cfg = SceneConfig(self.get_parameter('scene_config').value)
        self.robot = self.cfg.robot
        self.effort = float(self.get_parameter('grip_effort').value)
        self.frame_timeout = float(self.get_parameter('frame_timeout').value)
        cb = ReentrantCallbackGroup()

        self.frame = None
        self.frame_cond = threading.Condition()
        subs = [Subscriber(self, Image, '/camera/color/image_raw', callback_group=cb),
                Subscriber(self, Image, '/camera/depth/image_raw', callback_group=cb),
                Subscriber(self, CameraInfo, '/camera/color/camera_info', callback_group=cb)]
        self.sync = TimeSynchronizer(subs, 10)
        self.sync.registerCallback(self.on_frame)
        self.joints = None
        self.fingers = None
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10, callback_group=cb)

        self.detect = self.create_client(DetectTarget, '/perception/detect', callback_group=cb)
        self.pose = self.create_client(EstimatePose, '/perception/estimate_pose', callback_group=cb)
        self.scene = self.create_client(AnalyzeScene, '/perception/analyze_scene', callback_group=cb)
        self.grasp = self.create_client(PlanGrasp, '/manipulation/plan_grasp', callback_group=cb)
        self.place = self.create_client(PlanPlace, '/manipulation/plan_place', callback_group=cb)
        self.moveit = MoveItClient(self, self.robot['arm_joints'], hand_link=self.robot['hand_link'],
                                   frame=self.cfg.world_frame, callback_group=cb,
                                   velocity_scaling=float(self.get_parameter('velocity_scaling').value))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.busy = threading.Lock()
        self.server = ActionServer(self, PickPlace, '/pick_place', execute_callback=self.execute,
                                   callback_group=cb)
        self.get_logger().info('task manager ready: ros2 action send_goal /pick_place '
                               'ppp_interfaces/action/PickPlace "{target: %s}"' % self.cfg.object_names[0])

    # ---- inputs ---------------------------------------------------------------------------
    def on_frame(self, rgb, depth, info):
        with self.frame_cond:
            self.frame = (rgb, depth, info)
            self.frame_cond.notify_all()

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in self.robot['arm_joints']):
            self.joints = np.array([pos[j] for j in self.robot['arm_joints']])
        if all(j in pos for j in self.robot['finger_joints']):
            self.fingers = float(np.mean([pos[j] for j in self.robot['finger_joints']]))

    def fresh_frame(self):
        """First synchronized RGB-D frame captured after this call."""
        t_req = self.get_clock().now().nanoseconds * 1e-9
        deadline = time.monotonic() + self.frame_timeout
        with self.frame_cond:
            while time.monotonic() < deadline:
                if self.frame is not None and stamp_to_sec(self.frame[0].header.stamp) >= t_req - 0.05:
                    return self.frame
                self.frame_cond.wait(0.1)
        raise StageFailed('camera', 'no RGB-D frame within %.1fs' % self.frame_timeout)

    # ---- helpers --------------------------------------------------------------------------
    def service(self, stage, client, req, timeout, timings):
        sw = Stopwatch()
        try:
            res = call_service(client, req, timeout=timeout)
        except CallError as e:
            raise StageFailed(stage, str(e))
        timings[stage] = sw.ms()
        if hasattr(res, 'latency_ms'):
            timings[stage + '_compute'] = float(res.latency_ms)
        if not res.success:
            raise StageFailed(stage, res.message)
        return res

    def move(self, stage, fn, *args):
        try:
            traj = fn(*args)
            self.moveit.execute(traj)
        except CallError as e:
            raise StageFailed(stage, str(e))

    def go_home(self):
        if self.joints is not None and np.abs(self.joints - np.array(self.robot['home'])).max() < 0.05:
            return
        self.move('home', self.moveit.plan_to_joints, self.robot['home'])

    def add_table(self, table_height):
        """Table as a MoveIt collision box, at the height measured by the plane fit."""
        t = self.cfg.raw['table']
        center = np.array(t['center'], dtype=float)
        center[2] = table_height - t['size'][2] / 2.0
        self.moveit.add_box('table', tf.make_T(t=center), t['size'])

    # ---- the task -------------------------------------------------------------------------
    def execute(self, goal_handle):
        target = goal_handle.request.target
        result = PickPlace.Result()
        timings = OrderedDict()
        fb = PickPlace.Feedback()

        def stage(name):
            fb.stage = name
            goal_handle.publish_feedback(fb)

        if not self.busy.acquire(blocking=False):
            goal_handle.abort()
            result.failed_stage, result.message = 'busy', 'another pick_place goal is running'
            return result
        total = Stopwatch()
        moved = False
        try:
            self.moveit.grip(0.04)
            self.go_home()
            rgb, depth, info = self.fresh_frame()
            result.image_stamp = rgb.header.stamp
            perception = Stopwatch()

            stage('detect')
            det = self.service('detect', self.detect, DetectTarget.Request(target=target, rgb=rgb), 10.0, timings)
            result.bbox, result.detection_score = det.bbox, det.score

            stage('pose')
            pose = self.service('pose', self.pose, EstimatePose.Request(
                target=target, rgb=rgb, depth=depth, camera_info=info, mask=det.mask), 15.0, timings)
            result.estimated_pose = pose.pose

            stage('scene')
            scene = self.service('scene', self.scene, AnalyzeScene.Request(depth=depth, camera_info=info),
                                 10.0, timings)
            timings['perception_total'] = perception.ms()
            self.add_table(scene.table_height)

            stage('grasp_plan')
            sw = Stopwatch()
            grasp = self.service('grasp_plan', self.grasp, PlanGrasp.Request(
                target=target, object_pose=pose.pose, table_height=scene.table_height), 20.0, timings)
            result.grasp = grasp.grasp

            stage('place_plan')
            place = self.service('place_plan', self.place, PlanPlace.Request(
                target=target, object_pose=pose.pose, object_to_hand=grasp.object_to_hand,
                table_height=scene.table_height, place_zone=scene.place_zone), 20.0, timings)
            result.planned_place = place.object_place
            timings['planning_total'] = sw.ms()

            sw = Stopwatch()
            stage('approach')
            moved = True
            T_grasp = pose_to_T(grasp.grasp.pose)
            self.move('approach', self.moveit.plan_to_joints, list(grasp.pregrasp_joints.position))
            self.move('approach', self.moveit.plan_cartesian, [T_grasp])

            grip = self.moveit.grip(0.0, self.effort)
            if grip.reached_goal or grip.position < 0.002:
                raise StageFailed('grasp', 'gripper closed on nothing')
            stage('grasped')

            T_lift = T_grasp.copy()
            T_lift[2, 3] += self.robot['lift_height']
            self.move('lift', self.moveit.plan_cartesian, [T_lift])
            stage('lifted')
            time.sleep(0.3)
            if self.fingers is not None and self.fingers < 0.002:
                raise StageFailed('grasp', 'object slipped during lift')

            stage('placing')
            self.move('transfer', self.moveit.plan_to_joints, list(place.preplace_joints.position))
            self.move('place', self.moveit.plan_cartesian, [pose_to_T(place.place.pose)])
            self.moveit.grip(0.04)
            stage('released')

            T_retreat = pose_to_T(place.place.pose)
            T_retreat[2, 3] += self.robot['retreat_height']
            self.move('retreat', self.moveit.plan_cartesian, [T_retreat])
            self.move('home', self.moveit.plan_to_joints, self.robot['home'])
            timings['execution'] = sw.ms()
            stage('done')
            result.success = True
            result.message = 'ok'
            goal_handle.succeed()
        except StageFailed as e:
            result.success = False
            result.failed_stage = e.stage
            result.message = str(e)
            self.get_logger().warn('pick_place %s failed at %s: %s' % (target, e.stage, e))
            self.recover(moved)
            goal_handle.abort()
        except CallError as e:
            result.success, result.failed_stage, result.message = False, 'execution', str(e)
            self.recover(moved)
            goal_handle.abort()
        finally:
            timings['total'] = total.ms()
            result.stage_names = list(timings.keys())
            result.stage_latency_ms = [float(v) for v in timings.values()]
            self.busy.release()
        self.get_logger().info('pick_place %s: %s | %s' % (
            target, 'SUCCESS' if result.success else 'FAIL(%s)' % result.failed_stage,
            ', '.join('%s %.0f ms' % kv for kv in timings.items() if not kv[0].endswith('_compute'))))
        return result

    def recover(self, moved):
        """Best effort: open the gripper and, if the arm left home, back off upwards and go home."""
        try:
            self.moveit.grip(0.04)
        except Exception as e:  # noqa: BLE001 - recovery must never raise
            self.get_logger().warn('recovery: gripper: %s' % e)
        if not moved:
            return
        for name, fn, arg in (('lift', self.moveit.plan_cartesian, None),
                              ('home', self.moveit.plan_to_joints, self.robot['home'])):
            try:
                if name == 'lift':
                    T = self.hand_pose()
                    if T is None:
                        continue
                    T[2, 3] += 0.08
                    arg = [T]
                self.moveit.execute(fn(arg))
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn('recovery: %s: %s' % (name, e))

    def hand_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.cfg.world_frame, self.robot['hand_link'], rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        return transform_to_T(t.transform)


def main():
    rclpy.init()
    node = TaskManager()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
