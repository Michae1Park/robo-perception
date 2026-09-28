"""Trajectory / gripper execution for the simulated Franka.

Implements the controller interfaces MoveIt expects and streams position targets to Isaac Sim:
  /panda_arm_controller/follow_joint_trajectory  control_msgs/FollowJointTrajectory
  /panda_hand_controller/gripper_cmd             control_msgs/GripperCommand
  -> /joint_command (sensor_msgs/JointState, position targets), feedback from /joint_states
"""
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ppp_common.config import SceneConfig


def retime(q0, positions, max_vel):
    """Time stamps for waypoints when the trajectory carries none (constant max joint speed)."""
    t, out, prev = 0.0, [], np.asarray(q0)
    for q in positions:
        t += max(np.abs(np.asarray(q) - prev).max() / max_vel, 0.02)
        out.append(t)
        prev = np.asarray(q)
    return out


def limit_velocity(times, positions, max_vel):
    """Stretch the time axis uniformly so no segment exceeds `max_vel` (rad/s)."""
    factor = 1.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        need = np.abs(positions[i] - positions[i - 1]).max() / max_vel
        if dt > 0 and need > dt:
            factor = max(factor, need / dt)
    return [t * factor for t in times]


def interpolate(times, positions, t):
    if t <= times[0]:
        return positions[0]
    if t >= times[-1]:
        return positions[-1]
    i = int(np.searchsorted(times, t))
    a = (t - times[i - 1]) / (times[i] - times[i - 1])
    return positions[i - 1] + a * (positions[i] - positions[i - 1])


class ExecutorNode(Node):
    def __init__(self):
        super().__init__('sim_executor')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('rate', 100.0)
        self.declare_parameter('max_joint_velocity', 0.8)
        self.declare_parameter('goal_tolerance', 0.02)
        self.declare_parameter('settle_timeout', 3.0)
        cfg = SceneConfig(self.get_parameter('scene_config').value)
        self.fingers = cfg.robot['finger_joints']
        self.period = 1.0 / float(self.get_parameter('rate').value)
        self.max_vel = float(self.get_parameter('max_joint_velocity').value)
        self.tol = float(self.get_parameter('goal_tolerance').value)
        self.settle = float(self.get_parameter('settle_timeout').value)

        self.state = {}
        self.state_lock = threading.Lock()
        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, '/joint_states', self.on_state, 10, callback_group=cb)
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)
        self.arm_lock = threading.Lock()
        self.arm_server = ActionServer(self, FollowJointTrajectory, '/panda_arm_controller/follow_joint_trajectory',
                                       execute_callback=self.execute_trajectory, callback_group=cb,
                                       cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.gripper_server = ActionServer(self, GripperCommand, '/panda_hand_controller/gripper_cmd',
                                           execute_callback=self.execute_gripper, callback_group=cb,
                                           cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.get_logger().info('sim executor ready')

    def on_state(self, msg):
        with self.state_lock:
            for n, p in zip(msg.name, msg.position):
                self.state[n] = p
            for n, v in zip(msg.name, msg.velocity):
                self.state[n + '/v'] = v

    def current(self, names):
        with self.state_lock:
            if not all(n in self.state for n in names):
                return None
            return np.array([self.state[n] for n in names])

    def command(self, names, q):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(v) for v in q]
        self.cmd_pub.publish(msg)

    def execute_trajectory(self, goal_handle):
        traj = goal_handle.request.trajectory
        names = list(traj.joint_names)
        result = FollowJointTrajectory.Result()
        q0 = self.current(names)
        if q0 is None:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = 'no joint_states for %s' % names
            return result
        if not traj.points:
            goal_handle.succeed()
            return result
        pos = [np.array(p.positions) for p in traj.points]
        times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points]
        if times[-1] <= 0 or np.any(np.diff(times) <= 0):
            times = retime(q0, pos, self.max_vel)
        # start from where the robot actually is; MoveIt's Cartesian paths are timed at full speed
        times = [0.0] + list(times)
        pos = [q0] + pos
        times = limit_velocity(times, pos, self.max_vel)
        with self.arm_lock:
            t0 = time.monotonic()
            while True:
                if goal_handle.is_cancel_requested:
                    self.command(names, self.current(names))
                    goal_handle.canceled()
                    result.error_string = 'canceled'
                    return result
                t = time.monotonic() - t0
                self.command(names, interpolate(times, pos, t))
                if t >= times[-1]:
                    break
                time.sleep(self.period)
            deadline = time.monotonic() + self.settle
            err = np.inf
            while time.monotonic() < deadline:
                err = np.abs(self.current(names) - pos[-1]).max()
                if err < self.tol:
                    break
                self.command(names, pos[-1])
                time.sleep(self.period)
        goal_handle.succeed()
        if err >= self.tol:
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = 'final error %.3f rad' % err
        return result

    def execute_gripper(self, goal_handle):
        target = float(np.clip(goal_handle.request.command.position, 0.0, 0.04))
        result = GripperCommand.Result()
        t0 = time.monotonic()
        last, still_since = None, None
        while time.monotonic() - t0 < 3.0:
            self.command(self.fingers, [target, target])
            q = self.current(self.fingers)
            if q is not None:
                if np.abs(q - target).max() < 0.002:
                    result.reached_goal = True
                    break
                if last is not None and np.abs(q - last).max() < 2e-4:
                    still_since = still_since or time.monotonic()
                    if time.monotonic() - still_since > 0.3 and time.monotonic() - t0 > 0.5:
                        result.stalled = True
                        break
                else:
                    still_since = None
                last = q
            time.sleep(0.02)
        q = self.current(self.fingers)
        result.position = float(q.mean()) if q is not None else 0.0
        goal_handle.succeed()
        return result


def main():
    rclpy.init()
    node = ExecutorNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
