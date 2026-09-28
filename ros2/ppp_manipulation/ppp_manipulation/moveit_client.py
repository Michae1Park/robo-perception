"""Thin blocking wrapper around the move_group ROS 2 interfaces (Humble has no moveit_py binaries).

Planning happens in MoveIt; execution goes to our own FollowJointTrajectory / GripperCommand servers
(executor_node) that drive the simulated Franka.
"""
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (CollisionObject, Constraints, JointConstraint, MotionPlanRequest,
                             MoveItErrorCodes, PlanningScene, PositionIKRequest, RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from ppp_common.ros_conv import T_to_pose
from ppp_common.ros_utils import CallError, call_service, run_action


class MoveItClient:
    def __init__(self, node, arm_joints, group='panda_arm', hand_link='panda_hand', frame='world',
                 callback_group=None, velocity_scaling=0.3):
        self.node = node
        self.arm_joints = list(arm_joints)
        self.group = group
        self.hand_link = hand_link
        self.frame = frame
        self.velocity_scaling = velocity_scaling
        cb = callback_group
        self.ik = node.create_client(GetPositionIK, '/compute_ik', callback_group=cb)
        self.cartesian = node.create_client(GetCartesianPath, '/compute_cartesian_path', callback_group=cb)
        self.scene = node.create_client(ApplyPlanningScene, '/apply_planning_scene', callback_group=cb)
        self.move_group = ActionClient(node, MoveGroup, '/move_action', callback_group=cb)
        self.follow = ActionClient(node, FollowJointTrajectory,
                                   '/panda_arm_controller/follow_joint_trajectory', callback_group=cb)
        self.gripper = ActionClient(node, GripperCommand, '/panda_hand_controller/gripper_cmd',
                                    callback_group=cb)

    # ---- queries ------------------------------------------------------------------------
    def solve_ik(self, T_world_hand, seed, avoid_collisions=True, timeout=0.05):
        """Returns a list of arm joint positions or None."""
        req = GetPositionIK.Request()
        ik = PositionIKRequest()
        ik.group_name = self.group
        ik.ik_link_name = self.hand_link
        ik.avoid_collisions = avoid_collisions
        ik.robot_state = RobotState()
        ik.robot_state.joint_state = JointState(name=self.arm_joints, position=[float(v) for v in seed])
        ik.pose_stamped.header.frame_id = self.frame
        ik.pose_stamped.pose = T_to_pose(T_world_hand)
        ik.timeout = Duration(sec=0, nanosec=int(timeout * 1e9))
        req.ik_request = ik
        res = call_service(self.ik, req, timeout=5.0)
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        js = res.solution.joint_state
        pos = dict(zip(js.name, js.position))
        return [pos[j] for j in self.arm_joints]

    def plan_to_joints(self, joints, planning_time=5.0, attempts=3):
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.group
        req.num_planning_attempts = attempts
        req.allowed_planning_time = planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.velocity_scaling
        req.start_state.is_diff = True
        c = Constraints()
        for name, q in zip(self.arm_joints, joints):
            c.joint_constraints.append(JointConstraint(joint_name=name, position=float(q), tolerance_above=1e-3,
                                                       tolerance_below=1e-3, weight=1.0))
        req.goal_constraints.append(c)
        goal.request = req
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        res = run_action(self.move_group, goal, timeout=planning_time + 10.0)
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            raise CallError('MoveIt planning failed (error %d)' % res.error_code.val)
        return res.planned_trajectory.joint_trajectory

    def plan_cartesian(self, waypoints, max_step=0.005, min_fraction=0.95, avoid_collisions=True):
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.start_state.is_diff = True
        req.group_name = self.group
        req.link_name = self.hand_link
        req.waypoints = [T_to_pose(T) for T in waypoints]
        req.max_step = max_step
        req.jump_threshold = 0.0
        req.avoid_collisions = avoid_collisions
        res = call_service(self.cartesian, req, timeout=10.0)
        if res.fraction < min_fraction:
            raise CallError('Cartesian path only %.0f %% feasible' % (res.fraction * 100))
        return res.solution.joint_trajectory

    # ---- planning scene -----------------------------------------------------------------
    def add_box(self, name, T_world_box, size):
        co = CollisionObject()
        co.header.frame_id = self.frame
        co.id = name
        co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(s) for s in size]))
        co.primitive_poses.append(T_to_pose(T_world_box))
        co.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        req = ApplyPlanningScene.Request(scene=scene)
        return call_service(self.scene, req, timeout=5.0).success

    # ---- execution ----------------------------------------------------------------------
    def execute(self, trajectory, timeout=60.0):
        goal = FollowJointTrajectory.Goal(trajectory=trajectory)
        res = run_action(self.follow, goal, timeout=timeout)
        if res.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise CallError('trajectory execution failed: %s' % res.error_string)

    def grip(self, position, max_effort=40.0, timeout=5.0):
        """position = per-finger opening (m). Returns the GripperCommand result."""
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        return run_action(self.gripper, goal, timeout=timeout)
