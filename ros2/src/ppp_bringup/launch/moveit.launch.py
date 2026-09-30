"""MoveIt 2 for the Panda (moveit_resources_panda_moveit_config + the cell's longer fingers, common.py): move_group
with Pilz, STOMP and OMPL, executing through panda_arm_controller (FollowJointTrajectory) and panda_hand_controller
(GripperCommand), then the cell's fixed collision objects (table, pedestal, shelf, floor) from config.yaml.

  ros2 launch ppp_bringup moveit.launch.py
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from common import REPO, VENV_PY, moveit_config  # noqa: E402


def generate_launch_description():
    moveit = moveit_config()
    return LaunchDescription([
        SetEnvironmentVariable('PPP_REPO', REPO),
        Node(package='moveit_ros_move_group', executable='move_group', output='screen',
             parameters=[moveit.to_dict(), {'use_sim_time': True,
                                            # the sim arm tracks a few mrad behind its command at rest
                                            'trajectory_execution.allowed_start_tolerance': 0.02}]),
        Node(package='ppp_bringup', executable='planning_scene.py', output='screen',
             prefix=VENV_PY, parameters=[{'use_sim_time': True}]),
    ])
