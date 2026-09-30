"""The task manager (BehaviorTree.CPP): pick each target from the table and place it on the shelf, then exit.

  ros2 launch ppp_bringup task.launch.py [targets:='[mustard_bottle, tomato_soup_can]']   (default: config.yaml objects)
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from common import REPO, TARGETS, config, moveit_config, tcp  # noqa: E402


def generate_launch_description():
    robot = config()['robot']
    moveit = moveit_config()
    return LaunchDescription([
        DeclareLaunchArgument('targets', default_value='[%s]' % ', '.join(TARGETS)),
        Node(package='ppp_task', executable='task_manager', output='screen',
             parameters=[moveit.robot_description, moveit.robot_description_semantic,
                         moveit.robot_description_kinematics, moveit.joint_limits,
                         {'use_sim_time': True, 'home_joints': robot['home_joints'], 'look_joints': robot['look_joints'],
                          'tcp_offset': tcp(),
                          'ycb_dir': os.path.join(REPO, 'assets', 'ycb'),
                          'targets': ParameterValue(LaunchConfiguration('targets'), value_type=list[str])}]),
    ])
