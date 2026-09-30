"""The task manager (BehaviorTree.CPP): pick each target from the table and place it on the shelf, then exit.

  ros2 launch ppp_bringup task.launch.py [targets:='[mustard_bottle, tomato_soup_can]']
"""
import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder

REPO = os.environ.get('PPP_REPO') or str(Path(os.path.realpath(__file__)).parents[4])


def generate_launch_description():
    with open(os.path.join(REPO, 'config.yaml')) as f:
        look = yaml.safe_load(f)['sim']['wrist_camera']['look_joints']
    moveit = MoveItConfigsBuilder('moveit_resources_panda').to_moveit_configs()
    return LaunchDescription([
        DeclareLaunchArgument('targets', default_value='[mustard_bottle, tomato_soup_can]'),
        DeclareLaunchArgument('pose_service', default_value='/pose_node/estimate_pose'),
        Node(package='ppp_task', executable='task_manager', output='screen',
             parameters=[moveit.robot_description, moveit.robot_description_semantic,
                         moveit.robot_description_kinematics, moveit.joint_limits,
                         {'use_sim_time': True, 'look_joints': [float(v) for v in look],
                          'ycb_dir': os.path.join(REPO, 'assets', 'ycb'),
                          'pose_service': LaunchConfiguration('pose_service'),
                          'targets': ParameterValue(LaunchConfiguration('targets'), value_type=list[str])}]),
    ])
