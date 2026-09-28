"""Run N randomized episodes against a running sim + pipeline and write the result tables.

  ros2 launch ppp_bringup eval.launch.py num_trials:=50 output_dir:=~/ppp_eval/run1
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('ppp_bringup')
    lc = LaunchConfiguration
    sv = lambda n: ParameterValue(LaunchConfiguration(n), value_type=str)  # noqa: E731 - '' stays a string
    return LaunchDescription([
        DeclareLaunchArgument('scene_config', default_value=os.path.join(share, 'config', 'scene.yaml')),
        DeclareLaunchArgument('assets_dir', default_value=os.environ.get('PPP_ASSETS', '')),
        DeclareLaunchArgument('num_trials', default_value='50'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('output_dir', default_value='eval_results'),
        Node(package='ppp_eval', executable='eval_node', output='screen', parameters=[{
            'scene_config': sv('scene_config'), 'assets_dir': sv('assets_dir'),
            'num_trials': ParameterValue(lc('num_trials'), value_type=int),
            'seed': ParameterValue(lc('seed'), value_type=int), 'output_dir': sv('output_dir')}]),
    ])
