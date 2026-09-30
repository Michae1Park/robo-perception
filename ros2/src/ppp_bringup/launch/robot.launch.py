"""The robot side of the cell: robot_state_publisher (Panda URDF), camera extrinsics as static TF (from
config/calibration.yaml), and the ros2_control controllers. The controller_manager itself runs inside Isaac Sim
(sim/ros_cell.py, D-015); on a real Panda franka_ros2 would host it and this file would stay the same.

  ros2 launch ppp_bringup robot.launch.py [foxglove:=true]
"""
import os
import sys

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from common import moveit_config  # noqa: E402

SHARE = get_package_share_directory('ppp_bringup')


def static_tf(name, e):
    return Node(package='tf2_ros', executable='static_transform_publisher', name=name, output='log',
                arguments=['--x', str(e['xyz'][0]), '--y', str(e['xyz'][1]), '--z', str(e['xyz'][2]),
                           '--qx', str(e['quat_xyzw'][0]), '--qy', str(e['quat_xyzw'][1]),
                           '--qz', str(e['quat_xyzw'][2]), '--qw', str(e['quat_xyzw'][3]),
                           '--frame-id', e['parent'], '--child-frame-id', e['child']],
                parameters=[{'use_sim_time': True}])


def generate_launch_description():
    moveit = moveit_config()
    with open(os.path.join(SHARE, 'config', 'calibration.yaml')) as f:
        calib = yaml.safe_load(f)
    controllers = os.path.join(SHARE, 'config', 'ros2_controllers.yaml')
    spawner = Node(package='controller_manager', executable='spawner', output='screen',
                   arguments=['joint_state_broadcaster', 'panda_arm_controller', 'panda_hand_controller',
                              '--param-file', controllers, '--controller-manager-timeout', '120'],
                   parameters=[{'use_sim_time': True}])
    return LaunchDescription([
        DeclareLaunchArgument('foxglove', default_value='false', description='start foxglove_bridge (port 8765)'),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='log',
             parameters=[moveit.robot_description, {'use_sim_time': True}]),
        # MoveIt's Panda SRDF has a floating virtual joint world -> panda_link0: the base is the cell's origin
        static_tf('world_to_base', {'parent': 'world', 'child': 'panda_link0', 'xyz': [0, 0, 0], 'quat_xyzw': [0, 0, 0, 1]}),
        static_tf('fixed_camera_extrinsics', calib['fixed_camera']),
        static_tf('wrist_camera_extrinsics', calib['wrist_camera']),
        spawner,
        Node(package='foxglove_bridge', executable='foxglove_bridge', output='log',
             parameters=[{'use_sim_time': True, 'port': 8765}], condition=IfCondition(LaunchConfiguration('foxglove'))),
    ])
