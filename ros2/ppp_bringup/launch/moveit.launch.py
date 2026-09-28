"""MoveIt 2 move_group for the Franka Panda (moveit_resources config), planning only.

The simulator publishes /joint_states; the world frame coincides with panda_link0.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    share = get_package_share_directory('ppp_bringup')
    moveit_config = (
        MoveItConfigsBuilder('moveit_resources_panda')
        .robot_description(file_path='config/panda.urdf.xacro')
        .trajectory_execution(file_path=os.path.join(share, 'config', 'moveit_controllers.yaml'))
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )
    rviz = LaunchConfiguration('rviz')
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='world_to_base',
             arguments=['--frame-id', 'world', '--child-frame-id', 'panda_link0']),
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='log',
             parameters=[moveit_config.robot_description]),
        Node(package='moveit_ros_move_group', executable='move_group', output='screen',
             parameters=[moveit_config.to_dict(), {'publish_robot_description_semantic': True}]),
        Node(package='rviz2', executable='rviz2', output='log', condition=IfCondition(rviz),
             arguments=['-d', os.path.join(share, 'rviz', 'ppp.rviz')],
             parameters=[moveit_config.robot_description, moveit_config.robot_description_semantic,
                         moveit_config.robot_description_kinematics, moveit_config.planning_pipelines]),
    ])
