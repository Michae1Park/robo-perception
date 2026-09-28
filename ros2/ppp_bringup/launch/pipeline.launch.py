"""Full ROS 2 side of the pipeline (Isaac Sim runs separately: isaac_sim/scene.py).

  ros2 launch ppp_bringup pipeline.launch.py rviz:=true [yoloe_engine:=/path/yoloe.engine]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('ppp_bringup')
    lc = LaunchConfiguration
    sv = lambda n: ParameterValue(LaunchConfiguration(n), value_type=str)  # noqa: E731 - '' stays a string
    common = {'scene_config': sv('scene_config'), 'assets_dir': sv('assets_dir')}
    args = [
        DeclareLaunchArgument('scene_config', default_value=os.path.join(share, 'config', 'scene.yaml')),
        DeclareLaunchArgument('assets_dir', default_value=os.environ.get('PPP_ASSETS', '')),
        DeclareLaunchArgument('yoloe_weights', default_value='yoloe-11l-seg.pt'),
        DeclareLaunchArgument('yoloe_engine', default_value='', description='TensorRT engine; empty = PyTorch'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('foundationpose', default_value='true'),
        DeclareLaunchArgument('fp_models_dir', default_value=os.path.join(
            os.environ.get('ISAAC_ROS_WS', '/workspaces/isaac_ros-dev'), 'isaac_ros_assets', 'models',
            'foundationpose')),
        DeclareLaunchArgument('rviz', default_value='false'),
    ]
    include = lambda f, a: IncludeLaunchDescription(  # noqa: E731
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', f)), launch_arguments=a.items())
    return LaunchDescription(args + [
        include('moveit.launch.py', {'rviz': lc('rviz')}),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'foundationpose.launch.py')),
            launch_arguments={'scene_config': lc('scene_config'), 'assets_dir': lc('assets_dir'),
                              'fp_models_dir': lc('fp_models_dir')}.items(),
            condition=IfCondition(lc('foundationpose'))),
        Node(package='ppp_perception', executable='detector_node', output='screen',
             parameters=[common, {'weights': sv('yoloe_weights'), 'engine': sv('yoloe_engine'),
                                  'device': sv('device')}]),
        Node(package='ppp_perception', executable='pose_node', output='screen', parameters=[common]),
        Node(package='ppp_perception', executable='scene_node', output='screen', parameters=[common]),
        Node(package='ppp_manipulation', executable='grasp_node', output='screen', parameters=[common]),
        Node(package='ppp_manipulation', executable='place_node', output='screen', parameters=[common]),
        Node(package='ppp_manipulation', executable='executor_node', output='screen', parameters=[common]),
        Node(package='ppp_manipulation', executable='task_manager', output='screen', parameters=[common]),
    ])
