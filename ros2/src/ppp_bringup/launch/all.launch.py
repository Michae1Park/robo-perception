"""Everything except the sim (start sim/ros_cell.py first) and the task:

  ros2 launch ppp_bringup all.launch.py [foxglove:=true] [eval:=true] [foundationpose:=true]
  ros2 launch ppp_bringup task.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration

LAUNCH = os.path.join(get_package_share_directory('ppp_bringup'), 'launch')


def include(name, **args):
    return IncludeLaunchDescription(os.path.join(LAUNCH, name + '.launch.py'),
                                    launch_arguments={k: LaunchConfiguration(v) for k, v in args.items()}.items())


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('foxglove', default_value='false'),
        DeclareLaunchArgument('eval', default_value='false'),
        DeclareLaunchArgument('foundationpose', default_value='true'),
        include('robot', foxglove='foxglove'),
        include('moveit'),
        include('perception', eval='eval', foundationpose='foundationpose'),
    ])
