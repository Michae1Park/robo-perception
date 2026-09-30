"""MoveIt 2 for the Panda (moveit_resources_panda_moveit_config): move_group with OMPL + Pilz, executing through
panda_arm_controller (FollowJointTrajectory) and panda_hand_controller (GripperCommand), then the cell's fixed
collision objects (table, pedestal, shelf, floor) from config.yaml.

  ros2 launch ppp_bringup moveit.launch.py
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

REPO = os.environ.get('PPP_REPO') or str(Path(os.path.realpath(__file__)).parents[4])


def generate_launch_description():
    moveit = (MoveItConfigsBuilder('moveit_resources_panda')
              .trajectory_execution(file_path='config/gripper_moveit_controllers.yaml')
              .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
              .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner'])
              .to_moveit_configs())
    return LaunchDescription([
        SetEnvironmentVariable('PPP_REPO', REPO),
        Node(package='moveit_ros_move_group', executable='move_group', output='screen',
             parameters=[moveit.to_dict(), {'use_sim_time': True,
                                            # the sim arm tracks a few mrad behind its command at rest
                                            'trajectory_execution.allowed_start_tolerance': 0.02}]),
        Node(package='ppp_bringup', executable='planning_scene.py', output='screen',
             prefix=os.path.join(REPO, '.venv', 'bin', 'python'), parameters=[{'use_sim_time': True}]),
    ])
