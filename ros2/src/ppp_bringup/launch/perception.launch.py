"""Perception, one node per stage (D-010), all request-driven (D-016):

  detect_node   (Python, YOLOE)          fixed camera        ~/detect
  FoundationPose (Isaac ROS 4.5, Docker)  one per target      /ppp/fp/<target>/...   (foundationpose:=true)
  pose_node     (Python)                  detect + FP + TF    ~/estimate_pose
  spatial_node  (C++, ppp_geometry)       fixed camera        ~/analyze_table
  grasp_node    (Python)                  spatial + pose      ~/plan_grasps
  place_node    (Python, C++ RANSAC)      wrist camera        ~/get_placements
  eval_node     (Python, ground truth)    evaluation only     ~/report            (eval:=true)

  ros2 launch ppp_bringup perception.launch.py [foundationpose:=false] [eval:=false]
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO = os.environ.get('PPP_REPO') or str(Path(os.path.realpath(__file__)).parents[4])
VENV_PY = os.path.join(REPO, '.venv', 'bin', 'python')
TARGETS = ['mustard_bottle', 'tomato_soup_can']
FIXED = '/camera/camera'
WRIST = '/wrist_camera/camera'


def config():
    import yaml
    with open(os.path.join(REPO, 'config.yaml')) as f:
        return yaml.safe_load(f)


def py_node(name, remappings=(), params=None, condition=None):
    """Python nodes run on the repo's .venv (torch, ultralytics) with ROS sourced (I-012)."""
    return Node(package='ppp_perception', executable=name, name=name, output='screen', prefix=VENV_PY,
                remappings=list(remappings), parameters=[{'use_sim_time': True, **(params or {})}],
                condition=condition)


def generate_launch_description():
    sp = config()['spatial']
    return LaunchDescription([
        DeclareLaunchArgument('foundationpose', default_value='true',
                              description='start the Isaac ROS FoundationPose container (docker/foundationpose.sh)'),
        DeclareLaunchArgument('eval', default_value='false', description='start eval_node (reads /sim/gt/*)'),
        SetEnvironmentVariable('PPP_REPO', REPO),
        py_node('detect_node', [('color/image_raw', FIXED + '/color/image_raw'),
                                ('color/camera_info', FIXED + '/color/camera_info'),
                                ('depth/image_raw', FIXED + '/aligned_depth_to_color/image_raw')],
                {'targets': TARGETS}),
        py_node('pose_node', params={'targets': TARGETS}),
        Node(package='ppp_spatial', executable='spatial_node', name='spatial_node', output='screen',
             remappings=[('depth/image_raw', FIXED + '/aligned_depth_to_color/image_raw'),
                         ('depth/camera_info', FIXED + '/aligned_depth_to_color/camera_info')],
             parameters=[{'use_sim_time': True, 'ransac_dist': sp['ransac_dist'], 'ransac_iters': sp['ransac_iters'],
                          'min_inliers': sp['min_inliers'], 'resolution': sp['resolution'],
                          'height_thresh': sp['height_thresh'], 'max_height': sp['max_height']}]),
        py_node('grasp_node', params={'max_approach_tilt_deg': 95.0}),   # side grasps: the shelf needs them
        py_node('place_node', [('depth/image_raw', WRIST + '/aligned_depth_to_color/image_raw'),
                               ('depth/camera_info', WRIST + '/aligned_depth_to_color/camera_info')]),
        py_node('eval_node', params={'targets': TARGETS}, condition=IfCondition(LaunchConfiguration('eval'))),
        ExecuteProcess(cmd=[os.path.join(REPO, 'docker', 'foundationpose.sh')] + TARGETS, output='screen',
                       condition=IfCondition(LaunchConfiguration('foundationpose'))),
    ])
