"""Runs INSIDE the ppp-isaac-ros:4.5 container (docker/foundationpose.sh): one Isaac ROS FoundationPose pipeline
(FoundationPoseNode + refine and score TensorRT nodes) in /ppp/fp for every object (D-043). The networks don't depend
on the object; pose_node sets the node's mesh_file_path before each request and the node reloads it. Only the
requested object's mask ever reaches it (detect_node publishes on request, D-016).

  inputs  /ppp/fp/image (rgb8), depth_image (32FC1 m), camera_info, segmentation (mono8)
  output  /ppp/fp/output (vision_msgs/Detection3DArray, pose of the mesh frame in the camera frame)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

NS = '/ppp/fp'   # composable nodes don't inherit the container's namespace


def ycb_dir(repo, target):
    ycb = os.path.join(repo, 'assets', 'ycb')
    return os.path.join(ycb, next(d for d in sorted(os.listdir(ycb)) if d.split('_', 1)[-1] == target))


def trt(name, engine, outputs, max_batch):
    return ComposableNode(
        name=name + '_trt', namespace=NS, package='isaac_ros_tensor_rt', plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{'engine_file_path': engine, 'force_engine_update': False, 'verbose': False,
                     'input_tensor_names': ['input_tensor1', 'input_tensor2'], 'input_binding_names': ['input1', 'input2'],
                     'output_tensor_names': outputs, 'output_binding_names': ['output1', 'output2'][:len(outputs)],
                     'max_batch_size': max_batch}],
        remappings=[('tensor_pub', name + '/tensor_pub'), ('tensor_sub', name + '/tensor_sub')])


def pipeline(context):
    repo = LaunchConfiguration('repo').perform(context)
    engines = os.path.join(repo, 'models', 'isaac_ros', 'foundationpose')
    d = ycb_dir(repo, LaunchConfiguration('mesh').perform(context))
    fp = ComposableNode(
        name='foundationpose', namespace=NS, package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{'mesh_file_path': os.path.join(d, 'textured.obj'),   # the texture comes with the mesh (.mtl)
                     'tf_frame_name': 'fp_object',
                     'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
                     'score_input_tensor_names': ['input_tensor1', 'input_tensor2']}],
        remappings=[('pose_estimation/image', 'image'), ('pose_estimation/depth_image', 'depth_image'),
                    ('pose_estimation/camera_info', 'camera_info'),
                    ('pose_estimation/segmentation', 'segmentation'), ('pose_estimation/output', 'output')])
    return [ComposableNodeContainer(
        name='container', namespace=NS, package='rclcpp_components', executable='component_container_mt',
        output='screen', composable_node_descriptions=[
            trt('refine', os.path.join(engines, 'refine_trt_engine.plan'), ['output_tensor1', 'output_tensor2'], 42),
            trt('score', os.path.join(engines, 'score_trt_engine.plan'), ['output_tensor'], 252),
            fp])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('repo', description='repo path (mounted at the same path)'),
        DeclareLaunchArgument('mesh', default_value='mustard_bottle', description='object loaded at start'),
        OpaqueFunction(function=pipeline),
    ])
