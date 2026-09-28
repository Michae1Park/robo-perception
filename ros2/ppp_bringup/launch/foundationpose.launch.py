"""Isaac ROS FoundationPose, one instance per object mesh, namespaced /fp/<object>/.

Each instance loads the same refine/score TensorRT engines (build them once with
scripts/build_fp_engines.sh) and its own textured mesh from <assets>/ycb/<object>/.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ppp_common.config import SceneConfig

DEFAULT_MODELS = os.path.join(os.environ.get('ISAAC_ROS_WS', '/workspaces/isaac_ros-dev'),
                              'isaac_ros_assets', 'models', 'foundationpose')


def fp_node(cfg, name, models):
    return ComposableNode(
        name='foundationpose_' + name,
        namespace='fp/' + name,
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': cfg.mesh_path(name),
            'texture_path': cfg.texture_path(name),
            'refine_model_file_path': os.path.join(models, 'refine_model.onnx'),
            'refine_engine_file_path': os.path.join(models, 'refine_trt_engine.plan'),
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
            'score_model_file_path': os.path.join(models, 'score_model.onnx'),
            'score_engine_file_path': os.path.join(models, 'score_trt_engine.plan'),
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
            'tf_frame_name': 'fp_' + name,
        }])


def launch_setup(context):
    cfg = SceneConfig(context.launch_configurations['scene_config'],
                      context.launch_configurations['assets_dir'])
    models = context.launch_configurations['fp_models_dir']
    only = [o for o in context.launch_configurations['objects'].split(',') if o]
    names = [n for n in cfg.object_names if not only or n in only]
    return [ComposableNodeContainer(
        name='foundationpose_container', namespace='', package='rclcpp_components',
        executable='component_container_mt', output='screen',
        composable_node_descriptions=[fp_node(cfg, n, models) for n in names])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene_config', default_value=''),
        DeclareLaunchArgument('assets_dir', default_value=''),
        DeclareLaunchArgument('fp_models_dir', default_value=DEFAULT_MODELS),
        DeclareLaunchArgument('objects', default_value='', description='comma separated subset; empty = all'),
        OpaqueFunction(function=launch_setup),
    ])
