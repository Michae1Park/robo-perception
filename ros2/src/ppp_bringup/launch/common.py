"""Shared by the launch files: the repo, config.yaml, and MoveIt's Panda config with the cell's longer fingers."""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from moveit_configs_utils import MoveItConfigsBuilder

REPO = os.environ.get('PPP_REPO') or str(Path(os.path.realpath(__file__)).parents[4])
sys.path.insert(0, REPO)
from vision.grasp import FINGER_BASE, STOCK_TCP, tcp_offset  # noqa: E402

VENV_PY = os.path.join(REPO, '.venv', 'bin', 'python')
STOCK_TIP = STOCK_TCP + 0.0089 - FINGER_BASE   # finger tip in the finger link frame (53.9 mm)


def config():
    with open(os.path.join(REPO, 'config.yaml')) as f:
        return yaml.safe_load(f)


TARGETS = [n.split('_', 1)[1] for n in config()['objects']]   # what the robot can pick: '006_mustard_bottle' -> 'mustard_bottle'


def long_fingers(urdf, length):
    """Add the sim's finger extensions (sim/cell.py add_finger_extensions) to both finger links: a box continuing
    each finger `length` past its tip, pad face flush with the stock pad (y = 0 in the finger frame)."""
    root = ET.fromstring(urdf)
    z0, z1 = STOCK_TIP - 0.009, STOCK_TIP + length
    for link in root.findall('link'):
        side = {'panda_leftfinger': 1.0, 'panda_rightfinger': -1.0}.get(link.get('name'))
        if side is None:
            continue
        for tag in ('visual', 'collision'):
            el = ET.SubElement(link, tag)
            ET.SubElement(el, 'origin', xyz='0 %g %g' % (side * 0.006, (z0 + z1) / 2), rpy='0 0 0')
            ET.SubElement(ET.SubElement(el, 'geometry'), 'box', size='0.0176 0.012 %g' % (z1 - z0))
    return ET.tostring(root, encoding='unicode')


def moveit_config():
    moveit = (MoveItConfigsBuilder('moveit_resources_panda')
              .trajectory_execution(file_path='config/gripper_moveit_controllers.yaml')
              .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
              .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner', 'stomp'])
              .to_moveit_configs())
    ext = config()['gripper']['finger_extension']
    moveit.robot_description = {'robot_description': long_fingers(moveit.robot_description['robot_description'], ext)}
    # IK: pick_ik, searching globally for the valid solution that moves the joints least from the seed. MoveIt's
    # default KDL restarts from random joint values and returns arbitrary, often flipped configurations (D-037)
    moveit.robot_description_kinematics = {'robot_description_kinematics': {'panda_arm': {
        'kinematics_solver': 'pick_ik/PickIkPlugin', 'kinematics_solver_timeout': 0.05, 'mode': 'global',
        'stop_optimization_on_valid_solution': False, 'position_threshold': 0.001, 'orientation_threshold': 0.01,
        'minimal_displacement_weight': 0.02, 'cost_threshold': 100.0}}}
    # OMPL checks a path for collisions every this fraction of the state space's extent (default 0.005, ~8 cm at
    # the hand): too coarse for the shelf's 2 cm boards
    moveit.planning_pipelines['ompl']['panda_arm']['longest_valid_segment_fraction'] = 0.0005
    return moveit


def tcp():
    return tcp_offset(config()['gripper'])
