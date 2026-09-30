#!/usr/bin/env python3
"""Load the fixed parts of the cell into MoveIt's planning scene: table, robot pedestal, pantry (posts + boards),
floor. Geometry from config.yaml `sim:` - in a real cell these would be measured once (D-010). Objects on the table
and shelf are NOT added: perception finds them. Runs once and exits.
"""
import os
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

sys.path.insert(0, os.environ['PPP_REPO'])
from vision import load_config, shelf_boxes  # noqa: E402

BASE = 'panda_link0'


def box(name, center, size):
    o = CollisionObject()
    o.header.frame_id, o.id, o.operation = BASE, name, CollisionObject.ADD
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, center)
    p.orientation.w = 1.0
    o.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(s) for s in size]))
    o.primitive_poses.append(p)
    return o


def cell_objects(cfg):
    tw, tl, th = cfg['table_size']
    (tx, ty) = cfg['table_center']
    objs = [box('table', [tx, ty, -th / 2], [tw, tl, th]),
            box('pedestal', [0.0, 0.0, -th / 2 - 0.005], [0.2, 0.2, th]),     # top 5 mm below the base: no contact
            box('floor', [0.0, 0.0, -th - 0.01], [4.0, 4.0, 0.02])]
    return objs + [box('shelf_' + name, center, size) for name, center, size in shelf_boxes(cfg['shelf'], -th)]


def main():
    rclpy.init()
    node = rclpy.create_node('planning_scene_loader')
    cli = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
    while not cli.wait_for_service(timeout_sec=2.0):
        node.get_logger().info('waiting for move_group /apply_planning_scene ...')
    scene = PlanningScene(is_diff=True)
    scene.world.collision_objects = cell_objects(load_config()['sim'])
    fut = cli.call_async(ApplyPlanningScene.Request(scene=scene))
    rclpy.spin_until_future_complete(node, fut)
    ok = fut.result() is not None and fut.result().success
    node.get_logger().info('%s %d collision objects: %s' % ('added' if ok else 'FAILED to add',
                           len(scene.world.collision_objects), ', '.join(o.id for o in scene.world.collision_objects)))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
