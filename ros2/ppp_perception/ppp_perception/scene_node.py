"""③ Spatial perception: table plane (RANSAC) + free space of the place zone.

Service   /perception/analyze_scene   ppp_interfaces/AnalyzeScene
Publishes /perception/place_zone_grid (nav_msgs/OccupancyGrid, latched)
          /perception/scene_markers   (visualization_msgs/MarkerArray: plane, obstacles)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ppp_common import transforms as tf
from ppp_common.config import SceneConfig
from ppp_common.ros_conv import image_to_numpy, transform_to_T
from ppp_common.ros_utils import Stopwatch
from ppp_interfaces.srv import AnalyzeScene

from .core import free_space as fs
from .core.plane import fit_plane_ransac, plane_z_at


def grid_to_msg(grid, zone, resolution, z, frame_id, stamp):
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.info.map_load_time = stamp
    msg.info.resolution = float(resolution)
    msg.info.height, msg.info.width = grid.shape
    msg.info.origin.position.x = float(zone[0][0])
    msg.info.origin.position.y = float(zone[1][0])
    msg.info.origin.position.z = float(z)
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.astype(np.int8).flatten().tolist()
    return msg


class SceneNode(Node):
    def __init__(self):
        super().__init__('scene_analyzer')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('stride', 2)
        self.declare_parameter('ransac_iters', 300)
        self.declare_parameter('ransac_threshold', 0.006)
        self.cfg = SceneConfig(self.get_parameter('scene_config').value)
        self.stride = int(self.get_parameter('stride').value)
        self.iters = int(self.get_parameter('ransac_iters').value)
        self.thresh = float(self.get_parameter('ransac_threshold').value)
        pc = self.cfg.raw['place']
        self.res = float(pc['grid_resolution'])
        self.height_thresh = float(pc['height_threshold'])
        self.zone = self.cfg.zone('place')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/perception/place_zone_grid', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/scene_markers', latched)
        self.srv = self.create_service(AnalyzeScene, '/perception/analyze_scene', self.on_request)
        self.get_logger().info('scene analyzer ready')

    def camera_in_world(self, frame_id):
        try:
            t = self.tf_buffer.lookup_transform(self.cfg.world_frame, frame_id, rclpy.time.Time())
            return transform_to_T(t.transform)
        except Exception:
            return self.cfg.T_world_camera()

    def on_request(self, req, res):
        sw = Stopwatch()
        depth = image_to_numpy(req.depth)
        K = np.array(req.camera_info.k).reshape(3, 3)
        pts = tf.transform_points(self.camera_in_world(req.depth.header.frame_id or self.cfg.camera['frame_id']),
                                  tf.depth_to_points(depth, K, self.stride))
        # table-sized crop around the workspace keeps RANSAC fast and ignores the floor
        t = self.cfg.raw['table']
        lo = np.array(t['center']) - np.array(t['size']) / 2.0
        hi = np.array(t['center']) + np.array(t['size']) / 2.0
        m = (pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1]) \
            & (pts[:, 2] > lo[2] - 0.1) & (pts[:, 2] < hi[2] + 0.6)
        pts = pts[m]
        plane, inliers = fit_plane_ransac(pts, self.thresh, self.iters)
        if plane is None:
            res.success, res.message = False, 'no table plane found (%d points)' % len(pts)
            res.latency_ms = sw.ms()
            return res
        grid = fs.occupancy_grid(pts, plane, self.zone, self.res, self.height_thresh)
        c = [(self.zone[0][0] + self.zone[0][1]) / 2.0, (self.zone[1][0] + self.zone[1][1]) / 2.0]
        z = plane_z_at(plane, *c)
        stamp = req.depth.header.stamp
        res.success = True
        res.message = 'ok'
        res.plane = [float(v) for v in plane]
        res.table_height = float(z)
        res.plane_inliers = int(inliers.sum())
        res.place_zone = grid_to_msg(grid, self.zone, self.res, z, self.cfg.world_frame, stamp)
        res.latency_ms = sw.ms()
        self.grid_pub.publish(res.place_zone)
        self.marker_pub.publish(self.markers(pts[inliers], grid, z, stamp))
        free = int((grid == fs.FREE).sum())
        self.get_logger().info('scene: table z=%.4f, %d inliers, place zone %d/%d free cells, %.1f ms'
                               % (z, res.plane_inliers, free, grid.size, res.latency_ms))
        return res

    def markers(self, table_pts, grid, z, stamp):
        arr = MarkerArray()
        plane = Marker()
        plane.header.frame_id = self.cfg.world_frame
        plane.header.stamp = stamp
        plane.ns, plane.id, plane.type = 'table_plane', 0, Marker.POINTS
        plane.scale.x = plane.scale.y = 0.004
        plane.color.g, plane.color.b, plane.color.a = 0.6, 1.0, 0.6
        for p in table_pts[::max(1, len(table_pts) // 4000)]:
            plane.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        arr.markers.append(plane)
        occ = Marker()
        occ.header = plane.header
        occ.ns, occ.id, occ.type = 'place_obstacles', 1, Marker.CUBE_LIST
        occ.scale.x = occ.scale.y = self.res
        occ.scale.z = 0.002
        occ.color.r, occ.color.a = 1.0, 0.8
        centers = fs.cell_centers(self.zone, self.res, grid.shape)
        for (x, y) in centers[grid != fs.FREE]:
            occ.points.append(Point(x=float(x), y=float(y), z=float(z) + 0.002))
        arr.markers.append(occ)
        return arr


def main():
    rclpy.init()
    node = SceneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
