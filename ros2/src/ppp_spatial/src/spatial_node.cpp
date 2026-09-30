// Stage 3 as a ROS node (D-011, D-019): latest fixed-camera depth -> points in the robot base frame -> table plane
// (ppp_geometry RANSAC, normal within max_tilt_deg of base +z) -> occupancy grid over the table zone + the points
// standing on the table. Request-driven (D-016): the ~AnalyzeTable service does the work; the grid is also
// published for viewing. Same algorithm as vision/spatial.py, but in the calibrated base frame instead of a
// frame made up from the plane.
#include <Eigen/Geometry>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>

#include "ppp_geometry/ransac.hpp"
#include "ppp_interfaces/srv/analyze_table.hpp"

using AnalyzeTable = ppp_interfaces::srv::AnalyzeTable;
using ppp_geometry::Points;

namespace
{
double percentile(std::vector<double> v, double q)
{
  const std::size_t k = std::min(v.size() - 1, static_cast<std::size_t>(q / 100.0 * (v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}
}  // namespace

class SpatialNode : public rclcpp::Node
{
public:
  SpatialNode()
  : Node("spatial_node"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
  {
    base_frame_ = declare_parameter("base_frame", "panda_link0");
    stride_ = declare_parameter("stride", 2);
    ransac_.dist_thresh = declare_parameter("ransac_dist", 0.006);
    ransac_.iters = declare_parameter("ransac_iters", 300);
    ransac_.min_inliers = declare_parameter("min_inliers", 500);
    ransac_.max_tilt_deg = declare_parameter("max_tilt_deg", 10.0);
    resolution_ = declare_parameter("resolution", 0.01);
    height_thresh_ = declare_parameter("height_thresh", 0.012);
    max_height_ = declare_parameter("max_height", 0.6);
    max_range_ = declare_parameter("max_range", 2.0);

    auto qos = rclcpp::SensorDataQoS().reliable();
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "depth/image_raw", qos, [this](sensor_msgs::msg::Image::ConstSharedPtr m) {
        std::lock_guard<std::mutex> l(mtx_);
        depth_ = m;
      });
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "depth/camera_info", qos, [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
        std::lock_guard<std::mutex> l(mtx_);
        info_ = m;
      });
    grid_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("~/grid", rclcpp::QoS(1).transient_local());
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/above_table", rclcpp::QoS(1).transient_local());
    srv_ = create_service<AnalyzeTable>(
      "~/analyze_table", [this](const AnalyzeTable::Request::SharedPtr, AnalyzeTable::Response::SharedPtr res) {
        analyze(*res);
      });
    RCLCPP_INFO(get_logger(), "ready: ~/analyze_table on %s", depth_sub_->get_topic_name());
  }

private:
  void analyze(AnalyzeTable::Response & res)
  {
    const auto t0 = std::chrono::steady_clock::now();
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    {
      std::lock_guard<std::mutex> l(mtx_);
      depth = depth_;
      info = info_;
    }
    if (!depth || !info) {
      res.message = "no depth image / camera_info received yet";
      return;
    }
    if (depth->encoding != "16UC1") {
      res.message = "expected 16UC1 depth (mm), got " + depth->encoding;
      return;
    }
    Eigen::Isometry3d T_base_cam;
    try {
      T_base_cam = tf2::transformToEigen(
        tf_buffer_.lookupTransform(base_frame_, depth->header.frame_id, depth->header.stamp,
                                   rclcpp::Duration::from_seconds(0.5)));
    } catch (const tf2::TransformException & e) {
      res.message = std::string("no extrinsics: ") + e.what();
      return;
    }

    // back-project every stride-th pixel into the base frame
    const double fx = info->k[0], fy = info->k[4], cx = info->k[2], cy = info->k[5];
    std::vector<Eigen::Vector3d> pts;
    pts.reserve((depth->width / stride_ + 1) * (depth->height / stride_ + 1));
    for (uint32_t v = 0; v < depth->height; v += stride_) {
      const auto * row = reinterpret_cast<const uint16_t *>(&depth->data[v * depth->step]);
      for (uint32_t u = 0; u < depth->width; u += stride_) {
        const double z = row[u] * 1e-3;
        if (z <= 0.0 || z > max_range_) {
          continue;
        }
        pts.push_back(T_base_cam * Eigen::Vector3d((u - cx) * z / fx, (v - cy) * z / fy, z));
      }
    }
    Points P(pts.size(), 3);
    for (std::size_t i = 0; i < pts.size(); ++i) {
      P.row(i) = pts[i];
    }
    ransac_.up = Eigen::Vector3d::UnitZ();
    const auto fit = ppp_geometry::fit_plane_ransac(P, ransac_);
    if (!fit) {
      res.message = "RANSAC found no table plane";
      return;
    }
    const Eigen::Vector4d & pl = fit->plane;
    const Eigen::VectorXd h = ppp_geometry::height_above(pl, P);

    // table zone: 2nd-98th percentile of the table points' x / y (drops stray inliers far away)
    std::vector<double> xs, ys;
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
      if (fit->inliers[i]) {
        xs.push_back(P(i, 0));
        ys.push_back(P(i, 1));
      }
    }
    const double x0 = percentile(xs, 2), x1 = percentile(xs, 98), y0 = percentile(ys, 2), y1 = percentile(ys, 98);
    const int nx = static_cast<int>(std::ceil((x1 - x0) / resolution_)), ny = static_cast<int>(std::ceil((y1 - y0) / resolution_));

    // occupancy: FREE where only table points fell, OCCUPIED with >= 3 points above it (robust to flying pixels)
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.frame_id = base_frame_;
    grid.header.stamp = depth->header.stamp;
    grid.info.resolution = resolution_;
    grid.info.width = nx;
    grid.info.height = ny;
    grid.info.origin.position.x = x0;
    grid.info.origin.position.y = y0;
    grid.info.origin.position.z = -pl[3] / pl[2];
    grid.info.origin.orientation.w = 1.0;
    grid.data.assign(static_cast<std::size_t>(nx) * ny, -1);
    std::vector<int> counts(grid.data.size(), 0);
    std::vector<Eigen::Vector3d> above;
    for (Eigen::Index i = 0; i < P.rows(); ++i) {
      const bool obstacle = h[i] >= height_thresh_ && h[i] < max_height_;
      if (obstacle) {
        above.push_back(P.row(i));
      }
      const int ix = static_cast<int>((P(i, 0) - x0) / resolution_), iy = static_cast<int>((P(i, 1) - y0) / resolution_);
      if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) {
        continue;
      }
      const std::size_t c = static_cast<std::size_t>(iy) * nx + ix;
      if (std::abs(h[i]) < height_thresh_ && grid.data[c] < 0) {
        grid.data[c] = 0;
      } else if (obstacle && ++counts[c] >= 3) {
        grid.data[c] = 100;
      }
    }

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = grid.header;
    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(above.size());
    sensor_msgs::PointCloud2Iterator<float> it(cloud, "x");
    for (const auto & p : above) {
      it[0] = static_cast<float>(p.x());
      it[1] = static_cast<float>(p.y());
      it[2] = static_cast<float>(p.z());
      ++it;
    }

    res.success = true;
    std::copy(pl.data(), pl.data() + 4, res.plane.begin());
    const double xc = 0.5 * (x0 + x1), yc = 0.5 * (y0 + y1);
    res.table_z = static_cast<float>(-(pl[0] * xc + pl[1] * yc + pl[3]) / pl[2]);
    res.inliers = static_cast<uint32_t>(fit->count);
    res.grid = grid;
    res.above_table = cloud;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    char buf[256];
    std::snprintf(buf, sizeof(buf), "table z %.4f m, tilt %.2f deg, %zu/%zu points on it, %zu above, %.0f ms",
                  res.table_z, std::acos(std::min(1.0, pl[2])) * 180.0 / M_PI, fit->count, pts.size(), above.size(), ms);
    res.message = buf;
    RCLCPP_INFO(get_logger(), "%s", buf);
    grid_pub_->publish(grid);
    cloud_pub_->publish(cloud);
  }

  std::string base_frame_;
  int stride_;
  ppp_geometry::RansacParams ransac_;
  double resolution_, height_thresh_, max_height_, max_range_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::mutex mtx_;
  sensor_msgs::msg::Image::ConstSharedPtr depth_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Service<AnalyzeTable>::SharedPtr srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // the service callback blocks on TF while the listener's callbacks need to run: two threads
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  auto node = std::make_shared<SpatialNode>();
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
