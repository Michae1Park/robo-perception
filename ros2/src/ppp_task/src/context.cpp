#include "ppp_task/context.hpp"

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <moveit/robot_state/conversions.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <filesystem>
#include <set>
#include <thread>

namespace ppp_task
{

Context::Context(const rclcpp::Node::SharedPtr & n)
: node(n)
{
  auto home = node->declare_parameter<std::vector<double>>(
    "home_joints", {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785});
  auto look = node->declare_parameter<std::vector<double>>(
    "look_joints", {-1.515, -0.826, -0.008, -1.066, -0.045, 0.909, 0.793});
  named_joints = {{"home", home}, {"look", look}};
  velocity_scaling = node->declare_parameter("velocity_scaling", 0.4);
  linear_velocity_scaling = node->declare_parameter("linear_velocity_scaling", 0.15);
  carry_velocity_scaling = node->declare_parameter("carry_velocity_scaling", 0.2);   // gentle with it in the hand
  planning_time = node->declare_parameter("planning_time", 5.0);
  ycb_dir = node->declare_parameter("ycb_dir", std::string(""));
  const double tcp_offset = node->declare_parameter("tcp_offset", 0.1034);   // flange -> fingertips along z

  // panda_link8 -> panda_hand is -45 deg about z (URDF); the TCP shares the hand's axes (fingers along y)
  flange_to_tcp = Eigen::Translation3d(0, 0, tcp_offset) * Eigen::AngleAxisd(-M_PI / 4, Eigen::Vector3d::UnitZ());
  hand_to_tcp = Eigen::Translation3d(0, 0, tcp_offset) * Eigen::Isometry3d::Identity();

  arm = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "panda_arm");
  arm->setPoseReferenceFrame(base_frame);
  arm->setEndEffectorLink(flange);
  arm->setPlanningTime(planning_time);
  arm->setNumPlanningAttempts(5);
  arm->setMaxVelocityScalingFactor(velocity_scaling);
  arm->setMaxAccelerationScalingFactor(velocity_scaling);

  gripper = rclcpp_action::create_client<control_msgs::action::GripperCommand>(node, "/panda_hand_controller/gripper_cmd");
  estimate_pose = node->create_client<ppp_interfaces::srv::EstimatePose>(
    node->declare_parameter("pose_service", std::string("/pose_node/estimate_pose")));
  plan_grasps = node->create_client<ppp_interfaces::srv::PlanGrasps>("/grasp_node/plan_grasps");
  get_placements = node->create_client<ppp_interfaces::srv::GetPlacements>("/place_node/get_placements");
  compute_ik = node->create_client<moveit_msgs::srv::GetPositionIK>("/compute_ik");
  check_state = node->create_client<moveit_msgs::srv::GetStateValidity>("/check_state_validity");
  js_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10, [this](sensor_msgs::msg::JointState::ConstSharedPtr m) {
      std::lock_guard<std::mutex> l(js_mtx_);
      std::vector<double> arm(7, 0.0);
      for (std::size_t i = 0; i < m->name.size(); ++i) {
        if (m->name[i] == "panda_finger_joint1") {
          finger_ = m->position[i];
        } else if (m->name[i].rfind("panda_joint", 0) == 0) {
          arm[std::stoi(m->name[i].substr(11)) - 1] = m->position[i];
        }
      }
      arm_pos_ = arm;
      arm_stamp_ = m->header.stamp;
    });
}

geometry_msgs::msg::PoseStamped Context::tcp_to_flange(const geometry_msgs::msg::PoseStamped & tcp) const
{
  Eigen::Isometry3d T;
  tf2::fromMsg(tcp.pose, T);
  geometry_msgs::msg::PoseStamped out = tcp;
  out.pose = tf2::toMsg(Eigen::Isometry3d(T * flange_to_tcp.inverse()));
  return out;
}

Eigen::Isometry3d Context::current_tcp() const
{
  Eigen::Isometry3d T;
  tf2::fromMsg(arm->getCurrentPose(flange).pose, T);
  return T * flange_to_tcp;
}

double Context::finger_position()
{
  std::lock_guard<std::mutex> l(js_mtx_);
  return finger_;
}

bool Context::wait_settled(double max_wait_s)
{
  // at rest = no joint moved more than 0.5 mrad over 0.1 s of sim time (position-based: Isaac's reported joint
  // velocities are not reliable when something blocks the robot, see I-019)
  auto clock = node->get_clock();
  const auto start = clock->now();
  std::vector<double> ref;
  rclcpp::Time ref_t = start;
  while ((clock->now() - start).seconds() < max_wait_s) {
    std::vector<double> q;
    {
      std::lock_guard<std::mutex> l(js_mtx_);
      q = arm_pos_;
    }
    const auto now = clock->now();
    if (!q.empty()) {
      if (ref.empty()) {
        ref = q;
        ref_t = now;
      } else {
        double d = 0.0;
        for (std::size_t i = 0; i < q.size(); ++i) {
          d = std::max(d, std::abs(q[i] - ref[i]));
        }
        if (d > 5e-4) {
          ref = q;
          ref_t = now;
        } else if ((now - ref_t).seconds() > 0.1) {
          return true;
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  RCLCPP_WARN(node->get_logger(), "arm still moving after %.1f s", max_wait_s);
  return false;
}

void Context::start_from_current_state()
{
  auto state = arm->getCurrentState(1.0);
  if (!state) {
    arm->setStartStateToCurrentState();
    return;
  }
  state->enforceBounds();
  arm->setStartState(*state);
}

std::optional<moveit_msgs::msg::AttachedCollisionObject> Context::held_object(const std::string & target,
                                                                               const Eigen::Isometry3d & T_obj_tcp)
{
  std::string mesh_file;
  for (const auto & e : std::filesystem::directory_iterator(ycb_dir)) {
    const auto n = e.path().filename().string();
    if (n.size() > target.size() && n.compare(n.size() - target.size(), target.size(), target) == 0) {
      mesh_file = (e.path() / "textured.obj").string();
    }
  }
  if (mesh_file.empty()) {
    RCLCPP_ERROR(node->get_logger(), "no mesh for %s in %s", target.c_str(), ycb_dir.c_str());
    return std::nullopt;
  }
  std::unique_ptr<shapes::Mesh> mesh(shapes::createMeshFromResource("file://" + mesh_file));
  if (!mesh) {
    return std::nullopt;
  }
  shapes::ShapeMsg shape;
  shapes::constructMsgFromShape(mesh.get(), shape);
  moveit_msgs::msg::AttachedCollisionObject aco;
  aco.link_name = "panda_hand";
  aco.touch_links = {"panda_hand", "panda_leftfinger", "panda_rightfinger"};
  aco.object.id = "held_" + target;
  aco.object.header.frame_id = "panda_hand";
  aco.object.meshes.push_back(boost::get<shape_msgs::msg::Mesh>(shape));
  aco.object.mesh_poses.push_back(tf2::toMsg(Eigen::Isometry3d(hand_to_tcp * T_obj_tcp.inverse())));
  aco.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return aco;
}

static std::shared_ptr<moveit_msgs::srv::GetPositionIK::Request> ik_request(
  Context & c, const Eigen::Isometry3d & tcp, const moveit_msgs::msg::AttachedCollisionObject * held, bool avoid,
  double timeout = 0.1)
{
  auto req = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
  auto & ik = req->ik_request;
  ik.group_name = c.arm->getName();
  ik.ik_link_name = c.flange;
  ik.avoid_collisions = avoid;
  ik.timeout = rclcpp::Duration::from_seconds(timeout);
  ik.pose_stamped.header.frame_id = c.base_frame;
  ik.pose_stamped.pose = tf2::toMsg(Eigen::Isometry3d(tcp * c.flange_to_tcp.inverse()));
  moveit::core::robotStateToRobotStateMsg(*c.arm->getCurrentState(), ik.robot_state);
  if (held) {
    ik.robot_state.attached_collision_objects.push_back(*held);
  }
  return req;
}

bool Context::reachable(const Eigen::Isometry3d & tcp, const moveit_msgs::msg::AttachedCollisionObject * held)
{
  auto res = call<moveit_msgs::srv::GetPositionIK>(compute_ik, ik_request(*this, tcp, held, true), 5.0);
  return res && res->error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
}

std::string Context::current_contacts()
{
  auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
  moveit::core::robotStateToRobotStateMsg(*arm->getCurrentState(), req->robot_state);
  req->group_name = arm->getName();
  auto v = call<moveit_msgs::srv::GetStateValidity>(check_state, req, 5.0);
  std::set<std::string> pairs;
  if (v) {
    for (const auto & c : v->contacts) {
      pairs.insert(c.contact_body_1 + "-" + c.contact_body_2);
    }
  }
  std::string out;
  for (const auto & p : pairs) {
    out += (out.empty() ? "" : " ") + p;
  }
  return out;
}

std::string Context::why_unreachable(const Eigen::Isometry3d & tcp,
                                     const moveit_msgs::msg::AttachedCollisionObject * held)
{
  auto ik = call<moveit_msgs::srv::GetPositionIK>(compute_ik, ik_request(*this, tcp, held, false, 0.3), 5.0);
  if (!ik || ik->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
    return "no IK (out of reach)";
  }
  auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
  req->robot_state = ik->solution;
  req->group_name = arm->getName();
  auto v = call<moveit_msgs::srv::GetStateValidity>(check_state, req, 5.0);
  if (!v) {
    return "?";
  }
  std::set<std::string> pairs;
  for (const auto & c : v->contacts) {
    pairs.insert(c.contact_body_1 + "-" + c.contact_body_2);
  }
  std::string out = v->valid ? "valid at this IK solution (IK with collisions found none)" : "collides:";
  for (const auto & p : pairs) {
    out += " " + p;
  }
  return out;
}

}  // namespace ppp_task
