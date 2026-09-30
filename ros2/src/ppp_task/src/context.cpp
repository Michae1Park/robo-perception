#include "ppp_task/context.hpp"

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <moveit/robot_state/conversions.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <thread>

namespace ppp_task
{

Context::Context(const rclcpp::Node::SharedPtr & n)
: node(n)
{
  named_joints["home"] = node->declare_parameter<Joints>("home_joints", {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785});
  named_joints["look"] = node->declare_parameter<Joints>("look_joints", named_joints["home"]);   // from config.yaml
  velocity_scaling = node->declare_parameter("velocity_scaling", 0.4);
  linear_velocity_scaling = node->declare_parameter("linear_velocity_scaling", 0.15);
  carry_velocity_scaling = node->declare_parameter("carry_velocity_scaling", 0.2);   // gentle with it in the hand
  gripper_effort = node->declare_parameter("gripper_effort", 140.0);
  held_padding = node->declare_parameter("held_padding", 0.005);   // margin around the held object in planning
  ycb_dir = node->declare_parameter("ycb_dir", std::string(""));
  const double tcp_offset = node->declare_parameter("tcp_offset", 0.1034);   // panda_hand -> TCP along z

  // panda_link8 -> panda_hand is -45 deg about z (URDF); the TCP shares the hand's axes (fingers along y)
  hand_to_tcp = Eigen::Translation3d(0, 0, tcp_offset) * Eigen::Isometry3d::Identity();
  flange_to_tcp = Eigen::AngleAxisd(-M_PI / 4, Eigen::Vector3d::UnitZ()) * hand_to_tcp;

  arm = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node, "panda_arm");
  arm->setPoseReferenceFrame(base_frame);
  arm->setEndEffectorLink(flange);
  arm->setPlanningTime(node->declare_parameter("planning_time", 5.0));
  arm->setNumPlanningAttempts(4);   // OMPL: best of 4 (shortest after simplification)

  gripper = rclcpp_action::create_client<control_msgs::action::GripperCommand>(node, "/panda_hand_controller/gripper_cmd");
  estimate_pose = node->create_client<ppp_interfaces::srv::EstimatePose>("/pose_node/estimate_pose");
  plan_grasps = node->create_client<ppp_interfaces::srv::PlanGrasps>("/grasp_node/plan_grasps");
  get_placements = node->create_client<ppp_interfaces::srv::GetPlacements>("/place_node/get_placements");
  compute_ik = node->create_client<moveit_msgs::srv::GetPositionIK>("/compute_ik");
  check_state = node->create_client<moveit_msgs::srv::GetStateValidity>("/check_state_validity");
  target_pub = node->create_publisher<std_msgs::msg::String>("~/target", rclcpp::QoS(1).transient_local());
  place_goal_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("~/place_goal", 5);
  js_sub_ = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10, [this](sensor_msgs::msg::JointState::ConstSharedPtr m) {
      std::lock_guard<std::mutex> l(js_mtx_);
      Joints q(7, 0.0);
      for (std::size_t i = 0; i < m->name.size(); ++i) {
        if (m->name[i] == "panda_finger_joint1") {
          finger_ = m->position[i];
        } else if (m->name[i] == "panda_finger_joint2") {
          finger2_ = m->position[i];
        } else if (m->name[i].rfind("panda_joint", 0) == 0) {
          q[std::stoi(m->name[i].substr(11)) - 1] = m->position[i];
        }
      }
      arm_pos_ = q;
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

double Context::finger_position()
{
  std::lock_guard<std::mutex> l(js_mtx_);
  return finger_;
}

double Context::opening()
{
  std::lock_guard<std::mutex> l(js_mtx_);
  return finger_ + finger2_;
}

bool Context::wait_settled(double max_wait_s)
{
  // at rest = no joint moved more than 0.5 mrad over 0.1 s of sim time (by position: Isaac's joint velocities are
  // unreliable when something blocks the robot, I-019)
  auto clock = node->get_clock();
  const auto start = clock->now();
  Joints ref;
  rclcpp::Time ref_t = start;
  while ((clock->now() - start).seconds() < max_wait_s) {
    Joints q;
    {
      std::lock_guard<std::mutex> l(js_mtx_);
      q = arm_pos_;
    }
    const auto now = clock->now();
    double moved = ref.empty() ? 1.0 : 0.0;
    for (std::size_t i = 0; i < q.size() && !ref.empty(); ++i) {
      moved = std::max(moved, std::abs(q[i] - ref[i]));
    }
    if (moved > 5e-4) {
      ref = q;
      ref_t = now;
    } else if (!q.empty() && (now - ref_t).seconds() > 0.1) {
      return true;
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

std::optional<Held> Context::held_object(const std::string & target, const Eigen::Isometry3d & T_obj_tcp)
{
  std::string mesh_file;   // assets/ycb/<NNN_target>/textured.obj
  for (const auto & e : std::filesystem::directory_iterator(ycb_dir)) {
    const auto n = e.path().filename().string();
    if (n.size() > target.size() && n.compare(n.size() - target.size(), target.size(), target) == 0) {
      mesh_file = (e.path() / "textured.obj").string();
    }
  }
  std::unique_ptr<shapes::Mesh> mesh(mesh_file.empty() ? nullptr : shapes::createMeshFromResource("file://" + mesh_file));
  if (!mesh) {
    RCLCPP_ERROR(node->get_logger(), "no mesh for %s in %s", target.c_str(), ycb_dir.c_str());
    return std::nullopt;
  }
  mesh->padd(held_padding);
  shapes::ShapeMsg shape;
  shapes::constructMsgFromShape(mesh.get(), shape);
  Held aco;
  aco.link_name = "panda_hand";
  aco.touch_links = {"panda_hand", "panda_leftfinger", "panda_rightfinger"};
  aco.object.id = "held_" + target;
  aco.object.header.frame_id = "panda_hand";
  aco.object.meshes.push_back(boost::get<shape_msgs::msg::Mesh>(shape));
  aco.object.mesh_poses.push_back(tf2::toMsg(Eigen::Isometry3d(hand_to_tcp * T_obj_tcp.inverse())));
  aco.object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return aco;
}

std::optional<Joints> Context::ik(const Eigen::Isometry3d & tcp, const Joints & seed, const Held * held,
                                  double max_jump)
{
  auto req = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
  auto & r = req->ik_request;
  r.group_name = arm->getName();
  r.ik_link_name = flange;
  r.avoid_collisions = true;
  r.timeout = rclcpp::Duration::from_seconds(0.1);
  r.pose_stamped.header.frame_id = base_frame;
  r.pose_stamped.pose = tf2::toMsg(Eigen::Isometry3d(tcp * flange_to_tcp.inverse()));
  auto state = arm->getCurrentState(1.0);
  state->setJointGroupPositions(arm->getName(), seed);   // the solver starts here
  moveit::core::robotStateToRobotStateMsg(*state, r.robot_state);
  if (held) {
    r.robot_state.attached_collision_objects.push_back(*held);
  }
  auto res = call<moveit_msgs::srv::GetPositionIK>(compute_ik, req, 5.0);
  if (!res || res->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
    return std::nullopt;
  }
  moveit::core::RobotState sol(*state);
  moveit::core::robotStateMsgToRobotState(res->solution, sol);
  Joints q;
  sol.copyJointGroupPositions(arm->getName(), q);
  for (std::size_t i = 0; i < q.size(); ++i) {
    if (std::abs(q[i] - seed[i]) > max_jump) {
      return std::nullopt;
    }
  }
  return q;
}

bool Context::path_clear(const moveit_msgs::msg::RobotTrajectory & traj, double step)
{
  const auto & jt = traj.joint_trajectory;
  auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
  req->group_name = arm->getName();
  req->robot_state.is_diff = true;   // only the joints change: the scene's attached object stays on the hand
  req->robot_state.joint_state.name = jt.joint_names;
  for (std::size_t k = 1; k < jt.points.size(); ++k) {
    const auto & a = jt.points[k - 1].positions, & b = jt.points[k].positions;
    double d = 0.0;
    for (std::size_t j = 0; j < a.size(); ++j) {
      d = std::max(d, std::abs(b[j] - a[j]));
    }
    const int n = std::max(1, static_cast<int>(std::ceil(d / step)));
    for (int i = 1; i <= n; ++i) {
      auto & q = req->robot_state.joint_state.position;
      q.resize(a.size());
      for (std::size_t j = 0; j < a.size(); ++j) {
        q[j] = a[j] + (b[j] - a[j]) * i / n;
      }
      auto res = call<moveit_msgs::srv::GetStateValidity>(check_state, req, 5.0);
      if (!res || !res->valid) {
        std::string what;
        for (const auto & c : res ? res->contacts : decltype(res->contacts){}) {
          what += " " + c.contact_body_1 + "-" + c.contact_body_2;
        }
        RCLCPP_INFO(node->get_logger(), "path collides at point %zu/%zu:%s", k, jt.points.size() - 1, what.c_str());
        return false;
      }
    }
  }
  return true;
}

}  // namespace ppp_task
