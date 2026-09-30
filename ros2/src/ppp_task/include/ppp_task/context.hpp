// What every behavior-tree node shares: the ROS node, MoveIt, the gripper, and the perception service clients.
#pragma once

#include <Eigen/Geometry>
#include <control_msgs/action/gripper_command.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ppp_interfaces/srv/estimate_pose.hpp"
#include "ppp_interfaces/srv/get_placements.hpp"
#include "ppp_interfaces/srv/plan_grasps.hpp"

namespace ppp_task
{

using Joints = std::vector<double>;   // panda_joint1..7
using Held = moveit_msgs::msg::AttachedCollisionObject;

struct Context
{
  explicit Context(const rclcpp::Node::SharedPtr & node);

  rclcpp::Node::SharedPtr node;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm;
  moveit::planning_interface::PlanningSceneInterface scene;
  rclcpp_action::Client<control_msgs::action::GripperCommand>::SharedPtr gripper;
  rclcpp::Client<ppp_interfaces::srv::EstimatePose>::SharedPtr estimate_pose;
  rclcpp::Client<ppp_interfaces::srv::PlanGrasps>::SharedPtr plan_grasps;
  rclcpp::Client<ppp_interfaces::srv::GetPlacements>::SharedPtr get_placements;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr compute_ik;
  rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedPtr check_state;
  // for viewers only (the sim's overlay): the target being worked on ("" when done) and where it will be set down
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr target_pub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr place_goal_pub;

  std::map<std::string, Joints> named_joints;   // "home", "look" (parameters)
  std::string base_frame = "panda_link0";
  std::string flange = "panda_link8";           // what MoveIt plans for (tip of panda_arm)
  Eigen::Isometry3d flange_to_tcp;              // panda_link8 -> TCP between the finger pads
  Eigen::Isometry3d hand_to_tcp;                // panda_hand -> TCP
  double velocity_scaling, linear_velocity_scaling, carry_velocity_scaling, gripper_effort, held_padding;
  bool carrying = false;                        // an object is attached to the hand
  double free_scaling() const { return carrying ? carry_velocity_scaling : velocity_scaling; }
  std::string ycb_dir;                          // assets/ycb (meshes for the target and the held object)

  // TCP <-> flange (all task poses are TCP poses in the base frame)
  geometry_msgs::msg::PoseStamped tcp_to_flange(const geometry_msgs::msg::PoseStamped & tcp) const;
  double finger_position();                     // one finger's opening (m), from /joint_states
  double opening();                             // between the pads (m): both fingers (the mimic one lags a bit)
  // Wait until the arm has come to rest after a motion (the sim lags its command by a few mrad); false on timeout.
  bool wait_settled(double max_wait_s = 3.0);
  // Plan from the current state, clamped into the joint limits: PhysX lets a joint pressed against its limit
  // overshoot by a hair, and MoveIt refuses to plan from an out-of-bounds start state.
  void start_from_current_state();

  // The target's mesh (padded by held_padding) riding on panda_hand, for a grasp T_obj_tcp in the object frame.
  std::optional<Held> held_object(const std::string & target, const Eigen::Isometry3d & T_obj_tcp);
  // Collision-free IK for a TCP pose (pick_ik: the solution nearest to `seed`), optionally holding `held`.
  // Rejected if any joint ends up more than `max_jump` rad from the seed (a straight-line move must stay continuous).
  std::optional<Joints> ik(const Eigen::Isometry3d & tcp, const Joints & seed, const Held * held,
                           double max_jump = M_PI);
  // Every state along the trajectory, at most `step` rad apart per joint, is collision-free in the current
  // planning scene (held object included).
  bool path_clear(const moveit_msgs::msg::RobotTrajectory & traj, double step = 0.02);

  // blocking service call (the node spins on another thread)
  template <typename SrvT>
  typename SrvT::Response::SharedPtr call(typename rclcpp::Client<SrvT>::SharedPtr & client,
                                          typename SrvT::Request::SharedPtr req, double timeout_s)
  {
    if (!client->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(node->get_logger(), "service %s not available", client->get_service_name());
      return nullptr;
    }
    auto fut = client->async_send_request(req);
    if (fut.wait_for(std::chrono::duration<double>(timeout_s)) != std::future_status::ready) {
      RCLCPP_ERROR(node->get_logger(), "service %s timed out", client->get_service_name());
      client->remove_pending_request(fut);
      return nullptr;
    }
    return fut.get();
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  std::mutex js_mtx_;
  double finger_ = -1.0, finger2_ = -1.0;
  Joints arm_pos_;
};

}  // namespace ppp_task
