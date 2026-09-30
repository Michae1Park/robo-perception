// What every behavior-tree node shares: the ROS node, MoveIt, the gripper, and the perception service clients.
#pragma once

#include <Eigen/Geometry>
#include <control_msgs/action/gripper_command.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

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

  std::map<std::string, std::vector<double>> named_joints;   // "home", "look" (from parameters)
  std::string base_frame = "panda_link0";
  std::string flange = "panda_link8";                        // what MoveIt plans for (tip of panda_arm)
  Eigen::Isometry3d flange_to_tcp;                           // panda_link8 -> fingertip TCP
  Eigen::Isometry3d hand_to_tcp;                             // panda_hand -> fingertip TCP
  double velocity_scaling, linear_velocity_scaling, carry_velocity_scaling, planning_time;
  bool carrying = false;                                     // an object is attached to the hand
  double free_scaling() const { return carrying ? carry_velocity_scaling : velocity_scaling; }
  std::string ycb_dir;                                       // assets/ycb (meshes for attached objects)

  // TCP <-> flange (all task poses are TCP poses in the base frame)
  geometry_msgs::msg::PoseStamped tcp_to_flange(const geometry_msgs::msg::PoseStamped & tcp) const;
  Eigen::Isometry3d current_tcp() const;
  double finger_position();                                  // one finger's opening (m), from /joint_states
  // Wait until the arm has come to rest after a motion (the sim lags its command by a few mrad); false on timeout.
  bool wait_settled(double max_wait_s = 3.0);
  // Plan from the current state, clamped into the joint limits: PhysX lets a joint pressed against its limit
  // overshoot by a hair, and MoveIt refuses to plan from an out-of-bounds start state.
  void start_from_current_state();

  // The target's mesh riding on panda_hand, for a grasp with TCP pose T_obj_tcp in the object frame.
  std::optional<moveit_msgs::msg::AttachedCollisionObject> held_object(const std::string & target,
                                                                        const Eigen::Isometry3d & T_obj_tcp);
  // Is there a collision-free IK solution for this TCP pose (optionally holding `held`)?
  bool reachable(const Eigen::Isometry3d & tcp, const moveit_msgs::msg::AttachedCollisionObject * held);
  // Why not? "no IK" or the colliding body pairs at an IK solution that ignores collisions.
  // Collisions of the current state (as MoveIt sees it), "" if none.
  std::string current_contacts();
  std::string why_unreachable(const Eigen::Isometry3d & tcp, const moveit_msgs::msg::AttachedCollisionObject * held);

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
  double finger_ = -1.0;
  std::vector<double> arm_pos_;
  rclcpp::Time arm_stamp_{0, 0, RCL_ROS_TIME};
};

}  // namespace ppp_task
