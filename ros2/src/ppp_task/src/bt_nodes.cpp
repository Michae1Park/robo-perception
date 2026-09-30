#include "ppp_task/bt_nodes.hpp"

#include <behaviortree_cpp/decorators/loop_node.h>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <tuple>

#include "ppp_interfaces/msg/grasp.hpp"
#include "ppp_interfaces/msg/placement.hpp"

namespace ppp_task
{

using Pose = geometry_msgs::msg::PoseStamped;
using Grasp = ppp_interfaces::msg::Grasp;
using Placement = ppp_interfaces::msg::Placement;
using GraspQueue = BT::SharedQueue<Grasp>;
using PlacementQueue = BT::SharedQueue<Placement>;
using BT::InputPort;
using BT::NodeStatus;
using BT::OutputPort;
using MoveItCode = moveit::core::MoveItErrorCode;

namespace
{
Eigen::Isometry3d to_eigen(const geometry_msgs::msg::Pose & p)
{
  Eigen::Isometry3d T;
  tf2::fromMsg(p, T);
  return T;
}

Pose to_msg(const Eigen::Isometry3d & T, const std::string & frame)
{
  Pose p;
  p.header.frame_id = frame;
  p.pose = tf2::toMsg(T);
  return p;
}

// Summed joint motion (rad) along a trajectory: how much the arm moves to get there. A clean path is short.
double joint_travel(const moveit_msgs::msg::RobotTrajectory & traj)
{
  const auto & pts = traj.joint_trajectory.points;
  double sum = 0.0;
  for (std::size_t k = 1; k < pts.size(); ++k) {
    for (std::size_t j = 0; j < pts[k].positions.size(); ++j) {
      sum += std::abs(pts[k].positions[j] - pts[k - 1].positions[j]);
    }
  }
  return sum;
}

// `q` with joint 1 turned towards the TCP pose: an IK seed that faces the goal, so the solver stays in the same
// (natural) configuration family instead of wandering off to a flipped one.
Joints aim(Joints q, const Eigen::Isometry3d & tcp)
{
  q[0] = std::clamp(std::atan2(tcp.translation().y(), tcp.translation().x()), -2.8, 2.8);
  return q;
}

// Base class: access to the shared context, plus logging with the node's name in the tree.
class Leaf : public BT::SyncActionNode
{
public:
  Leaf(const std::string & name, const BT::NodeConfig & cfg, std::shared_ptr<Context> ctx)
  : BT::SyncActionNode(name, cfg), ctx_(std::move(ctx)) {}

protected:
  template <typename T>
  T in(const std::string & key)
  {
    auto v = getInput<T>(key);
    if (!v) {
      throw BT::RuntimeError(name(), ": missing input [", key, "]: ", v.error());
    }
    return v.value();
  }
  rclcpp::Logger log() const { return ctx_->node->get_logger().get_child(name()); }
  NodeStatus fail(const std::string & why)
  {
    RCLCPP_WARN(log(), "FAILURE: %s", why.c_str());
    return NodeStatus::FAILURE;
  }
  std::shared_ptr<Context> ctx_;
};

// ---------------------------------------------------------------- motion
// Free-space motion to a joint configuration: a named one ("home", "look") or one from the plan (SelectPlans IK).
// Joint targets, not pose targets: a pose target lets the planner pick any of the arm's many IK solutions. Tried in
// order, each result checked densely for collisions before it runs (the shelf boards are 2 cm thick):
//   1. Pilz PTP: the straight line in joint space - the shortest motion, when nothing is in the way
//   2. two straight lines through "home" (up and back): clears the shelf's board edges with an object in the hand
//   3. STOMP: bends the straight line around obstacles (smooth, but only a local fix)
//   4. OMPL RRTConnect: sampling-based, for whatever is left
class MoveToJoints : public Leaf
{
public:
  using Leaf::Leaf;
  using Plan = moveit::planning_interface::MoveGroupInterface::Plan;
  static BT::PortsList providedPorts()
  {
    return {InputPort<std::string>("pose", "", "named joint pose: home | look"),
            InputPort<Joints>("joints", "joint target (when no name is given)")};
  }
  NodeStatus tick() override
  {
    const auto name = in<std::string>("pose");
    Joints q;
    if (!name.empty()) {
      const auto it = ctx_->named_joints.find(name);
      if (it == ctx_->named_joints.end()) {
        return fail("unknown joint pose " + name);
      }
      q = it->second;
    } else {
      q = in<Joints>("joints");
    }
    ctx_->wait_settled();
    ctx_->arm->setMaxVelocityScalingFactor(ctx_->free_scaling());
    ctx_->arm->setMaxAccelerationScalingFactor(ctx_->free_scaling());
    const auto & home = ctx_->named_joints.at("home");

    std::string how;
    double travel = 0.0;
    std::optional<std::pair<bool, double>> p;   // (executed ok, joint travel) of the last leg
    if ((p = plan("PTP", q))) {
      how = "PTP";
      travel = p->second;
    } else if (q != home && via_home_clear(q) && (p = plan("PTP", home))) {
      how = "PTP via home";
      travel = p->second;
      if (!p->first) {
        return fail("execution failed");
      }
      ctx_->wait_settled();
      p = plan("PTP", q);   // from where the arm really stopped (a few mrad off home)
      if (!p) {
        return fail("no straight line on from home");
      }
      travel += p->second;
    } else if ((p = plan("stomp", q, 2))) {
      how = "STOMP";
      travel = p->second;
    } else if ((p = plan("RRTConnectkConfigDefault", q, 3))) {
      how = "RRTConnect";
      travel = p->second;
    } else {
      return fail("no plan to " + (name.empty() ? std::string("joint target") : name));
    }
    if (!p->first) {
      return fail("execution failed");
    }
    RCLCPP_INFO(log(), "at %s (%s, joint travel %.2f rad)", name.empty() ? "target" : name.c_str(), how.c_str(), travel);
    return NodeStatus::SUCCESS;
  }

private:
  // Plan from the current state to q with one planner (retried `attempts` times) and, if the path is clear, execute
  // it. -> (executed ok, joint travel), or nothing if no clear path was found.
  std::optional<std::pair<bool, double>> plan(const std::string & planner, const Joints & q, int attempts = 1)
  {
    auto & arm = *ctx_->arm;
    arm.setPlanningPipelineId(planner == "PTP" ? "pilz_industrial_motion_planner" : planner == "stomp" ? "stomp" : "ompl");
    arm.setPlannerId(planner);
    arm.setJointValueTarget(q);
    moveit::planning_interface::MoveGroupInterface::Plan p;
    for (int k = 0; k < attempts; ++k) {   // STOMP and RRTConnect are randomized: another try can succeed
      ctx_->start_from_current_state();
      if (arm.plan(p) == MoveItCode::SUCCESS && ctx_->path_clear(p.trajectory)) {
        return std::make_pair(arm.execute(p) == MoveItCode::SUCCESS, joint_travel(p.trajectory));
      }
    }
    return std::nullopt;
  }

  // Is the straight line home -> q clear? (Checked before leaving, so the arm doesn't go home for nothing.)
  bool via_home_clear(const Joints & q)
  {
    auto & arm = *ctx_->arm;
    auto at_home = *arm.getCurrentState(1.0);
    at_home.setJointGroupPositions(arm.getName(), ctx_->named_joints.at("home"));
    arm.setStartState(at_home);
    arm.setPlanningPipelineId("pilz_industrial_motion_planner");
    arm.setPlannerId("PTP");
    arm.setJointValueTarget(q);
    moveit::planning_interface::MoveGroupInterface::Plan p;
    return arm.plan(p) == MoveItCode::SUCCESS && ctx_->path_clear(p.trajectory);
  }
};

// Straight-line TCP motion (approach, lift, retreat), collision-checked every 5 mm, slow.
class MoveLinear : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<Pose>("pose", "TCP goal, base frame"),
            InputPort<double>("min_fraction", 0.98, "fraction of the line that must be feasible")};
  }
  NodeStatus tick() override
  {
    const auto goal = ctx_->tcp_to_flange(in<Pose>("pose"));
    ctx_->wait_settled();
    ctx_->start_from_current_state();
    moveit_msgs::msg::RobotTrajectory traj;
    const double fraction = ctx_->arm->computeCartesianPath({goal.pose}, 0.005, traj, true);
    if (fraction < in<double>("min_fraction")) {
      return fail("only " + std::to_string(int(fraction * 100)) + "% of the straight line is feasible");
    }
    robot_trajectory::RobotTrajectory rt(ctx_->arm->getRobotModel(), ctx_->arm->getName());   // re-time: slower
    rt.setRobotTrajectoryMsg(*ctx_->arm->getCurrentState(), traj);
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    totg.computeTimeStamps(rt, ctx_->linear_velocity_scaling, ctx_->linear_velocity_scaling);
    rt.getRobotTrajectoryMsg(traj);
    if (ctx_->arm->execute(traj) != MoveItCode::SUCCESS) {
      return fail("execution failed");
    }
    return NodeStatus::SUCCESS;
  }
};

// ---------------------------------------------------------------- gripper
class Gripper : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<double>("position", "one finger's opening (m): 0.04 open, 0 closed")};
  }
  NodeStatus tick() override
  {
    using GC = control_msgs::action::GripperCommand;
    if (!ctx_->gripper->wait_for_action_server(std::chrono::seconds(5))) {
      return fail("gripper action server not available");
    }
    GC::Goal goal;
    goal.command.position = in<double>("position");
    goal.command.max_effort = ctx_->gripper_effort;
    auto gh = ctx_->gripper->async_send_goal(goal);
    if (gh.wait_for(std::chrono::seconds(5)) != std::future_status::ready || !gh.get()) {
      return fail("gripper goal rejected");
    }
    // Done when the controller reports a result, or when the finger has stopped for 0.5 s (closed on an object:
    // Isaac reports finger velocity while blocked, so the controller never sees the stall, I-019). The goal stays
    // active, so the fingers keep squeezing until the next command.
    auto res = ctx_->gripper->async_get_result(gh.get());
    auto clock = ctx_->node->get_clock();
    const auto start = clock->now();
    auto last_move = start;
    double last_pos = ctx_->finger_position();
    while (res.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready) {
      const auto now = clock->now();
      const double pos = ctx_->finger_position();
      if (std::abs(pos - last_pos) > 2e-4) {
        last_pos = pos;
        last_move = now;
      } else if ((now - last_move).seconds() > 0.5) {
        break;
      }
      if ((now - start).seconds() > 15.0) {
        return fail("gripper timed out");
      }
    }
    RCLCPP_INFO(log(), "finger at %.4f m (commanded %.4f)", ctx_->finger_position(), goal.command.position);
    return NodeStatus::SUCCESS;
  }
};

// Is something in the hand? The fingers stopped between fully closed (missed) and fully open (nothing grasped).
class CheckGrasp : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts() { return {}; }
  NodeStatus tick() override
  {
    const double w = ctx_->opening();
    if (w < 0.006) {
      return fail("fingers closed on nothing");
    }
    if (w > 0.074) {
      return fail("hand open, nothing held");
    }
    RCLCPP_INFO(log(), "holding something %.1f mm wide", 1000 * w);
    return NodeStatus::SUCCESS;
  }
};

// ---------------------------------------------------------------- perception
class EstimatePose : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts() { return {InputPort<std::string>("target"), OutputPort<Pose>("object_pose")}; }
  NodeStatus tick() override
  {
    auto req = std::make_shared<ppp_interfaces::srv::EstimatePose::Request>();
    req->target = in<std::string>("target");
    auto res = ctx_->call<ppp_interfaces::srv::EstimatePose>(ctx_->estimate_pose, req, 60.0);
    if (!res || !res->success) {
      return fail(res ? res->message : "no response");
    }
    RCLCPP_INFO(log(), "%s", res->message.c_str());
    setOutput("object_pose", res->pose);
    return NodeStatus::SUCCESS;
  }
};

class PlanGrasps : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<std::string>("target"), InputPort<Pose>("object_pose"), OutputPort<GraspQueue>("grasps"),
            OutputPort<sensor_msgs::msg::PointCloud2>("obstacles")};
  }
  NodeStatus tick() override
  {
    auto req = std::make_shared<ppp_interfaces::srv::PlanGrasps::Request>();
    req->target = in<std::string>("target");
    req->object_pose = in<Pose>("object_pose");
    auto res = ctx_->call<ppp_interfaces::srv::PlanGrasps>(ctx_->plan_grasps, req, 30.0);
    if (!res || !res->success) {
      return fail(res ? res->message : "no response");
    }
    RCLCPP_INFO(log(), "%s", res->message.c_str());
    setOutput("grasps", std::make_shared<std::deque<Grasp>>(res->grasps.begin(), res->grasps.end()));
    setOutput("obstacles", res->obstacles);
    return NodeStatus::SUCCESS;
  }
};

class GetPlacements : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<std::string>("target"), OutputPort<PlacementQueue>("placements"),
            OutputPort<sensor_msgs::msg::PointCloud2>("obstacles")};
  }
  NodeStatus tick() override
  {
    auto req = std::make_shared<ppp_interfaces::srv::GetPlacements::Request>();
    req->target = in<std::string>("target");
    // a wrist frame taken after the arm stopped (the last motion's execute() returned just now)
    req->not_before = (ctx_->node->now() + rclcpp::Duration::from_seconds(0.2)).operator builtin_interfaces::msg::Time();
    auto res = ctx_->call<ppp_interfaces::srv::GetPlacements>(ctx_->get_placements, req, 30.0);
    if (!res || !res->success) {
      return fail(res ? res->message : "no response");
    }
    RCLCPP_INFO(log(), "%s", res->message.c_str());
    setOutput("placements", std::make_shared<std::deque<Placement>>(res->placements.begin(), res->placements.end()));
    setOutput("obstacles", res->obstacles);
    return NodeStatus::SUCCESS;
  }
};

// One way to do the whole job: a grasp, where the object goes down, the TCP poses of the straight-line moves and
// the joint configurations of the free-space moves (all from one chain of IK solutions, so they fit together).
struct PickPlacePlan
{
  Grasp grasp;
  Placement placement;
  Pose grasp_pose, lifted, place, depart, pregrasp;   // TCP poses (base frame); depart: backed off after release
  Joints pregrasp_q, lifted_q, preplace_q;
  std::string approach;                               // how the object goes in: "inline", "above", "front"
};
using PlanQueue = BT::SharedQueue<PickPlacePlan>;

// Grasps x placements -> plans whose every pose has a collision-free IK solution (the held mesh on the hand from
// the lift on). IK returns the solution nearest its seed (pick_ik): free-space goals are seeded with where the arm
// comes from, turned towards the goal (home for the pregrasp, the lift for the preplace); straight-line goals with
// the pose before, and must stay within kLineJump of it. So the arm keeps one natural configuration throughout.
// Placements in stage 5's order (upright first, then most room); for each, grasps least tilted first. The object
// is set down in the placement's rest pose, turned about vertical so that a side grasp points from the robot into
// the shelf (otherwise quarter turns are tried).
class SelectPlans : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<std::string>("target"), InputPort<GraspQueue>("grasps"),
            InputPort<PlacementQueue>("placements"), OutputPort<PlanQueue>("plans")};
  }
  NodeStatus tick() override
  {
    constexpr double kPregrasp = 0.10;   // m back from the grasp along the approach
    constexpr double kLift = 0.18;       // m straight up after closing: room to turn a tall object over the table
    constexpr double kPreplace = 0.12;   // m from the place pose
    constexpr double kDropGap = 0.01;    // m above the support when released
    constexpr double kDepart = 0.06;     // m back along the gripper axis after releasing
    constexpr std::size_t kMaxPlans = 4;
    constexpr double kLineJump = 0.8;    // rad, any joint: along a straight-line move the arm barely changes
    const auto target = in<std::string>("target");
    const auto grasps = in<GraspQueue>("grasps");
    const auto placements = in<PlacementQueue>("placements");
    const auto & home = ctx_->named_joints.at("home");
    const auto t0 = std::chrono::steady_clock::now();

    struct Pick
    {
      bool checked = false;
      std::optional<Held> held;
      std::optional<Joints> pre, at, up;
    };
    std::vector<Pick> picks(grasps->size());   // the pick half depends only on the grasp: once per grasp
    int no_pick = 0, no_place = 0;
    auto plans = std::make_shared<std::deque<PickPlacePlan>>();
    for (const auto & pl : *placements) {
      const Eigen::Vector3d p(pl.point.point.x, pl.point.point.y, pl.point.point.z);
      const Eigen::Vector2d into = p.head<2>().normalized();   // from the robot towards the place point
      for (std::size_t gi = 0; gi < grasps->size() && plans->size() < kMaxPlans; ++gi) {
        const auto & g = (*grasps)[gi];
        const Eigen::Isometry3d T_obj_tcp = to_eigen(g.tcp_in_object), tcp0 = to_eigen(g.tcp.pose);
        Eigen::Isometry3d pre0 = tcp0, up0 = tcp0;
        pre0.translation() -= kPregrasp * tcp0.linear().col(2);
        up0.translation().z() += kLift;
        auto & pk = picks[gi];
        if (!pk.checked) {
          pk.checked = true;
          pk.held = ctx_->held_object(target, T_obj_tcp);
          pk.pre = ctx_->ik(pre0, aim(home, pre0), nullptr);
          pk.at = pk.pre ? ctx_->ik(tcp0, *pk.pre, nullptr, kLineJump) : std::nullopt;
          pk.up = pk.at && pk.held ? ctx_->ik(up0, *pk.at, &*pk.held, kLineJump) : std::nullopt;
          no_pick += !pk.up;
        }
        if (!pk.up) {
          continue;
        }
        const Eigen::Matrix3d R_rest =
          Eigen::Quaterniond(pl.orientation.w, pl.orientation.x, pl.orientation.y, pl.orientation.z).toRotationMatrix();
        const Eigen::Vector3d z_rest = R_rest * T_obj_tcp.linear().col(2);   // approach, rest pose at yaw 0
        std::vector<double> yaws = {0.0, M_PI / 2, M_PI, -M_PI / 2};
        if (z_rest.head<2>().norm() > 0.5) {   // side grasp: point it into the shelf
          yaws = {std::atan2(into.y(), into.x()) - std::atan2(z_rest.y(), z_rest.x())};
        }
        std::optional<PickPlacePlan> plan;
        for (std::size_t k = 0; k < yaws.size() && !plan; ++k) {
          Eigen::Isometry3d T_obj = Eigen::Isometry3d::Identity();
          T_obj.linear() = Eigen::AngleAxisd(yaws[k], Eigen::Vector3d::UnitZ()) * R_rest;
          T_obj.translation() = p + Eigen::Vector3d(0, 0, pl.object_bottom + kDropGap);
          const Eigen::Isometry3d place = T_obj * T_obj_tcp;
          const Eigen::Vector3d zp = place.linear().col(2);
          Eigen::Isometry3d inl = place, above = place, front = place, dep = place;
          inl.translation() -= kPreplace * zp;
          above.translation().z() += kPreplace;
          front.translation().head<2>() -= kPreplace * into;
          dep.translation() -= kDepart * zp;
          const std::vector<std::pair<std::string, Eigen::Isometry3d>> approaches = {
            {"inline", inl}, {"above", above}, {"front", front}};
          for (const auto & [name, pre] : approaches) {
            const auto pre_q = ctx_->ik(pre, aim(*pk.up, pre), &*pk.held);
            if (pre_q && ctx_->ik(place, *pre_q, &*pk.held, kLineJump)) {
              const auto & F = ctx_->base_frame;
              plan = PickPlacePlan{g, pl, to_msg(tcp0, F), to_msg(up0, F), to_msg(place, F), to_msg(dep, F),
                                   to_msg(pre0, F), *pk.pre, *pk.up, *pre_q, name};
              break;
            }
          }
        }
        if (!plan) {
          ++no_place;
          continue;
        }
        plans->push_back(*plan);
        RCLCPP_INFO(log(), "plan %zu: grasp %s (tilt %.0f deg) -> %s at (%.3f, %.3f) on z %.2f, in %s",
                    plans->size(), g.face.c_str(), g.tilt * 180 / M_PI, pl.rest_pose.c_str(), p.x(), p.y(), p.z(),
                    plan->approach.c_str());
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    RCLCPP_INFO(log(), "%zu plans from %zu grasps x %zu placements (%d grasps unreachable, %d grasp-placement pairs "
                "unreachable at the shelf, %.0f ms)", plans->size(), grasps->size(), placements->size(), no_pick,
                no_place, ms);
    if (plans->empty()) {
      return fail("no grasp + placement with reachable poses");
    }
    setOutput("plans", plans);
    return NodeStatus::SUCCESS;
  }
};

// Next plan from the queue.
class PopPlan : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {BT::BidirectionalPort<PlanQueue>("plans"), OutputPort<Grasp>("grasp"), OutputPort<Pose>("grasp_pose"),
            OutputPort<Pose>("lifted"), OutputPort<Pose>("place"), OutputPort<Pose>("depart"),
            OutputPort<Pose>("pregrasp"), OutputPort<Joints>("pregrasp_q"), OutputPort<Joints>("lifted_q"),
            OutputPort<Joints>("preplace_q")};
  }
  NodeStatus tick() override
  {
    auto q = in<PlanQueue>("plans");
    if (!q || q->empty()) {
      return fail("no plan left");
    }
    const PickPlacePlan pp = q->front();
    q->pop_front();
    setOutput("grasp", pp.grasp);
    setOutput("grasp_pose", pp.grasp_pose);
    setOutput("lifted", pp.lifted);
    setOutput("place", pp.place);
    setOutput("depart", pp.depart);
    setOutput("pregrasp", pp.pregrasp);
    setOutput("pregrasp_q", pp.pregrasp_q);
    setOutput("lifted_q", pp.lifted_q);
    setOutput("preplace_q", pp.preplace_q);
    // for viewers: the object pose once set down = hand at place x (grasp seen from the object)^-1
    ctx_->place_goal_pub->publish(to_msg(to_eigen(pp.place.pose) * to_eigen(pp.grasp.tcp_in_object).inverse(),
                                         pp.place.header.frame_id));
    RCLCPP_INFO(log(), "plan: grasp %s, place at (%.3f, %.3f) on z %.2f, in %s (%zu left)", pp.grasp.face.c_str(),
                pp.placement.point.point.x, pp.placement.point.point.y, pp.placement.point.point.z,
                pp.approach.c_str(), q->size());
    return NodeStatus::SUCCESS;
  }
};

// ---------------------------------------------------------------- planning scene
// What else perception saw goes into MoveIt as voxel boxes, so motions keep clear of it: the other things on the
// table, and the items on the shelf. Replaced on every call. (The target itself is added separately, AddTarget.)
class UpdateScene : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<sensor_msgs::msg::PointCloud2>("obstacles"), InputPort<std::string>("id", "object id")};
  }
  NodeStatus tick() override
  {
    constexpr double v = 0.025;   // voxel size (m)
    constexpr int kMinPoints = 4;   // per voxel: drops flying pixels
    const auto cloud = in<sensor_msgs::msg::PointCloud2>("obstacles");
    std::map<std::tuple<int, int, int>, int> cells;
    for (sensor_msgs::PointCloud2ConstIterator<float> it(cloud, "x"); it != it.end(); ++it) {
      ++cells[{int(std::floor(it[0] / v)), int(std::floor(it[1] / v)), int(std::floor(it[2] / v))}];
    }
    moveit_msgs::msg::CollisionObject clutter;
    clutter.header.frame_id = ctx_->base_frame;
    clutter.id = in<std::string>("id");
    clutter.operation = moveit_msgs::msg::CollisionObject::ADD;
    clutter.pose.orientation.w = 1.0;
    for (const auto & [c, n] : cells) {
      if (n < kMinPoints) {
        continue;
      }
      shape_msgs::msg::SolidPrimitive box;
      box.type = box.BOX;
      box.dimensions = {v, v, v};
      geometry_msgs::msg::Pose p;
      p.position.x = (std::get<0>(c) + 0.5) * v;
      p.position.y = (std::get<1>(c) + 0.5) * v;
      p.position.z = (std::get<2>(c) + 0.5) * v;
      p.orientation.w = 1.0;
      clutter.primitives.push_back(box);
      clutter.primitive_poses.push_back(p);
    }
    if (!ctx_->scene.applyCollisionObject(clutter)) {
      return fail("planning scene update failed");
    }
    RCLCPP_INFO(log(), "planning scene: %s = %zu voxels", clutter.id.c_str(), clutter.primitives.size());
    return NodeStatus::SUCCESS;
  }
};

// The target (its mesh at the estimated pose) joins the planning scene while the arm travels to it, so the path to
// the pregrasp can't knock it over. Not during SelectPlans: there the grasp and lift poses overlap it by design.
class AddTarget : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts() { return {InputPort<std::string>("target"), InputPort<Pose>("object_pose")}; }
  NodeStatus tick() override
  {
    const auto target = in<std::string>("target");
    auto held = ctx_->held_object(target, Eigen::Isometry3d::Identity());   // its mesh, reused as a world object
    if (!held) {
      return fail("no mesh for " + target);
    }
    auto obj = held->object;
    obj.id = "target_" + target;
    obj.header.frame_id = ctx_->base_frame;
    obj.mesh_poses = {in<Pose>("object_pose").pose};
    if (!ctx_->scene.applyCollisionObject(obj)) {
      return fail("planning scene update failed");
    }
    return NodeStatus::SUCCESS;
  }
};

// The target leaves the world before the fingers close around it (then it is attached to the hand instead).
class RemoveTarget : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts() { return {InputPort<std::string>("target")}; }
  NodeStatus tick() override
  {
    ctx_->scene.removeCollisionObjects({"target_" + in<std::string>("target")});
    return NodeStatus::SUCCESS;
  }
};

// The held object becomes part of the robot for planning: its mesh rides on panda_hand, so every IK, plan and
// collision check (shelf, clutter, the robot itself) includes it.
class AttachObject : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts() { return {InputPort<std::string>("target"), InputPort<Grasp>("grasp")}; }
  NodeStatus tick() override
  {
    const auto target = in<std::string>("target");
    auto aco = ctx_->held_object(target, to_eigen(in<Grasp>("grasp").tcp_in_object));
    if (!aco || !ctx_->scene.applyAttachedCollisionObject(*aco)) {
      return fail("attach failed");
    }
    ctx_->carrying = true;
    return NodeStatus::SUCCESS;
  }
};

// Released: drop the object from the planning scene altogether (the fingers still surround it).
class DetachObject : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts() { return {InputPort<std::string>("target")}; }
  NodeStatus tick() override
  {
    const std::string id = "held_" + in<std::string>("target");
    Held aco;
    aco.link_name = "panda_hand";
    aco.object.id = id;
    aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    ctx_->scene.applyAttachedCollisionObject(aco);
    ctx_->scene.removeCollisionObjects({id});
    ctx_->carrying = false;
    return NodeStatus::SUCCESS;
  }
};

template <typename T>
void reg(BT::BehaviorTreeFactory & f, const std::string & id, const std::shared_ptr<Context> & ctx)
{
  f.registerNodeType<T>(id, ctx);
}
}  // namespace

void register_nodes(BT::BehaviorTreeFactory & f, const std::shared_ptr<Context> & ctx)
{
  reg<MoveToJoints>(f, "MoveToJoints", ctx);
  reg<MoveLinear>(f, "MoveLinear", ctx);
  reg<Gripper>(f, "Gripper", ctx);
  reg<CheckGrasp>(f, "CheckGrasp", ctx);
  reg<EstimatePose>(f, "EstimatePose", ctx);
  reg<PlanGrasps>(f, "PlanGrasps", ctx);
  reg<GetPlacements>(f, "GetPlacements", ctx);
  reg<SelectPlans>(f, "SelectPlans", ctx);
  reg<PopPlan>(f, "PopPlan", ctx);
  reg<UpdateScene>(f, "UpdateScene", ctx);
  reg<AddTarget>(f, "AddTarget", ctx);
  reg<RemoveTarget>(f, "RemoveTarget", ctx);
  reg<AttachObject>(f, "AttachObject", ctx);
  reg<DetachObject>(f, "DetachObject", ctx);
}

}  // namespace ppp_task
