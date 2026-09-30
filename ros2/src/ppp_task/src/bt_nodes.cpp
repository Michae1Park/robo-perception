#include "ppp_task/bt_nodes.hpp"

#include <behaviortree_cpp/decorators/loop_node.h>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <optional>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <map>
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
class MoveToJoints : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<std::string>("pose", "named joint pose: home | look"),
            InputPort<int>("attempts", 3, "planning attempts")};
  }
  NodeStatus tick() override
  {
    const auto name = in<std::string>("pose");
    const auto it = ctx_->named_joints.find(name);
    if (it == ctx_->named_joints.end()) {
      return fail("unknown joint pose " + name);
    }
    ctx_->wait_settled();
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool ok = false;
    for (int k = 0; k < in<int>("attempts") && !ok; ++k) {
      ctx_->start_from_current_state();
      ctx_->arm->setJointValueTarget(it->second);
      ctx_->arm->setMaxVelocityScalingFactor(ctx_->free_scaling());
      ctx_->arm->setMaxAccelerationScalingFactor(ctx_->free_scaling());
      ok = ctx_->arm->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
    }
    if (!ok) {
      return fail("no plan to " + name);
    }
    if (ctx_->arm->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      return fail("execution to " + name + " failed");
    }
    RCLCPP_INFO(log(), "at %s", name.c_str());
    return NodeStatus::SUCCESS;
  }
};

// Free-space motion (OMPL) to a TCP pose. Planning is retried: MoveIt's final check can reject an OMPL path
// whose interpolation clips an obstacle (tight shelf, held object), and a fresh attempt usually finds another.
class MoveToPose : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<Pose>("pose", "TCP goal, base frame"), InputPort<int>("attempts", 3, "planning attempts")};
  }
  NodeStatus tick() override
  {
    const auto goal = ctx_->tcp_to_flange(in<Pose>("pose"));
    ctx_->wait_settled();
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool ok = false;
    for (int k = 0; k < in<int>("attempts") && !ok; ++k) {
      ctx_->start_from_current_state();
      ctx_->arm->setPoseTarget(goal, ctx_->flange);
      ctx_->arm->setMaxVelocityScalingFactor(ctx_->free_scaling());
      ctx_->arm->setMaxAccelerationScalingFactor(ctx_->free_scaling());
      ok = ctx_->arm->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
    }
    ctx_->arm->clearPoseTargets();
    if (!ok) {
      return fail("no plan to TCP pose");
    }
    if (ctx_->arm->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      return fail("execution failed");
    }
    return NodeStatus::SUCCESS;
  }
};

// Straight-line TCP motion (approach, lift, retreat), collision-checked, slow.
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
      const auto contacts = fraction < 0.1 ? ctx_->current_contacts() : "";
      return fail("only " + std::to_string(int(fraction * 100)) + "% of the straight line is feasible" +
                  (contacts.empty() ? "" : " (start state collides: " + contacts + ")"));
    }
    // re-time slower than free-space motions
    robot_trajectory::RobotTrajectory rt(ctx_->arm->getRobotModel(), ctx_->arm->getName());
    rt.setRobotTrajectoryMsg(*ctx_->arm->getCurrentState(), traj);
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    totg.computeTimeStamps(rt, ctx_->linear_velocity_scaling, ctx_->linear_velocity_scaling);
    rt.getRobotTrajectoryMsg(traj);
    if (ctx_->arm->execute(traj) != moveit::core::MoveItErrorCode::SUCCESS) {
      return fail("execution failed");
    }
    ctx_->wait_settled();
    const Eigen::Isometry3d want = to_eigen(in<Pose>("pose").pose), got = ctx_->current_tcp();
    RCLCPP_INFO(log(), "TCP at (%.3f, %.3f, %.3f), %.1f mm from the goal", got.translation().x(),
                got.translation().y(), got.translation().z(), 1000 * (got.translation() - want.translation()).norm());
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
    return {InputPort<double>("position", "one finger's opening (m): 0.04 open, 0 closed"),
            InputPort<double>("max_effort", 70.0, "N"),
            InputPort<double>("settle_time", 0.5, "s without finger motion = done")};
  }
  NodeStatus tick() override
  {
    using GC = control_msgs::action::GripperCommand;
    if (!ctx_->gripper->wait_for_action_server(std::chrono::seconds(5))) {
      return fail("gripper action server not available");
    }
    GC::Goal goal;
    goal.command.position = in<double>("position");
    goal.command.max_effort = in<double>("max_effort");
    auto gh = ctx_->gripper->async_send_goal(goal);
    if (gh.wait_for(std::chrono::seconds(5)) != std::future_status::ready || !gh.get()) {
      return fail("gripper goal rejected");
    }
    // Done when the controller reports a result, or when the finger has stopped moving (closed on an object).
    // Isaac reports a nonzero finger velocity while the finger is blocked, so the controller's own stall check
    // never fires there (I-019); the goal stays active, so the fingers keep squeezing until the next command.
    auto res = ctx_->gripper->async_get_result(gh.get());
    const double settle = in<double>("settle_time");
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
      } else if ((now - last_move).seconds() > settle && (now - start).seconds() > settle) {
        RCLCPP_INFO(log(), "finger stopped at %.4f m (commanded %.4f)", pos, goal.command.position);
        return NodeStatus::SUCCESS;
      }
      if ((now - start).seconds() > 15.0) {
        return fail("gripper timed out");
      }
    }
    const auto r = res.get();
    RCLCPP_INFO(log(), "finger at %.4f m (%s)", r.result->position,
                r.result->reached_goal ? "reached" : r.result->stalled ? "stalled on something" : "?");
    return NodeStatus::SUCCESS;
  }
};

// Is something in the hand? The fingers stopped between fully closed (missed) and fully open (nothing grasped).
class CheckGrasp : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<double>("min_opening", 0.003, "one finger (m); less = closed on nothing"),
            InputPort<double>("max_opening", 0.037, "one finger (m); more = hand open"),
            OutputPort<double>("width")};
  }
  NodeStatus tick() override
  {
    const double f = ctx_->finger_position();
    setOutput("width", 2 * f);
    if (f < in<double>("min_opening")) {
      return fail("fingers closed on nothing (" + std::to_string(f * 1000) + " mm)");
    }
    if (f > in<double>("max_opening")) {
      return fail("hand open, nothing held");
    }
    RCLCPP_INFO(log(), "holding something %.1f mm wide", 2000 * f);
    return NodeStatus::SUCCESS;
  }
};

// ---------------------------------------------------------------- perception
class EstimatePose : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<std::string>("target"), OutputPort<Pose>("object_pose")};
  }
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
    return {InputPort<std::string>("target"), InputPort<Pose>("object_pose", "as it stood on the table (pick time)"),
            InputPort<bool>("rest_poses", true, "every stable rest pose (upright first), else only the table one"),
            OutputPort<PlacementQueue>("placements"), OutputPort<sensor_msgs::msg::PointCloud2>("obstacles")};
  }
  NodeStatus tick() override
  {
    auto req = std::make_shared<ppp_interfaces::srv::GetPlacements::Request>();
    req->target = in<std::string>("target");
    req->object_orientation = in<Pose>("object_pose").pose.orientation;
    req->rest_poses = in<bool>("rest_poses");
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

// One way to do the whole job: a grasp, where to put the object down, and every TCP pose on the way.
struct PickPlacePlan
{
  Grasp grasp;
  Placement placement;
  Pose pregrasp, grasp_pose, lifted, preplace, place, depart;   // depart: backed off along the gripper axis
  std::string approach;   // how the object goes in: "inline" (along the gripper axis), "above", "front"
  double yaw = 0.0;       // rest pose turned about vertical (rad)
};
using PlanQueue = BT::SharedQueue<PickPlacePlan>;

// Grasps x placements -> plans whose every pose has a collision-free IK solution (MoveIt /compute_ik; the held
// mesh is on the hand from the lift on). Placements best first; for each, grasps least tilted first. The object
// is set down in the rest pose it had on the table, turned about vertical so that a side grasp points from the
// robot towards the place point (into the shelf). Motions between the poses are planned only when executed.
class SelectPlans : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {InputPort<std::string>("target"), InputPort<Pose>("object_pose"), InputPort<GraspQueue>("grasps"),
            InputPort<PlacementQueue>("placements"),
            InputPort<double>("drop_gap", 0.01, "m above the support when released"),
            InputPort<double>("approach", 0.10, "m: preplace distance"),
            InputPort<double>("pregrasp", 0.06, "m: pregrasp distance (short: neighbours on the table)"),
            InputPort<double>("lift", 0.12, "m"), InputPort<int>("max_plans", 4, "stop after this many"),
            InputPort<double>("depart", 0.06, "m: after release, back off along the gripper axis first"),
            OutputPort<PlanQueue>("plans")};
  }
  NodeStatus tick() override
  {
    const auto target = in<std::string>("target");
    (void)in<Pose>("object_pose");
    const auto grasps = in<GraspQueue>("grasps");
    const auto placements = in<PlacementQueue>("placements");
    const double gap = in<double>("drop_gap"), a = in<double>("approach");
    const double lift = in<double>("lift"), pregrasp = in<double>("pregrasp"), depart = in<double>("depart");
    const auto max_plans = static_cast<std::size_t>(in<int>("max_plans"));
    auto plans = std::make_shared<std::deque<PickPlacePlan>>();
    int checked = 0;
    const auto t0 = std::chrono::steady_clock::now();

    // the pick half depends only on the grasp: check it once per grasp
    std::vector<std::optional<moveit_msgs::msg::AttachedCollisionObject>> held(grasps->size());
    std::vector<int> pick_ok(grasps->size(), -1);
    reported_.assign(grasps->size(), false);
    for (const auto & pl : *placements) {
      const Eigen::Vector3d p(pl.point.point.x, pl.point.point.y, pl.point.point.z);
      const Eigen::Vector2d into = p.head<2>().normalized();   // from the robot towards the place point
      for (std::size_t gi = 0; gi < grasps->size() && plans->size() < max_plans; ++gi) {
        const auto & g = (*grasps)[gi];
        const Eigen::Isometry3d T_obj_tcp = to_eigen(g.tcp_in_object);
        const Eigen::Isometry3d tcp0 = to_eigen(g.tcp.pose);
        Eigen::Isometry3d pre0 = tcp0, up0 = tcp0;
        pre0.translation() -= pregrasp * tcp0.linear().col(2);
        up0.translation().z() += lift;
        if (pick_ok[gi] < 0) {
          held[gi] = ctx_->held_object(target, T_obj_tcp);
          pick_ok[gi] = held[gi] && ctx_->reachable(pre0, nullptr) && ctx_->reachable(tcp0, nullptr) &&
                        ctx_->reachable(up0, &*held[gi]);
          checked += 3;
        }
        if (!pick_ok[gi]) {
          if (pick_ok[gi] == 0 && !reported_[gi]) {
            const bool pre = ctx_->reachable(pre0, nullptr), at = pre && ctx_->reachable(tcp0, nullptr);
            const std::string which = !pre ? "pregrasp" : !at ? "grasp" : "lift";
            const auto & bad = !pre ? pre0 : !at ? tcp0 : up0;
            RCLCPP_INFO(log(), "  grasp %zu %s (tilt %.0f deg): %s not reachable: %s", gi, g.face.c_str(),
                        g.tilt * 180 / M_PI, which.c_str(),
                        ctx_->why_unreachable(bad, which == "lift" && held[gi] ? &*held[gi] : nullptr).c_str());
            reported_[gi] = true;
          }
          continue;
        }
        // the object goes down in this placement's rest pose, turned about vertical: a side grasp is turned to
        // point from the robot into the shelf; otherwise four quarter turns are tried
        const Eigen::Quaterniond q_rest(pl.orientation.w, pl.orientation.x, pl.orientation.y, pl.orientation.z);
        const Eigen::Matrix3d R_rest = q_rest.toRotationMatrix();
        const Eigen::Vector3d z_rest = R_rest * T_obj_tcp.linear().col(2);   // approach, rest pose at yaw 0
        std::vector<double> yaws;
        if (z_rest.head<2>().norm() > 0.5) {
          yaws = {std::atan2(into.y(), into.x()) - std::atan2(z_rest.y(), z_rest.x())};
        } else {
          yaws = {0.0, M_PI / 2, M_PI, -M_PI / 2};
        }
        bool found = false;
        for (const double yaw : yaws) {
          Eigen::Isometry3d T_obj = Eigen::Isometry3d::Identity();
          T_obj.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * R_rest;
          T_obj.translation() = p + Eigen::Vector3d(0, 0, pl.object_bottom + gap);
          const Eigen::Isometry3d place = T_obj * T_obj_tcp;
          const Eigen::Vector3d zp = place.linear().col(2);
          Eigen::Isometry3d inl = place, above = place, front = place, dep = place;
          dep.translation() -= depart * zp;
          inl.translation() -= a * zp;
          above.translation().z() += a;
          front.translation().head<2>() -= a * into;
          ++checked;
          if (!ctx_->reachable(place, &*held[gi])) {
            continue;
          }
          for (const auto & [name, pre] : std::vector<std::pair<std::string, Eigen::Isometry3d>>{
                 {"inline", inl}, {"above", above}, {"front", front}}) {
            ++checked;
            if (ctx_->reachable(pre, &*held[gi])) {
              plans->push_back(PickPlacePlan{g, pl, to_msg(pre0, ctx_->base_frame), to_msg(tcp0, ctx_->base_frame),
                                             to_msg(up0, ctx_->base_frame), to_msg(pre, ctx_->base_frame),
                                             to_msg(place, ctx_->base_frame), to_msg(dep, ctx_->base_frame), name, yaw});
              RCLCPP_INFO(log(), "plan %zu: grasp %s (tilt %.0f deg) -> %s at (%.3f, %.3f) on z %.2f, in %s, yaw %.0f deg",
                          plans->size(), g.face.c_str(), g.tilt * 180 / M_PI, pl.rest_pose.c_str(), p.x(), p.y(),
                          p.z(), name.c_str(), yaw * 180 / M_PI);
              found = true;
              break;
            }
          }
          if (found) {
            break;
          }
        }
        if (!found) {
          Eigen::Isometry3d T_obj = Eigen::Isometry3d::Identity();   // explain the first yaw tried
          T_obj.linear() = Eigen::AngleAxisd(yaws[0], Eigen::Vector3d::UnitZ()) * R_rest;
          T_obj.translation() = p + Eigen::Vector3d(0, 0, pl.object_bottom + gap);
          RCLCPP_INFO(log(), "  grasp %zu %s -> %s at (%.3f, %.3f) on z %.2f: place not reachable (%s)", gi,
                      g.face.c_str(), pl.rest_pose.c_str(), p.x(), p.y(), p.z(),
                      ctx_->why_unreachable(T_obj * T_obj_tcp, &*held[gi]).c_str());
        }
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (plans->empty()) {
      return fail("no grasp + placement with reachable poses (" + std::to_string(checked) + " IK checks)");
    }
    RCLCPP_INFO(log(), "%zu plans from %zu grasps x %zu placements (%d IK checks, %.0f ms)", plans->size(),
                grasps->size(), placements->size(), checked, ms);
    setOutput("plans", plans);
    return NodeStatus::SUCCESS;
  }

private:
  std::vector<bool> reported_;
};

// Next plan; with same_grasp=true only plans using the grasp already taken (the object is in the hand).
class PopPlan : public Leaf
{
public:
  using Leaf::Leaf;
  static BT::PortsList providedPorts()
  {
    return {BT::BidirectionalPort<PlanQueue>("plans"), InputPort<bool>("same_grasp", false, "only plans with the grasp in hand"),
            BT::BidirectionalPort<Grasp>("grasp"), OutputPort<Pose>("pregrasp"), OutputPort<Pose>("grasp_pose"),
            OutputPort<Pose>("lifted"), OutputPort<Pose>("preplace"), OutputPort<Pose>("place"),
            OutputPort<Pose>("depart")};
  }
  NodeStatus tick() override
  {
    auto q = in<PlanQueue>("plans");
    const bool same = in<bool>("same_grasp");
    Grasp taken;
    if (same) {
      taken = in<Grasp>("grasp");
    }
    while (q && !q->empty()) {
      const PickPlacePlan pp = q->front();
      q->pop_front();
      if (same && (pp.grasp.face != taken.face || pp.grasp.tcp.pose != taken.tcp.pose)) {
        continue;
      }
      setOutput("grasp", pp.grasp);
      setOutput("pregrasp", pp.pregrasp);
      setOutput("grasp_pose", pp.grasp_pose);
      setOutput("lifted", pp.lifted);
      setOutput("preplace", pp.preplace);
      setOutput("place", pp.place);
      setOutput("depart", pp.depart);
      RCLCPP_INFO(log(), "plan: grasp %s, place at (%.3f, %.3f) on z %.2f, in %s (%zu left)", pp.grasp.face.c_str(),
                  pp.placement.point.point.x, pp.placement.point.point.y, pp.placement.point.point.z,
                  pp.approach.c_str(), q->size());
      return NodeStatus::SUCCESS;
    }
    return fail(same ? "no other placement for the grasp in hand" : "no plan left");
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
    return {InputPort<sensor_msgs::msg::PointCloud2>("obstacles"), InputPort<std::string>("id", "table_clutter", "object id"),
            InputPort<double>("voxel", 0.025, "m"),
            InputPort<int>("min_points", 4, "points per voxel (drops flying pixels)")};
  }
  NodeStatus tick() override
  {
    const auto cloud = in<sensor_msgs::msg::PointCloud2>("obstacles");
    const double v = in<double>("voxel");
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
      if (n < in<int>("min_points")) {
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

// The held object (its mesh) rides on the hand, so MoveIt keeps it clear of the shelf.
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
    RCLCPP_INFO(log(), "%s attached to the hand", target.c_str());
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
    moveit_msgs::msg::AttachedCollisionObject aco;
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
  reg<MoveToPose>(f, "MoveToPose", ctx);
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
