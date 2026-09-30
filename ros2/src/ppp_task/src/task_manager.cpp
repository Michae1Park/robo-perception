// Task manager: runs the pick-and-place behavior tree (trees/pick_and_place.xml) once per target, with Recover
// after a failure, prints a summary and exits. Groot2 can attach live (port 1667) to watch the tree.
//
//   ros2 launch ppp_bringup task.launch.py targets:="[mustard_bottle, tomato_soup_can]"
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include <thread>

#include "ppp_task/bt_nodes.hpp"
#include "ppp_task/context.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("task_manager");
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  std::thread spinner([&exec] { exec.spin(); });

  const auto tree_file = node->declare_parameter(
    "tree", ament_index_cpp::get_package_share_directory("ppp_task") + "/trees/pick_and_place.xml");
  const auto targets = node->declare_parameter<std::vector<std::string>>(
    "targets", {"mustard_bottle", "tomato_soup_can"});
  const bool groot = node->declare_parameter("groot", true);   // live view in Groot2, port 1667
  auto ctx = std::make_shared<ppp_task::Context>(node);

  BT::BehaviorTreeFactory factory;
  ppp_task::register_nodes(factory, ctx);
  factory.registerBehaviorTreeFromFile(tree_file);

  // back to the cell's fixed geometry: drop what this task put into the planning scene (also after a failure)
  auto clear_scene = [&] {
    std::vector<std::string> mine{"table_clutter", "shelf_clutter"};
    for (const auto & t : targets) {
      mine.push_back("target_" + t);
      mine.push_back("held_" + t);
      moveit_msgs::msg::AttachedCollisionObject aco;
      aco.link_name = "panda_hand";
      aco.object.id = "held_" + t;
      aco.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      ctx->scene.applyAttachedCollisionObject(aco);
    }
    ctx->scene.removeCollisionObjects(mine);
    ctx->carrying = false;
  };
  auto run = [&](const std::string & id, const BT::Blackboard::Ptr & bb) {
    auto tree = factory.createTree(id, bb);
    std::unique_ptr<BT::Groot2Publisher> groot_pub;
    if (groot) {
      groot_pub = std::make_unique<BT::Groot2Publisher>(tree, 1667);
    }
    return tree.tickWhileRunning(std::chrono::milliseconds(10));
  };

  while (rclcpp::ok() && node->now().nanoseconds() == 0) {   // sim time: wait for /clock
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::vector<std::pair<std::string, bool>> results;
  for (const auto & target : targets) {
    if (!rclcpp::ok()) {
      break;
    }
    RCLCPP_INFO(node->get_logger(), "===== %s =====", target.c_str());
    clear_scene();
    ctx->target_pub->publish(std_msgs::msg::String().set__data(target));
    const auto t0 = node->now();
    auto bb = BT::Blackboard::create();   // shared by PickAndPlace and Recover (where the object was picked)
    bb->set("target", target);
    bool ok = false;
    try {
      ok = run("PickAndPlace", bb) == BT::NodeStatus::SUCCESS;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(node->get_logger(), "tree error: %s", e.what());
    }
    RCLCPP_INFO(node->get_logger(), "===== %s: %s in %.0f s (sim time) =====", target.c_str(),
                ok ? "PLACED" : "FAILED", (node->now() - t0).seconds());
    if (!ok) {
      try {
        run("Recover", bb);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(node->get_logger(), "recover error: %s", e.what());
      }
    }
    results.emplace_back(target, ok);
    ctx->target_pub->publish(std_msgs::msg::String());   // done with it: viewers clear its overlay
  }
  int placed = 0;
  for (const auto & [t, ok] : results) {
    RCLCPP_INFO(node->get_logger(), "  %-20s %s", t.c_str(), ok ? "placed" : "failed");
    placed += ok;
  }
  RCLCPP_INFO(node->get_logger(), "done: %d/%zu placed", placed, results.size());

  exec.cancel();
  spinner.join();
  rclcpp::shutdown();
  return placed == static_cast<int>(results.size()) ? 0 : 1;
}
