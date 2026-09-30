// Behavior-tree leaves for pick-and-place. All synchronous: each tick runs one step (a service call, a planned
// motion, a gripper command) to completion and returns SUCCESS or FAILURE; the tree decides what to try next.
#pragma once

#include <behaviortree_cpp/bt_factory.h>

#include <memory>

#include "ppp_task/context.hpp"

namespace ppp_task
{

void register_nodes(BT::BehaviorTreeFactory & factory, const std::shared_ptr<Context> & ctx);

}  // namespace ppp_task
