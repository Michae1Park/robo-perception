# ROS 2 pipeline

The five vision stages as ROS 2 Jazzy nodes, driving the simulated Panda through MoveIt 2 and ros2_control, sequenced
by a BehaviorTree.CPP task manager. The pipeline only uses what a real cell would have ([D-010](DECISIONS.md#d-010)):
camera images, joint states, calibrated extrinsics, the cell's fixed geometry and a prompt. Ground truth goes only to
`eval_node` and `sim/episodes.py`.

## Commands

Workstation, one terminal each (Jazzy is sourced by `.bashrc`).

| # | What | Command |
|---|---|---|
| 1 | Sim, live over ROS | `.venv-sim/bin/python sim/ros_cell.py` (wait for `[ros_cell] running`) |
| 2 | Robot, MoveIt, perception (+ FoundationPose container) | `source ros2/install/setup.bash && ros2 launch ppp_bringup all.launch.py eval:=true` |
| 3 | The task: mustard → shelf, tomato can → shelf | `source ros2/install/setup.bash && ros2 launch ppp_bringup task.launch.py` |
| — | One target | `ros2 launch ppp_bringup task.launch.py targets:='[tomato_soup_can]'` |
| — | New object yaws | `ros2 service call /sim/reset std_srvs/srv/Trigger` |
| — | Where things ended up (ground truth) | `ros2 service call /eval_node/report std_srvs/srv/Trigger` |
| — | N randomised episodes | `.venv/bin/python sim/episodes.py -n 10` → `output/episodes.json` |
| — | Watch the tree | Groot2 → `localhost:1667` while the task runs |
| — | ROS data on the laptop | add `foxglove:=true` to step 2, open `ws://192.168.33.118:8765` in Foxglove |
| — | The sim on the laptop | add `--livestream` to step 1 ([SIM.md](SIM.md#watching-from-the-laptop-livestream)) |

**Pitfalls.** Run one sim at a time: `ros_cell.py` exits if `/clock` already has a publisher (I-030). After restarting
the sim, restart step 2 too: sim time restarts at 0 and running nodes keep stale TF (I-026). If every move aborts,
an old step 2 is still running: `ros2 node list | sort | uniq -d` must print nothing (I-031).

## Setup (once)

| What | Command |
|---|---|
| apt (needs sudo) | `sudo apt install ros-jazzy-moveit ros-jazzy-moveit-resources-panda-moveit-config ros-jazzy-moveit-resources-panda-description ros-jazzy-pick-ik ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-behaviortree-cpp ros-jazzy-vision-msgs ros-jazzy-foxglove-bridge` |
| pybind11 for numpy 2 | `.venv/bin/pip install -r requirements.txt` |
| Build the workspace | `ros2/build.sh` |
| FoundationPose image (~40 GB, ~30 min) | `docker build -f docker/Dockerfile.isaac_ros -t ppp-isaac-ros:4.5 docker/` |
| Camera extrinsics (ground-truth stand-in, I-018) | `.venv/bin/python sim/write_calibration.py` (again after moving a camera) |

`docker/foundationpose.sh` fetches the ONNX models and builds the TensorRT engines on first start
(`models/isaac_ros/foundationpose/`, NVIDIA's model license applies).

## Architecture

```
Isaac Sim (sim/ros_cell.py) ──RGB-D, /clock, /joint_states──►  ros2_control controllers (inside Isaac, D-015)
   ▲  follow_joint_trajectory / gripper_cmd                          ▲
   │                                                                 │
MoveIt 2 (move_group, pick_ik) ◄── task_manager (BehaviorTree.CPP) ──┘
                                   │  EstimatePose   PlanGrasps   GetPlacements
                                   ▼
   pose_node ─► detect_node (YOLOE) ─► /ppp/fp/<target>/* ─► FoundationPose ×2 (Isaac ROS, Docker)
   grasp_node ─► spatial_node (C++, ppp_geometry RANSAC)
   place_node (C++ RANSAC via pybind11)                          eval_node ◄── /sim/gt/*  (evaluation only)
```

| Package | Language | What |
|---|---|---|
| `ppp_interfaces` | msg/srv | `Detect`, `EstimatePose`, `AnalyzeTable`, `PlanGrasps`, `GetPlacements`, `Grasp`, `Placement` |
| `ppp_geometry` | C++ + pybind11 | Plane RANSAC + sequential RANSAC shared by stages 3 and 5 (D-019) |
| `ppp_spatial` | C++ | `spatial_node` (stage 3): table plane, occupancy grid, points on the table |
| `ppp_perception` | Python | `detect_node`, `pose_node`, `grasp_node`, `place_node`, `eval_node`, thin wrappers around `vision/` |
| `ppp_task` | C++ | `task_manager` + `trees/pick_and_place.xml` |
| `ppp_bringup` | launch | `robot`, `moveit`, `perception`, `task`, `all`, `foundationpose` (in the container); `common.py`: MoveIt config with the longer fingers |

## Topics and services

| Name | Type | From → to |
|---|---|---|
| `/camera/camera/color/image_raw`, `.../aligned_depth_to_color/image_raw` (16UC1 mm), `.../camera_info` | Image | sim → detect, spatial |
| `/wrist_camera/camera/...` | Image | sim → place |
| `/detect_node/detect` | Detect | pose_node → detect |
| `/ppp/fp/<target>/{image, depth_image (32FC1 m), camera_info, segmentation}` | Image | detect → FoundationPose |
| `/ppp/fp/<target>/output` | Detection3DArray | FoundationPose → pose_node |
| `/pose_node/estimate_pose` | EstimatePose | task → pose |
| `/spatial_node/analyze_table` | AnalyzeTable | grasp → spatial |
| `/grasp_node/plan_grasps` | PlanGrasps | task → grasp |
| `/place_node/get_placements` | GetPlacements | task → place |
| `/sim/gt/<object>/pose`, `/sim/gt/camera_pose`, `/sim/gt/wrist_camera_pose`, `/sim/reset` | — | sim → eval only |

## The task (behavior tree)

`task_manager` runs `PickAndPlace` once per target, and `Recover` after a failure.

| Subtree | Steps |
|---|---|
| Perceive | home → **EstimatePose** (detect + FoundationPose) → **PlanGrasps** (table + grasps) → clutter into MoveIt → look at the shelf → **GetPlacements** → shelf items into MoveIt → **SelectPlans** → target into MoveIt |
| Pick | next plan: pregrasp → straight in → close → straight up (18 cm) → still holding it? → it becomes part of the robot for planning |
| Place | preplace → straight in → still holding it? → open → back out along the gripper axis |
| Recover | still holding it → put it back where it was picked; then home |

**Look before you pick** ([D-023](DECISIONS.md#d-023)). The grasp has to suit the place: a side grasp can go into a
shelf gap, a top-down one only where there is room above for the hand. So the shelf is seen first, and `SelectPlans` pairs
placements (upright first, [D-031](DECISIONS.md#d-031)) with grasps (least tilted first). The object is set down in
the placement's rest pose, turned so that a side grasp points into the shelf.

**Clean motions** ([D-037](DECISIONS.md#d-037)). Every pose of a plan must have a collision-free IK solution, and the
solutions are chained: each is seeded with the one before (joint 1 turned towards free-space goals), and **pick_ik**
returns the solution nearest its seed. The arm therefore keeps one natural configuration, and free-space moves go to
*joint* targets rather than poses, so the planner can't pick a flipped configuration. Each free-space move tries, in
order ([D-040](DECISIONS.md#d-040)):

| # | Planner | Why |
|---|---|---|
| 1 | Pilz PTP | The straight line in joint space: the shortest motion, when nothing is in the way |
| 2 | PTP via home | Two straight lines through home (up and back): clears the board edges with an object in the hand |
| 3 | STOMP | Bends the straight line around obstacles: smooth and short, but only a local fix |
| 4 | OMPL RRTConnect | Sampling-based: whatever is left |

Every result is collision-checked every 0.02 rad before it runs (Pilz and STOMP check at their own, sparser samples;
the shelf boards are 2 cm thick), and OMPL's own path checking is 10× finer than MoveIt's default. Approach, lift and retreat are straight TCP lines, checked
every 5 mm.

**The held object is part of the robot.** From the lift until release, the object's mesh (padded 5 mm) is attached to
`panda_hand` as a MoveIt `AttachedCollisionObject`. Every IK, plan and collision check then includes it, against the
shelf, the clutter voxels and the robot itself: the dynamic equivalent of adding it to the URDF.

## Measured (sim, L40S)

| What | Result |
|---|---|
| Camera extrinsics through TF vs sim | 0.0 mm, 0.000° (fixed and wrist camera) |
| Stage 3 in C++ (`spatial_node`) | table z 0.7 mm (true 0), tilt 0.06°, 11 ms |
| C++ vs numpy RANSAC (D-019 tolerances) | table plane and all shelf supports within 0.5° / 2 mm / 1 % |
| Stage 1 visual + text check (old layout, 24 scenes) | 20 accepted, all right; the 4 rejected were all wrong |
| Stage 2 Isaac ROS FoundationPose, live | 0.4–1.4 mm; can 0.5° axis error; mustard 1.4° (sometimes flipped 180° about its axis, I-024) |
| Motions (D-037) | straight joint-space lines of 2.6–5 rad; via home ~10 rad when the direct line would sweep the held object through the shelf (before: OMPL detours of 10–15 rad) |
| Episodes (`sim/episodes.py -n 10`) | see [DECISIONS.md D-037](DECISIONS.md#d-037) |

## Files

| Path | What |
|---|---|
| `sim/ros_cell.py` | Isaac Sim over ROS 2 (cameras, clock, ros2_control, ground truth, reset, viewport overlay) |
| `sim/cell.py` | Scene building shared with `sim/scene.py` |
| `sim/write_calibration.py` | Camera extrinsics file (ground-truth stand-in) |
| `sim/episodes.py` | Randomised episodes, success judged on ground truth |
| `docker/Dockerfile.isaac_ros`, `docker/foundationpose.sh` | Isaac ROS 4.5 FoundationPose image and start script |
| `ros2/build.sh` | colcon build with the right pybind11 |
| `ros2/src/ppp_bringup/config/` | `ros2_controllers.yaml`, `calibration.yaml` |
