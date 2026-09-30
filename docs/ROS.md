# ROS 2 pipeline — Isaac Sim cell, perception nodes, MoveIt 2, behavior tree

The five vision stages as ROS 2 Jazzy nodes, driving a simulated Franka Panda through MoveIt 2 and ros2_control,
sequenced by a BehaviorTree.CPP task manager. The pipeline only uses what a real cell would have (D-010): camera
images, joint states, calibrated extrinsics, the cell's fixed geometry and a prompt. Ground truth goes only to
`eval_node` and `sim/episodes.py`.

## Commands

Workstation, one Cursor terminal each (Jazzy is sourced by `.bashrc`).

| # | What | Command |
|---|---|---|
| 1 | Sim, live over ROS | `.venv-sim/bin/python sim/ros_cell.py` (wait for `[ros_cell] running`) |
| 2 | Robot, MoveIt, perception (+ FoundationPose container) | `source ros2/install/setup.bash && ros2 launch ppp_bringup all.launch.py eval:=true` |
| 3 | The task: mustard → shelf, tomato can → shelf | `source ros2/install/setup.bash && ros2 launch ppp_bringup task.launch.py` |
| — | One target | `ros2 launch ppp_bringup task.launch.py targets:='[tomato_soup_can]'` |
| — | New object yaws | `ros2 service call /sim/reset std_srvs/srv/Trigger` |
| — | Where did things end up (ground truth) | `ros2 service call /eval_node/report std_srvs/srv/Trigger` |
| — | N randomised episodes | `.venv/bin/python sim/episodes.py -n 10` → `output/episodes.json` |
| — | Watch the tree live | Groot2 → connect to `localhost:1667` while the task runs |
| — | View in Foxglove (laptop) | add `foxglove:=true` to step 2, open `ws://192.168.33.118:8765` |
| — | Watch the sim (laptop) | add `--livestream` to step 1, Isaac Sim WebRTC Streaming Client → `192.168.33.118` ([SIM.md](SIM.md#watching-the-scene-from-the-laptop-livestream)) |

Restarting the sim restarts sim time at 0: restart step 2 as well (nodes keep old TF otherwise). Stop the old sim
first: `ros_cell.py` exits if another sim already publishes `/clock` (two clocks flood TF_OLD_DATA, I-030).
If every move aborts with `unknown goal response`, an old step 2 is still running: `ros2 node list | sort | uniq -d`
must print nothing (I-031).

## Setup (once)

| What | Command |
|---|---|
| apt (needs sudo) | `sudo apt install ros-jazzy-moveit ros-jazzy-moveit-resources-panda-moveit-config ros-jazzy-moveit-resources-panda-description ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-behaviortree-cpp ros-jazzy-vision-msgs ros-jazzy-foxglove-bridge` |
| pybind11 for numpy 2 | `.venv/bin/pip install -r requirements.txt` |
| Build the workspace | `ros2/build.sh` |
| FoundationPose image (~40 GB, ~30 min) | `docker build -f docker/Dockerfile.isaac_ros -t ppp-isaac-ros:4.5 docker/` |
| Camera extrinsics (GT stand-in, I-018) | `.venv/bin/python sim/write_calibration.py` |

The ONNX models and TensorRT engines are fetched / built by `docker/foundationpose.sh` on first start
(`models/isaac_ros/foundationpose/`, NVIDIA's model license applies).

## Architecture

```
Isaac Sim (sim/ros_cell.py) ──RGB-D, /clock, /joint_states──►  ros2_control controllers (inside Isaac, D-015)
   ▲  follow_joint_trajectory / gripper_cmd                          ▲
   │                                                                 │
MoveIt 2 (move_group) ◄──────── task_manager (BehaviorTree.CPP) ─────┘
                                   │  EstimatePose   PlanGrasps   GetPlacements
                                   ▼
   pose_node ─► detect_node (YOLOE) ─► /ppp/fp/<target>/* ─► FoundationPose ×2 (Isaac ROS, Docker)
   grasp_node ─► spatial_node (C++, ppp_geometry RANSAC)
   place_node (C++ RANSAC via pybind11)                          eval_node ◄── /sim/gt/*  (evaluation only)
```

| Package | Language | What |
|---|---|---|
| `ppp_interfaces` | msg/srv | `Detect`, `EstimatePose`, `AnalyzeTable`, `PlanGrasps`, `GetPlacements`, `Grasp`, `Placement` |
| `ppp_geometry` | C++ + pybind11 | Plane RANSAC + sequential RANSAC shared by stages 3 and 5 (D-019); gtest + a numpy-equivalence test |
| `ppp_spatial` | C++ | `spatial_node` (stage 3): table plane, occupancy grid, points on the table |
| `ppp_perception` | Python | `detect_node`, `pose_node`, `grasp_node`, `place_node`, `eval_node` around `vision/` |
| `ppp_task` | C++ | `task_manager` + `trees/pick_and_place.xml` |
| `ppp_bringup` | launch | `robot`, `moveit`, `perception`, `task`, `all`, `foundationpose` (in the container); cell config |

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
| Perceive | home → **EstimatePose** (detect + FoundationPose) → **PlanGrasps** (table + grasps, obstacles) → UpdateScene (clutter voxels into MoveIt) → look at the shelf → **GetPlacements** → **SelectPlans** → AddTarget |
| Placement pose | stage 5 offers placements per stable rest pose of the mesh, upright first ([D-031](DECISIONS.md#d-031)): the lying can is stood up; a bottle too tall for the free space would be laid down |
| Pick | best plan: MoveToPose(pregrasp) → RemoveTarget → MoveLinear(grasp) → close → CheckGrasp → MoveLinear(lift) → CheckGrasp → AttachObject; next plan if a motion fails |
| Place | MoveToPose(preplace) → MoveLinear(place), else another placement for the same grasp → CheckGrasp → open → DetachObject → MoveLinear(back out) |
| Recover | still holding it → put it back where it was picked; hand empty → home; can't put it back → keep holding, home |

**SelectPlans (look before you pick).** A grasp has to suit the place too: a top-down grasp can't put a 19 cm bottle
into a 30 cm shelf gap (no room above it for the wrist) or on the top board (out of reach). So for each placement
(preferred rest pose first, then best clearance) × grasp (least tilted first), the object is set down in that rest
pose, turned about vertical so that a side grasp points into the shelf, and every pose on the way (pregrasp, grasp, lift,
preplace, place) must have a collision-free IK solution (MoveIt `/compute_ik`, held mesh attached from the lift on).

## Measured (sim, L40S)

| What | Result |
|---|---|
| Camera extrinsics through TF (joint states → Panda URDF → calibration) vs sim | 0.0 mm, 0.000° (fixed and wrist camera) |
| Stage 3 in C++ (`spatial_node`) | table z 0.7 mm (true 0), tilt 0.06°, 11 ms |
| C++ vs numpy RANSAC (D-019 tolerances) | table plane and all shelf supports within 0.5° / 2 mm / 1 % |
| Stage 1 visual prompt alone: right tomato can | 7/12 and 9/12 on two sets of random scenes (the potted meat can looks alike) |
| Stage 1 visual + text check | 20 of 24 accepted, all right; the 4 rejected were all wrong |
| Stage 2 Isaac ROS FoundationPose, live | 0.4–1.4 mm; can 0.5° axis error; mustard 1.4° (sometimes flipped 180° about its axis, I-024) |
| Arm tracking after settling (sim drives, D-026) | 0.2–7 mm TCP error |
| Episodes (`sim/episodes.py -n 8`, real perception end to end) | tomato can **6/8** placed (stood upright), mustard bottle **3/8**; the task's own verdict matched ground truth in 13/16 (the other 3: the object ended on the shelf anyway after a failed attempt) |

## Files

| Path | What |
|---|---|
| `sim/ros_cell.py` | Isaac Sim over ROS 2 (cameras, clock, ros2_control, ground truth, reset) |
| `sim/cell.py` | Scene building shared with `sim/scene.py` |
| `sim/write_calibration.py` | Camera extrinsics file (ground-truth stand-in) |
| `sim/episodes.py` | Randomised episodes, success judged on ground truth |
| `docker/Dockerfile.isaac_ros`, `docker/foundationpose.sh` | Isaac ROS 4.5 FoundationPose image and start script |
| `ros2/build.sh` | colcon build with the right pybind11 |
| `ros2/src/ppp_bringup/config/` | `ros2_controllers.yaml`, `calibration.yaml` |
