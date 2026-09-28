# Setup and run

Tested target: Ubuntu 22.04, NVIDIA L40S (headless), Isaac Sim 4.5 / 5.x, Isaac ROS 3.x (ROS 2 Humble).
Two processes talk over DDS on the same host:

| Process | Where | What |
|---|---|---|
| Isaac Sim | host (or Isaac Sim container) | `isaac_sim/scene.py`: physics, rendering, RGB-D, Franka, ground truth |
| ROS 2 pipeline | Isaac ROS dev container (`--network host`) | FoundationPose, YOLOE, MoveIt 2, planners, task manager, evaluation |

Both sides: same `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`. If topics show up in
`ros2 topic list` but carry no data, export `FASTRTPS_DEFAULT_PROFILES_FILE=<repo>/docker/fastdds_udp.xml`
on both sides.

## 1. Repository and assets

```bash
git clone git@github.com:Michae1Park/prompt-pick-place.git ~/prompt-pick-place
export PPP_ROOT=~/prompt-pick-place
cd $PPP_ROOT
python3 scripts/prepare_ycb.py          # downloads the 6 YCB scans -> assets/ycb/<name>/ (numpy, pyyaml)
```

Each mesh is yaw-aligned and centred once; the same OBJ is used by the simulator (converted to USD with
`isaac_sim/ycb_usd.py`), FoundationPose, grasp planning and evaluation, so all poses share one frame.

## 2. Isaac Sim

```bash
export ISAACSIM=~/isaacsim                       # install root (python.sh lives here)
export ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=0
# use Isaac Sim's bundled ROS 2 Humble libraries (do not source a system ROS install here)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAACSIM/exts/isaacsim.ros2.bridge/humble/lib

# once: render the visual prompts (one example image per object, side view) -> assets/prompts/
$ISAACSIM/python.sh isaac_sim/capture_prompts.py

# the cell (headless + WebRTC; connect with the Isaac Sim WebRTC Streaming Client to this host)
$ISAACSIM/python.sh isaac_sim/scene.py --livestream
```

`pyyaml` is required inside Isaac Sim's Python (`$ISAACSIM/python.sh -m pip install pyyaml` if missing).
WebRTC needs TCP 49100 and UDP 47998 reachable from the viewer.

Sanity check from any ROS 2 shell: `ros2 topic hz /camera/color/image_raw` (~10 Hz), `ros2 topic echo /sim/ready`.

## 3. Isaac ROS container

```bash
export ISAAC_ROS_WS=~/workspaces/isaac_ros-dev
mkdir -p $ISAAC_ROS_WS/src && cd $ISAAC_ROS_WS/src
git clone -b release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
ln -s $PPP_ROOT prompt-pick-place

# image layer with FoundationPose, MoveIt 2 and ultralytics (docker/Dockerfile.ppp)
cat > ~/.isaac_ros_common-config <<EOF
CONFIG_IMAGE_KEY=ros2_humble.ppp
CONFIG_DOCKER_SEARCH_DIRS=($PPP_ROOT/docker)
EOF
./isaac_ros_common/scripts/run_dev.sh -d $ISAAC_ROS_WS
```

Inside the container:

```bash
# FoundationPose models (NGC) + TensorRT engines (built once, shared by every object instance)
mkdir -p ${ISAAC_ROS_WS}/isaac_ros_assets/models/foundationpose && cd $_
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.0_onnx/files/refine_model.onnx' -O refine_model.onnx
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.0_onnx/files/score_model.onnx' -O score_model.onnx
$ISAAC_ROS_WS/src/prompt-pick-place/scripts/build_fp_engines.sh

# build the pipeline
cd $ISAAC_ROS_WS
colcon build --symlink-install --base-paths src/prompt-pick-place/ros2
source install/setup.bash
export PPP_ASSETS=$ISAAC_ROS_WS/src/prompt-pick-place/assets
```

(If the NGC URLs moved, follow the model download step of the Isaac ROS FoundationPose quickstart.)

## 4. Run

```bash
# whole ROS 2 side: move_group, FoundationPose x6, YOLOE, spatial perception, planners, executor, task manager
ros2 launch ppp_bringup pipeline.launch.py rviz:=true

# one pick-and-place, target given by its visual prompt name
ros2 action send_goal --feedback /pick_place ppp_interfaces/action/PickPlace "{target: 006_mustard_bottle}"

# new random layout
ros2 topic pub --once /sim/reset std_msgs/String "{data: '{\"seed\": 7, \"force\": true}'}"
```

Useful launch arguments: `yoloe_engine:=<path>` (TensorRT backend), `yoloe_weights:=yoloe-11s-seg.pt`,
`foundationpose:=false` (start FoundationPose separately), `fp_models_dir:=...`.

## 5. Evaluation

```bash
ros2 launch ppp_bringup eval.launch.py num_trials:=50 seed:=0 output_dir:=$PPP_ROOT/results/eval_n50
ros2 run ppp_eval summarize $PPP_ROOT/results/eval_n50/trials.jsonl --by-object
```

## 6. TensorRT

```bash
cd $PPP_ROOT
python3 scripts/snapshot.py --out results/frames --topics /camera/color/image_raw --count 30   # real scene frames
python3 scripts/yoloe_trt.py export --out models/yoloe_vp.engine
python3 scripts/yoloe_trt.py bench --engine models/yoloe_vp.engine --images 'results/frames/*.png'
ros2 launch ppp_bringup pipeline.launch.py yoloe_engine:=$PPP_ROOT/models/yoloe_vp.engine
```

## 7. Figures for the README

```bash
python3 scripts/snapshot.py --out results/figures      # camera, detection overlay, pose overlay (after a goal)
```

RViz (`rviz:=true`) shows the table-plane inliers, the place-zone occupancy grid, grasp candidates
(green = chosen, yellow = reachable, grey = rejected by IK) and the planned placement box.

## Unit tests (no GPU / ROS needed)

```bash
pip install numpy pyyaml pytest && python3 -m pytest tests
```
