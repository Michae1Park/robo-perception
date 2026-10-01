# Isaac Sim cell

One cell (`sim/cell.py`, parameters in `config.yaml` → `sim:`), used two ways:

| Script | What |
|---|---|
| `sim/scene.py` | Writes RGB-D datasets in FoundationPose's layout, with ground-truth masks and poses (stages 2 and 5 offline) |
| `sim/ros_cell.py` | The same cell live over ROS 2: cameras, clock, ros2_control. See [ROS.md](ROS.md) |

![Isaac Sim cell](sim/overview.png)

## Layout

World frame = robot base (`panda_link0`): z up, z = 0 is the table top. Chosen from the Panda's reach
([D-035](DECISIONS.md#d-035), [D-040](DECISIONS.md#d-040)): top-down grasps are comfortable 0.5–0.72 m from the
base, horizontal placements with the hand 0.1–0.6 m high. Moving the shelf further away cost placements.

| Piece | What |
|---|---|
| Robot | Franka Panda on a 20 cm pedestal at the origin. Home faces between the table and the shelf (joint 1 at −45°) |
| Gripper | Franka Hand with fingers **5 cm longer** than stock (a box continuing each finger, in the sim and in MoveIt's model), squeezing with **140 N** ([D-036](DECISIONS.md#d-036)) |
| Table | 0.40 × 0.95 m, long side along y, in front of the robot (x 0.40–0.80 m, y −0.30–0.65 m), top at z = 0 |
| Pick targets | `006_mustard_bottle` (upright, random yaw ±30°) and `005_tomato_soup_can` (**lying on its side**, any yaw), 0.55–0.65 m from the base |
| Clutter | Sugar box, foam brick, Rubik's cube: obstacles, never targets. No potted meat can anywhere: YOLOE mistakes it for the tomato can ([I-025](DECISIONS.md#i-025)) |
| Shelf | Metal pantry, 2.05 m tall: four corner posts and five boards, open at the back and sides ([D-041](DECISIONS.md#d-041)). 0.80 m wide, 0.35 m deep, on the robot's right: front 0.47 m from the base, 0.17 m clear of the table. Board tops at z = −0.35, 0.05, 0.45, 0.95 and 1.30 m; the arm works on 0.05 (0.38 m clear above) and 0.45 (0.48 m) |
| Shelf items | Sugar box on the 0.05 m board, Rubik's cube + foam brick on the 0.45 m board: stage 5 must find the space around them |
| Fixed camera | RealSense D455 (NVIDIA model) on a floor tripod beyond the table's far side, 0.85–0.95 m from the targets, all five items in view. Pinhole 640×480, fx = fy = 615 ([D-007](DECISIONS.md#d-007)) |
| Wrist camera | D455 flat on the side of the hand, looking along the fingers. `robot.look_joints` puts it at (0.12, −0.12, 0.90) looking into the shelf |

Both camera bodies and the tripod are visual only (no colliders), and never appear in their own images.

| Camera rig | D455 model | Fixed camera view |
|---|---|---|
| ![Tripod rig](sim/rig.png) | ![D455 front](sim/d455.png) | ![Fixed camera](sim/camera.png) |

## Setup (once)

Isaac Sim 6.1 gets its own venv, `.venv-sim` (Python 3.12, ~26 GB): `scripts/install.sh sim`
([README](../README.md#install)).

The first run compiles shaders (~3 min, once). After that a `scene.py` run takes ~15 s. Robot and camera assets come from NVIDIA's asset server.

## Datasets (`scene.py`)

```bash
.venv-sim/bin/python sim/scene.py                     # 1 frame -> data/sim/
.venv-sim/bin/python sim/scene.py --frames 10         # 10 frames, new random target yaws each
.venv-sim/bin/python sim/scene.py --overview docs/sim/overview.png   # the still above
```

Each frame: the table shot with the arm at home, then the arm moves to `look_joints` and the wrist camera takes the
shelf shot. `data/sim/` (gitignored) is rewritten each run:

| File | What |
|---|---|
| `scene/rgb/`, `scene/depth/` | Fixed camera: RGB, depth (uint16 **mm**, 0 = none) |
| `scene/cam_K.txt`, `scene/T_base_cam.txt` | Intrinsics; **ground-truth** camera pose in the base frame (OpenCV optical frame) |
| `<target>/masks/`, `<target>/ob_in_cam/` | The target's visible pixels (occlusion-aware); ground-truth pose in the camera frame |
| `<target>/mesh` | → `assets/ycb/<target>/`: the same OBJ FoundationPose loads, so estimate and ground truth share one object frame |
| `shelf/rgb/`, `shelf/depth/`, `shelf/T_base_cam.txt` | Wrist camera shot of the shelf, and its pose (= forward kinematics) |
| `shelf/shelf.json` | Ground-truth shelf: board heights, footprint, thickness |

Each `<target>/` is a complete FoundationPose dataset (`rgb`, `depth`, `cam_K.txt` link to `scene/`), so stage 2 runs
on it directly and reports the error against ground truth:

```bash
.venv/bin/python pipeline/02_pose.py --data data/sim/006_mustard_bottle --frame 0 --tag sim_mustard
#   ...
#   vs ground truth: rotation 1.14 deg | translation 0.2 mm
```

| Target (3 frames, seed 0) | Rotation error | Translation error |
|---|---|---|
| Mustard bottle (upright) | 0.49 / 1.48 / 0.92° | 0.7 / 0.2 / 0.6 mm |
| Tomato can (lying) | 1.14 / 0.94 / 0.58° | 0.2 / 0.4 / 0.7 mm |

~0.9 s per registration. Sim depth is perfect and the masks exact: FoundationPose's best case ([I-015](DECISIONS.md#i-015)).

## Watching from the laptop (livestream)

The workstation renders; the laptop only shows a WebRTC stream.

| Step | Where | What |
|---|---|---|
| 1 | Workstation | `.venv-sim/bin/python sim/ros_cell.py --livestream` (or `scene.py --livestream`), wait for `streaming on ...` |
| 2 | Laptop | Install the **Isaac Sim WebRTC Streaming Client** (Windows build, "Download Isaac Sim" page; once) |
| 3 | Laptop | Connect to `192.168.33.118` |

- Same LAN (or VPN) only: the video is UDP 47998 (+ TCP 49100 signalling), which the SSH tunnel can't carry. One client
  at a time.
- `scene.py --livestream` is view-only (Replicator never returns data while streaming, [I-002](DECISIONS.md#i-002));
  add `--arm look` to show the shelf-looking pose. `ros_cell.py --livestream` keeps publishing its cameras.
- The view: `config.yaml` `sim.viewport`.

## Knobs (`config.yaml`)

| Knob | What |
|---|---|
| `sim.objects` | `{name, xy, yaw_deg, tilt_deg, target}` per table item. `yaw_deg` fixed or `[lo, hi]` (random per frame); `tilt_deg: 90` = on its side; `target: true` = save mask + ground truth |
| `sim.shelf`, `sim.shelf_objects` | Footprint, board heights, board and post thickness; items on the boards `{name, level, x, yaw_deg}` (level = index into `levels_z`) |
| `robot.home_joints`, `robot.look_joints` | Home pose; the pose for the shelf shot |
| `gripper.finger_extension` | Extra finger length (sim, MoveIt model, grasp model, TCP) |
| `sim.gripper` | Finger drive: stiffness, max force (N), pad friction |
| `sim.arm_drive` | Arm joint drive stiffness / damping |
| `sim.camera`, `sim.wrist_camera` | Intrinsics, fixed camera eye / target, tripod; wrist mount |
| `sim.colors` | sRGB 0–1 (converted to linear for USD) |

## Isaac Sim 6.1 notes

- `isaacsim.core.api` / `core.prims` still work but live in `extsDeprecated/`; the Franka example class isn't loaded,
  so `cell.py` references the Franka USD directly ([I-004](DECISIONS.md#i-004)).
- Set joint positions **and** drive targets, or the arm sags back to the drive targets ([I-005](DECISIONS.md#i-005)).
- USD material colours are linear: an sRGB value used as-is renders much paler ([I-006](DECISIONS.md#i-006)).
- The finger mimic joint drifts under a large squeeze: the articulation runs 64 solver iterations (default 32).
- Livestream: `omni.kit.livestream.app` on the default headless app; the bundled streaming app hung at start
  ([I-003](DECISIONS.md#i-003)).
- The pip package ships agent guides: `.venv-sim/lib/python3.12/site-packages/isaacsim/skills/`.
