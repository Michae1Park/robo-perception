# Decision & issue log

A running record of **why** the project looks the way it does: technical decisions, and the issues found
along the way. Newest entries go at the bottom of each list.

- **Don't rewrite history.** When a decision changes, add a new entry and mark the old one
  `Superseded by D-xxx`. When an issue is fixed, change its status and add how.
- **Status values.** Decisions: `Accepted` (agreed, or already built) · `Proposed` (in the plan, not yet
  agreed or built) · `Superseded`. Issues: `Open` · `Workaround` · `Resolved`.
- Keep entries short. Link commits, docs and files instead of repeating them.

## Decisions

| ID | Date | Decision | Status |
|---|---|---|---|
| [D-001](#d-001) | 2026-09-28 | Keep YOLOE-seg for visual-prompt detection (not OWLv2 + SAM2) | Accepted |
| [D-002](#d-002) | 2026-09-29 | Build and understand the vision pipeline standalone first (ROS 2 / Isaac Sim stack removed) | Accepted (now followed by D-010) |
| [D-003](#d-003) | 2026-09-29 | Stage 1 on the host `.venv`; only FoundationPose runs in Docker | Accepted |
| [D-004](#d-004) | 2026-09-29 | Isaac Sim 6.1 via pip in its own `.venv-sim` | Accepted |
| [D-005](#d-005) | 2026-09-29 | Sim writes FoundationPose-layout datasets + ground truth; sim and FoundationPose share one mesh | Accepted |
| [D-006](#d-006) | 2026-09-29 | `--livestream` is view-only | Accepted |
| [D-007](#d-007) | 2026-09-29 | Fixed camera: D455 body on a tripod, D435 colour intrinsics, ~0.73 m from the targets | Accepted |
| [D-008](#d-008) | 2026-09-29 | Demo picks: mustard bottle (upright) + tomato can (lying on its side) | Accepted |
| [D-009](#d-009) | 2026-09-29 | Shelf 90° to the robot's right is the place target; a wrist camera looks at it | Accepted |
| [D-010](#d-010) | 2026-09-29 | Integrate everything with ROS 2 Jazzy: one node per stage, "no cheating" information flow | Accepted |
| [D-011](#d-011) | 2026-09-29 | Vision nodes stay Python; RANSAC (stage 3) in C++ | Accepted |
| [D-012](#d-012) | 2026-09-29 | Stage 2 via Isaac ROS FoundationPose, pinned to Isaac ROS 4.5 | Accepted |
| [D-013](#d-013) | 2026-09-29 | Robot stays a Franka Panda (demo, not deployment) | Accepted |
| [D-014](#d-014) | 2026-09-29 | Task logic in BehaviorTree.CPP v4 (C++) | Accepted |
| [D-015](#d-015) | 2026-09-29 | Robot control through ros2_control hosted inside Isaac Sim | Accepted (built 2026-09-30) |
| [D-016](#d-016) | 2026-09-29 | Perception is request-driven; one FoundationPose instance per target | Accepted (built 2026-09-30) |
| [D-017](#d-017) | 2026-09-29 | View ROS data from the laptop with Foxglove | Accepted (`foxglove:=true`; not yet tried from the laptop) |
| [D-018](#d-018) | 2026-09-29 | Keep this log in the repo (`docs/DECISIONS.md`) | Accepted |
| [D-019](#d-019) | 2026-09-29 | One shared C++ RANSAC library for stages 3 and 5, with a Python binding | Accepted |
| [D-020](#d-020) | 2026-09-29 | Stage 5 joins the ROS plan: `place_node` + a "look at the shelf" step | Accepted (built 2026-09-30; the look now comes before the pick, D-023) |
| [D-021](#d-021) | 2026-09-29 | Shelf items are never pick targets (tomato can on the shelf → foam brick) | Accepted |
| [D-022](#d-022) | 2026-09-30 | FoundationPose image: NVIDIA's Jazzy base + Isaac ROS release-4.5 apt, installed at build time | Accepted |
| [D-023](#d-023) | 2026-09-30 | Look before you pick: grasp and placement chosen together, by IK checks | Accepted |
| [D-024](#d-024) | 2026-09-30 | Objects go down in their table rest pose; side grasps allowed and turned into the shelf | Superseded by D-031 (orientation); side grasps still Accepted |
| [D-025](#d-025) | 2026-09-30 | Stage 1 in ROS: visual prompt + text check, rejects lookalikes | Accepted |
| [D-026](#d-026) | 2026-09-30 | Sim joint drives set closer to a real Franka (arm damping, 70 N gripper, pad friction) | Accepted |
| [D-027](#d-027) | 2026-09-30 | MoveIt planning scene: fixed cell boxes + table clutter voxels + the target while approaching | Accepted |
| [D-028](#d-028) | 2026-09-30 | Rotationally symmetric objects declared in config.yaml; grasp model turned about the axis | Accepted |
| [D-029](#d-029) | 2026-09-30 | Python nodes run on the `.venv` with Jazzy sourced; pybind11 from the `.venv` | Accepted |
| [D-030](#d-030) | 2026-09-30 | Recovery never drops an object somewhere random | Accepted |
| [D-031](#d-031) | 2026-09-30 | Place in the most natural stable rest pose that fits: upright first (supersedes the orientation rule of D-024) | Accepted |
| [D-032](#d-032) | 2026-09-30 | `ros_cell.py --livestream`: watch the live ROS cell in the WebRTC client | Accepted |

### D-001
**Keep YOLOE-seg (not OWLv2 + SAM2).** One `ultralytics` model gives box + mask in a single pass and was
already working end to end. OWLv2 + SAM2 means two chained models, ~2× latency, and glue work, for a
mask-quality gain that only matters if grasping turns out to need it.
*Revisit only if* YOLOE masks block pose/grasp on hard objects (transparent, reflective, thin).

### D-002
**Standalone vision first.** The earlier ROS 2 / Isaac Sim stack was removed (`e5e8710`) to build and
understand each stage on real RGB-D frames without ROS or a simulator (stages 1–5, `docs/STAGE*.md`).
The old ROS packages (Humble, Python) are still in git history: reference only.

### D-003
**Stage 1 on the host.** YOLOE runs from the host `.venv` (`1e7f05a`). FoundationPose stays in its Docker
container because of its custom CUDA extensions.

### D-004
**Isaac Sim 6.1 in `.venv-sim`.** `pip install "isaacsim[all,extscache]==6.1.0.0"` (Python 3.12, ~25 GB),
separate from the vision `.venv`. Guide: [SIM.md](SIM.md).

### D-005
**Sim datasets = FoundationPose layout + ground truth.** `data/sim/scene/` holds shared RGB-D,
`data/sim/<target>/` holds per-target masks and `ob_in_cam` poses, and `T_base_cam.txt` holds the
extrinsics. The sim converts the **same** `assets/ycb/<obj>/textured.obj` that FoundationPose loads, so
ground truth and estimates share one object frame. Stage 2 reports error vs ground truth via `--data`.

### D-006
**Livestream is view-only.** With WebRTC streaming on, Replicator never returns data (see
[I-002](#i-002)), so `--livestream` shows the scene and saves nothing. Capture runs separately (it can
run at the same time).

### D-007
**Fixed camera.** Rendered as NVIDIA's RealSense **D455** model on a visual-only floor tripod; the colour
lens sits on the optical centre and the rig never appears in its own images. Intrinsics stay at
**fx = 615 (D435 colour)** rather than the D455's real ~320 at 640×480, which would halve the objects in
the image. Moved from 1.05 m to **~0.73 m** from the targets (twice the pixels on the bottle).

### D-008
**Demo picks: mustard (upright) + tomato can lying on its side.** The lying can was tested: it rocks
~7 mm onto a flat of its collision hull in the first 2 s, then stays put (20 s, every yaw). So we kept
the tomato can rather than falling back to the potted meat can (which can't roll).
FoundationPose on the lying can: 0.6–1.1°, < 1 mm (sim, ground-truth mask).

### D-009
**Shelf = place target; wrist camera for it.** Open shelf 90° to the robot's right, 3 levels
(0 / 0.32 / 0.64 m), all within reach (0.6–0.81 m from the shoulder). The shelf is seen by an
**eye-in-hand wrist camera** from a fixed "look at the shelf" arm pose (`cd5338a`), which feeds stage 5
placement (`7ef4cc5`, [STAGE5_PLACE.md](STAGE5_PLACE.md)). Stage 5 uses no shelf model: any horizontal
surface the camera sees is a candidate support.

### D-010
**ROS 2 Jazzy integration, no cheating.** Each vision stage becomes a node. The pipeline may only use
what a real robot would have:

| Allowed into the pipeline | Source |
|---|---|
| Camera images: RGB, aligned depth (`16UC1` mm), `camera_info`, with RealSense-style topic names | Isaac camera publishers |
| `/joint_states`, `/clock` | ros2_control inside Isaac |
| Robot model → TF | `moveit_resources_panda_description` |
| Camera extrinsics (fixed + wrist) | Calibration files (see [I-018](#i-018)) |
| Table / shelf collision boxes for MoveIt | `config.yaml` (measured once, as in a real cell) |
| What to pick | A prompt (example image or text) |

**Never** into the pipeline: `/sim/gt/*` (object poses, true extrinsics) and `/sim/reset`. Only
`eval_node` and test harnesses use them. A launch test will fail if any pipeline node subscribes to
`/sim/gt/*`.

### D-011
**Python vision nodes; C++ RANSAC.** ML stages (YOLOE, and the parts of FoundationPose we own) stay
Python around the tested `vision/` code. Stage 3's table-plane RANSAC + occupancy grid becomes a C++ node
(Eigen), checked against the Python version on the same recorded frames. If that turns out not to be
simple, it stays Python. Open: [Q-001](#q-001).

### D-012
**Isaac ROS FoundationPose, pinned to 4.5.** It's cleaner than rebuilding the FoundationPose container
for Jazzy (Python 3.8 → 3.12, [I-009](#i-009)), and it's C++/TensorRT. Version choice:

| Isaac ROS | ROS 2 | Needs | Fits this machine (driver 580, CUDA 13.0)? |
|---|---|---|---|
| 5.0 (2026-09-21) | Lyrical | driver 595+, CUDA 13.2+ | ✗ wrong ROS distro |
| 4.6 | Jazzy | driver 595+, CUDA 13.2+ | ✗ needs a driver upgrade |
| **4.5** | Jazzy | driver 580+, CUDA 13.0+ | **✓** |

Runs in NVIDIA's Isaac ROS container (`--network host`); models from NGC (ONNX → `trtexec` → `.plan`,
FP32 only). ~7 GB GPU per instance.

### D-013
**Panda.** Matches Isaac's Franka asset and the apt MoveIt config (`moveit_resources_panda_*`). No FR3:
this is a demonstration, not a deployment.

### D-014
**BehaviorTree.CPP v4** (apt: `ros-jazzy-behaviortree-cpp` 4.10) for the task manager, in C++.

### D-015
**ros2_control inside Isaac Sim** (`isaacsim.ros2.control`, 6.1, Jazzy build included): a standard
`controller_manager` runs in-process on the Panda articulation. MoveIt → `joint_trajectory_controller` →
sim is the same chain a real robot would use; only the hardware plugin differs.

### D-016
**Request-driven perception; one FoundationPose per target.** Isaac ROS FoundationPose takes one mesh per
node instance and runs on every mask it receives, so: two instances (mustard, tomato), and `detect_node`
publishes a mask only when the behavior tree asks for an object. `detect_node` re-publishes the exact
RGB / depth / `camera_info` it detected on, with the mask, so FoundationPose's time-synced inputs always
match.

### D-017
**Foxglove for viewing.** `foxglove_bridge` on the workstation, Foxglove on the laptop. RViz needs a
local display, and the WebRTC stream can't run alongside ROS cameras ([I-013](#i-013)).

### D-018
**This log lives in the repo**, versioned with the code it explains.

### D-019
**One C++ RANSAC, used by stages 3 and 5.** Resolves [Q-001](#q-001). In Python they already share
`spatial.fit_plane_ransac()` (stage 5's `place.find_planes()` calls it repeatedly to peel off one
horizontal plane after another). The C++ version keeps that shape:

| Piece | What |
|---|---|
| `ppp_geometry` (C++ library, Eigen) | `fit_plane_ransac()` (largest plane within a tilt of "up", then least-squares refit) + `find_planes()` (sequential RANSAC) |
| `spatial_node` (C++, stage 3) | Links the library directly |
| `place_node` (Python, stage 5) | Calls the same code through a **pybind11** module; the rest of stage 5 stays Python (D-011) |
| `vision/` (offline playgrounds) | Can switch to the binding too, so offline and ROS results come from one implementation |

**Testing:** C++ and numpy draw different random samples, so compare *results*, not bits: plane normal
within 0.5°, offset within 2 mm, inlier count within 1%, on recorded sim frames of the table and shelf.

### D-020
**Stage 5 in the ROS plan.** Resolves [Q-002](#q-002). Add `place_node` (Python): wrist-camera RGB-D +
`T_base_cam` from TF (forward kinematics + wrist hand-eye calibration) → supports → place candidates, as
a `GetPlacements` service. The behavior tree gains a step before placing: move to the "look at the
shelf" joint pose (`sim.wrist_camera.look_joints`), capture, ask `place_node`, then place at the best
reachable candidate.

### D-021
**Shelf items are never pick targets.** Resolves [I-014](#i-014). The tomato can on shelf level 1 became
a **foam brick**. Neither option (swap vs. a table-area filter in `detect_node`) costs GPU or RAM:
FoundationPose instances exist only for the targets, and YOLOE's cost doesn't depend on how many objects
are in view. The swap wins because it removes the ambiguity instead of adding filter logic, and the
shelf item's pose is never needed. Stage 5 results are unchanged (same 7 supports, middle level still
best with 125 mm of room, 8 candidates).

### D-022
**Isaac ROS 4.5 FoundationPose image** (`docker/Dockerfile.isaac_ros`, resolves [I-011](#i-011)). NGC has one Jazzy
base image (CUDA 13.0, TensorRT 10.13, driver 580+); its apt source is switched from `release-4.0` to `release-4.5`
and `ros-jazzy-isaac-ros-foundationpose` installed at build time, where the NVIDIA runtime's GL/EGL bind mounts
don't exist yet. 4.5's pip-shim packages refuse to install until `isaac-ros-cli` names the environment, so
`/etc/isaac-ros-cli/environment.conf` says `docker-activated`, as NVIDIA's own container layer does.
`docker/foundationpose.sh` builds the engines once and runs one namespaced pipeline per target (`/ppp/fp/<target>`)
as the host user, with host network + IPC (DDS shared memory with host processes).

### D-023
**Look before you pick** (`SelectPlans` in `ppp_task`). The grasp has to suit the place as well as the pick: a
top-down grasp can't put the 19 cm mustard bottle into a 30 cm shelf gap (no room for the wrist above it) nor onto
the top board (flange out of reach). So the arm looks at the shelf *before* picking, and each placement (best
clearance first) × grasp (least tilted first) is checked: pregrasp, grasp, lift, preplace, place must all have a
collision-free IK solution (MoveIt `/compute_ik`, held mesh attached from the lift on). Motions between them are
only planned when executed; a failed motion moves on to the next plan. Supersedes the order in D-020.

### D-024
**Place in the table rest pose.** The object goes down oriented as it stood on the table (a pose known to be
stable), turned about vertical only. A side grasp (approach > 30° from vertical) is turned so it points from the
robot towards the place point, i.e. into the shelf. The ROS `grasp_node` allows approaches up to 95° from vertical
(the playground keeps 50°): without side grasps the bottle can't go into the shelf at all.

### D-025
**Visual prompt + text check** (`detect_node`, `prompt: visual+text`). Measured on the live sim: the visual prompt
alone picks the right tomato can in 7/12 and 9/12 random scenes; the potted meat can looks alike. Each visual
detection is now checked by a text-prompt YOLOE with the target's name + a generic vocabulary (config.yaml
`detect.vocabulary`, no scene-specific names): score = visual + text(target) − text(best other name) on the same
mask. Over 24 scenes, threshold −0.25: 20 accepted, all right; the 4 rejected were all wrong detections. A rejection
fails the task honestly instead of picking the wrong object. Costs one more YOLOE pass (~40 ms).

### D-026
**Sim drives closer to a real Franka** (config.yaml `sim.arm_drive`, `sim.gripper`). franka.usd's arm damping
(80 N·m·s/deg vs stiffness 400 N·m/deg: 0.2 s time constant) made the arm trail a moving trajectory by 25–45 mm;
damping 10 → 0.2–7 mm. The finger drive was capped at 7.2 N and the bottle slipped out while the arm moved
(the task first reported it placed, ground truth had it on the table): 70 N (a real Franka Hand's continuous force)
and friction 1.0 on the finger pads.

### D-027
**Planning scene.** Fixed: table, pedestal, shelf panels + boards, floor (`planning_scene.py`, from config.yaml, as
D-010 allows). Per perception: the points standing on the table's footprint minus the target, as 2.5 cm voxel boxes
(`UpdateScene`) - without them the arm knocked the bottle over while moving between pregrasps of different plans.
The target's mesh joins while the arm travels to it (`AddTarget`) and leaves right before the fingers close around
it (`RemoveTarget`); from the lift it is attached to the hand. The task manager clears its own objects at start.

### D-028
**Symmetric objects.** config.yaml `objects.<name>.symmetry_axis` (the tomato can: z). FoundationPose's roll about
that axis is arbitrary (a live estimate came back 121° off, and the box grasp model then had no face pointing up:
0/16 grasps); `grasp_node` also tries the box model turned about the axis every 15°.

### D-029
**Python nodes on the `.venv`** (resolves [I-012](#i-012)). With Jazzy sourced, the `.venv`'s Python 3.12 imports
`rclpy` and the generated messages fine; no separate venv. Launch files run the nodes with `.venv/bin/python` as
prefix. The pybind11 module is built with the `.venv`'s pybind11 (I-022); `ros2/build.sh` does both.

### D-030
**Recovery never drops an object somewhere random.** After a failed pick-and-place the `Recover` tree puts a still
held object back where it was picked; if that fails, it keeps holding it and goes home. (First version opened the
gripper wherever the arm was, and the bottle fell into the shelf.) The BT also re-checks the grasp after the lift
and before releasing, so a dropped object is never reported as placed.

### D-031
**Upright first, on its side if that is what fits** (`vision/place.py rest_poses`, `place_node`, `SelectPlans`).
Keeping the table pose (D-024) stored the tomato can lying down, which a person wouldn't do. Now the object's stable
rest poses come from its mesh: each face of the convex hull whose support polygon holds the centre of mass, at
least 2.8 cm wide (narrower = a curved surface on one facet of the tessellated hull, e.g. a can on its side, which
rolls), never upside down. YCB meshes are modelled upright, so poses are ranked upright first, then on a side by
stability margin. Stage 5 returns placements per rest pose (its own footprint and headroom); `SelectPlans` tries them
in that order with the grasp from the table (same grasp in the object frame, the wrist does the turning), picking
the yaw that points a side grasp into the shelf, else trying quarter turns. Result: the lying can is stood upright;
a bottle too tall for the free space would be laid down. Possible later: a VLM prior for which poses are acceptable
for unfamiliar objects (e.g. keep containers of liquid upright).

### D-032
**`ros_cell.py --livestream`** enables `omni.kit.livestream.app` as in `scene.py`, so the live cell can be
watched in the Isaac Sim WebRTC Streaming Client while the whole ROS pipeline runs (I-013 turned out not to
apply). Foxglove (D-017) stays the way to see ROS data: detections, planned paths, TF.

## Issues

| ID | Date | Issue | Status |
|---|---|---|---|
| [I-001](#i-001) | 2026-09-29 | Random memory-corruption crashes in Python (possible bad RAM) | **Open** |
| [I-002](#i-002) | 2026-09-29 | Replicator hangs while WebRTC streaming is on | Workaround |
| [I-003](#i-003) | 2026-09-29 | Bundled streaming app config hangs at startup | Workaround |
| [I-004](#i-004) | 2026-09-29 | Isaac Sim 6.1 API moves: `core.api` deprecated, Franka example class not loaded | Resolved |
| [I-005](#i-005) | 2026-09-29 | Franka sagged to all-zero joints | Resolved |
| [I-006](#i-006) | 2026-09-29 | Scene colours washed out | Resolved |
| [I-007](#i-007) | 2026-09-29 | Livestream wrote `NvStreamer-*.etli` traces into the repo root | Resolved (unverified with a client) |
| [I-008](#i-008) | 2026-09-29 | NVIDIA driver 580 is below Isaac ROS 4.6+ requirements | Workaround |
| [I-009](#i-009) | 2026-09-29 | FoundationPose container (Python 3.8, no ROS) can't host a Jazzy node | Resolved by D-012 |
| [I-010](#i-010) | 2026-09-29 | Isaac ROS FoundationPose depth encoding not confirmed | Resolved |
| [I-011](#i-011) | 2026-09-29 | Known apt conflict installing Isaac ROS FoundationPose in its container | Resolved by D-022 |
| [I-012](#i-012) | 2026-09-29 | Python ROS nodes need system `rclpy` and `.venv` packages together | Resolved by D-029 |
| [I-013](#i-013) | 2026-09-29 | WebRTC livestream expected to conflict with ROS camera publishing | Resolved (no conflict, D-032) |
| [I-014](#i-014) | 2026-09-29 | Two tomato cans in the scene: "tomato can" prompt will match both | Resolved by D-021 |
| [I-015](#i-015) | 2026-09-29 | Sim depth is perfect and masks exact: pose results are a best case | Open |
| [I-016](#i-016) | 2026-09-29 | FoundationPose run-to-run spread on the same frame | Noted |
| [I-017](#i-017) | 2026-09-29 | Robot model differs from a newer real Franka (FR3) | Accepted (D-013) |
| [I-018](#i-018) | 2026-09-29 | Ground-truth extrinsics used as a stand-in for calibration | Open |
| [I-019](#i-019) | 2026-09-30 | Isaac reports finger velocity while the finger is blocked: gripper stall detection never fires | Workaround |
| [I-020](#i-020) | 2026-09-30 | Isaac 6.1 URDF export drops the fixed panda_link7 → panda_hand joint | Workaround |
| [I-021](#i-021) | 2026-09-30 | In-process ros2_control loads Isaac's bundled URDF plugins, whose libraries aren't on the loader path | Workaround |
| [I-022](#i-022) | 2026-09-30 | Ubuntu's pybind11 2.11 segfaults with numpy 2 | Resolved |
| [I-023](#i-023) | 2026-09-30 | ROS's launch_testing pytest plugin breaks the `.venv`'s pytest | Resolved |
| [I-024](#i-024) | 2026-09-30 | FoundationPose sometimes flips the mustard bottle 180° about its axis | Noted |
| [I-025](#i-025) | 2026-09-30 | Visual prompt confuses the tomato can with the potted meat can | Mitigated by D-025 |
| [I-026](#i-026) | 2026-09-30 | Restarting the sim resets sim time; running nodes keep stale TF | Workaround |
| [I-027](#i-027) | 2026-09-30 | Mustard side grasps blocked by neighbours at some yaws; no regrasp | Open |
| [I-028](#i-028) | 2026-09-30 | Small MoveIt/PhysX mismatches: SRDF virtual joint, start state past a joint limit, sim lag | Resolved |
| [I-029](#i-029) | 2026-09-30 | Held objects dropped in transit: gripper controller released the squeeze on a stall; shelf items not in MoveIt | Resolved |
| [I-030](#i-030) | 2026-09-30 | A second `ros_cell.py` started while the first still ran: two `/clock`s, TF_OLD_DATA flood | Resolved |
| [I-031](#i-031) | 2026-09-30 | Orphaned step-2 nodes from an earlier run: two `move_group`s, every execute aborted | Workaround |

### I-001
Python fails with errors ordinary code can't produce: `unknown opcode`, `invalid SRE code`, a bogus
`UnboundLocalError`, `SystemError: error return without exception set`, segfaults. Seen in the host
`.venv` and in ~5 of 10 FoundationPose container runs (2026-09-29). Retries pass. Both environments are
affected, which points to hardware (RAM). **Action:** run memtest86+ overnight before long-running ROS
processes (Phase 3).

### I-002
With `omni.kit.livestream.app` enabled, `rep.orchestrator.step()` and annotator `get_data()` never return.
**Workaround:** D-006 (streaming mode saves nothing).

### I-003
`apps/isaacsim.exp.full.streaming.kit` started, then hung before the scene was built (crash report after
~145 s). **Workaround:** default headless app + `enable_extension('omni.kit.livestream.app')`.

### I-004
6.1 moved `isaacsim.core.api` / `core.prims` to `extsDeprecated/` (they still import), and
`isaacsim.robot.manipulators.examples` (the `Franka` class) isn't loaded. **Fix:** reference
`/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` directly and wrap it in `SingleArticulation`.

### I-005
Setting joint positions alone left the drives targeting zero, so the arm sagged. **Fix:** also set the
drive targets (`apply_action(ArticulationAction(...))`).

### I-006
USD material colours are **linear**; sRGB values used as-is render much paler (0.42 → looks ~0.68), and
auto-exposure hides lighting changes. **Fix:** `sim.colors` is sRGB, converted in code (`** 2.2`).

### I-007
The stream's event tracing (`primaryStream.enableEventTracing = true` by default) wrote trace files to
the working directory. **Fix:** disabled via carb setting before enabling the extension, plus a
`.gitignore` backstop. Not yet confirmed with a client connected.

### I-008
Isaac ROS 4.6 and 5.0 need driver 595+ / CUDA 13.2+. **Workaround:** pin 4.5 (D-012). Upgrading the
driver would need a reboot and re-testing Isaac Sim and the FoundationPose container.

### I-009
The existing `foundationpose` image uses conda Python 3.8 with no ROS; Jazzy's `rclpy` is Python 3.12.
**Resolved by** D-012. The container remains useful for the offline stage 2 playground.

### I-010
Our topics publish depth as `16UC1` millimetres (like `realsense2_camera`). If Isaac ROS FoundationPose
expects `32FC1` metres, insert a converter node (e.g. `isaac_ros_depth_image_proc`). Verify in Phase 0/3.
**Resolved:** the 4.5 node accepts `32FC1` and `mono16`; `detect_node` sends FoundationPose `32FC1` metres
(`fp_depth_encoding`), the cameras keep publishing `16UC1` mm.

### I-011
Installing `ros-jazzy-isaac-ros-foundationpose` inside the Isaac ROS container fails on GL/EGL libraries
that the NVIDIA toolkit bind-mounts. **Known fix:** `dpkg-divert` the conflicting files first
([NVIDIA forum](https://forums.developer.nvidia.com/t/cant-install-ros-jazzy-isaac-ros-foundationpose-in-isaac-ros-environment/370217)).
**Resolved by** [D-022](#d-022): installed at image build time, where the bind mounts don't exist.

### I-012
`detect_node` / `grasp_node` / `place_node` need `rclpy` (system Python 3.12) and torch / ultralytics
(from `.venv`). **Plan:** a ROS-aware venv (e.g. `--system-site-packages` with Jazzy sourced), set up in
Phase 0.
**Resolved by** [D-029](#d-029).

### I-013
Isaac's ROS camera publishers are built on Replicator, the same path that hangs while streaming
(I-002). Expect to use Foxglove instead of the WebRTC stream while ROS runs (D-017).
**Resolved (2026-09-30):** it doesn't happen. With `ros_cell.py --livestream` both cameras still publish at
~9 Hz (10 Hz target) at real-time factor 1. The ROS camera path doesn't block the way `scene.py`'s
`orchestrator.step()` / `get_data()` do. See [D-032](#d-032). (`ros2 topic hz` on the raw images showed ~1 Hz: that
is the Python CLI deserialising 1280×720 images, not the sim. Count with `raw=True` instead.)

### I-014
The scene has a tomato can on the table (a demo target) **and** one on shelf level 1 (stage 5 obstacle).
Ground-truth masks are fine (shelf items are labelled `shelf_<name>`), but at runtime YOLOE will detect
both. **Plan:** `detect_node` keeps only detections inside the table workspace (stage 3's table plane +
bounds). Alternatively, swap the shelf item for a different object.
**Resolved by** [D-021](#d-021): the shelf item was swapped for a foam brick.

### I-015
All sim results so far use perfect depth and exact masks. **Plan (Phase 5):** RealSense-like depth
noise, YOLOE masks instead of ground truth, and randomised episodes.

### I-016
FoundationPose gives slightly different results on the same frame across runs (~0.4–0.9° on one mustard
frame). Compare methods over several frames and runs, not single numbers.

### I-017
Isaac's Franka is a Panda; newer real robots are FR3 (slightly different kinematics). **Accepted** by
D-013: not deploying to hardware.

### I-018
Until hand-eye calibration exists, the extrinsics fed to the pipeline come from the sim's ground truth
(`T_base_cam.txt`), marked as a stand-in in the calibration file. It must be replaced in Phase 5 by:
eye-to-hand calibration for the fixed camera, and eye-in-hand (panda_hand → wrist camera) for the wrist
camera. Score both against ground truth.

### I-019
While the fingers squeeze an object, Isaac reports `panda_finger_joint1` velocity ≈ −0.06 m/s at constant position,
so `GripperActionController`'s stall check never fires and the close never returns. **Workaround:** the BT's
`Gripper` node treats "finger position unchanged for 0.5 s (sim time)" as done and leaves the goal active (the
fingers keep squeezing). `wait_settled` also judges rest by position, not velocity.

### I-020
`isaacsim.ros2.control`'s URDF synthesis (6.1) leaves out the Franka's fixed `panda_link7 → panda_hand` joint: two
root links, and the controller_manager rejects the URDF. **Workaround:** `sim/ros_cell.py` re-adds missing fixed
joints, posed from the USD, and calls the backend's `setup_cm` itself. Also: `panda_finger_joint2` is exported as a
mimic joint (state only), so the hand controller commands `panda_finger_joint1` only.

### I-021
The in-process controller_manager's pluginlib also finds Isaac's bundled `sdformat_urdf` parser plugin, whose
`libtinyxml2.so.9` exists only in Isaac's `jazzy/lib`. The extension sets `LD_LIBRARY_PATH` at runtime, too late for
the loader. **Workaround:** `sim/ros_cell.py` re-executes itself once with Jazzy sourced and Isaac's ROS library
directories appended (marker variable, since `.bashrc` already sources Jazzy).

### I-022
`ppp_geometry_py` built against Ubuntu's pybind11 2.11 segfaults on the first call from the `.venv` (numpy 2.5).
**Fix:** build with the `.venv`'s pybind11 3 (supports numpy 1 and 2): `ros2/build.sh`.

### I-023
With `ros2/install` sourced, pytest autoloads ROS's `launch_testing` plugins, which fail with the `.venv`'s pytest.
**Fix:** `pytest.ini` disables them.

### I-024
Live Isaac ROS FoundationPose returned the mustard bottle rotated ~180° about its long axis in several estimates
(axis within 1.4°, translation ~1 mm); other runs were right (1.35°). The bottle is nearly front/back symmetric.
Harmless here: the grasp model is its bounding box and it is placed upright. `eval_node` reports the raw error.

### I-025
The visual prompt (a crop of the real BOP image) scores the lookalike potted meat can about as high as the tomato
can in sim renders. **Mitigated by** [D-025](#d-025). Better prompts (more views, sim-domain examples) remain open.

### I-026
Restarting `sim/ros_cell.py` restarts sim time at 0; Python nodes keep TF from the old run ("extrapolation into
the past"). **Workaround:** restart the ROS launch after restarting the sim (docs/ROS.md).

### I-027
The mustard bottle can only be grasped across its 58 mm side (the 95 mm side doesn't fit the hand), and only from
the side if it is to go into the shelf. At some random yaws those faces point at the sugar box / Rubik's cube, and
the hand (20 cm across) can't get in without hitting them: no plan, the task fails cleanly. **Plan:** regrasp
(set it down turned, grasp again), or push it clear first. 2026-09-30, 8 episodes: mustard 3/8, can 6/8.

### I-028
Integration details, all fixed: MoveIt's Panda SRDF has a floating virtual joint `world → panda_link0` (static TF
added in `robot.launch.py`; Cartesian paths failed without it); PhysX lets a joint pressed against its limit overshoot
by a hair and MoveIt refuses such a start state (the task clamps the start state into bounds); motions were planned
while the arm was still settling (`wait_settled` before each plan).

### I-029
Objects slipped out between the table and the shelf (the task noticed at the shelf: fingers closed on nothing).
Three causes, three fixes: `GripperActionController` sets the finger command to the current position when it
detects a stall, which takes the squeeze off (its stall detection fires only now and then in Isaac, I-019), so
`allow_stalling: false` and the goal keeps squeezing; the finger pads got rubber-like friction 2.0 (was 1.0); the
arm moves at 20 % speed while carrying. Also, the items on the shelf were real in the sim but unknown to MoveIt, so
motions into the shelf swept the held object through them: `place_node` now returns what isn't a support
(minus the boards' own edges, already modelled) and it goes into the planning scene as voxels (`shelf_clutter`).

### I-030
Restarting the sim with `--livestream` without stopping the first one left two sims running, both publishing
`/clock` (sim time jumping between ~53 s and ~1336 s) and running their own `controller_manager`. Every node
logged `TF_OLD_DATA`, and MoveIt and the task couldn't run. **Fix:** `ros_cell.py` now exits at start if
`/clock` already has a publisher. Restarting the sim still needs step 2 restarted too (I-026).

### I-031
The task scanned the shelf, then every move failed (`unknown goal response`, `Execute request aborted`,
`execution to home failed`, 0/2 placed). A second copy of step 2 (`robot`, `moveit` and `perception` launches,
reparented to init) was still running from an earlier session, along with two `gt_pose_stub.py` scratch
publishers. Two `move_group`s answered the same `/move_action` goal. **Workaround:** before step 2, check that
`ros2 node list | sort | uniq -d` prints nothing, and stop leftovers with `pgrep -af "ros2 launch|move_group"`.

| ID | Question | Notes |
|---|---|---|
| <a id="q-001"></a>Q-001 | Does stage 5 (multi-plane RANSAC for shelf supports) share the C++ RANSAC with stage 3? | **Answered by [D-019](#d-019):** yes, one library + pybind11 |
| <a id="q-002"></a>Q-002 | ROS plan needs stage 5 added | **Answered by [D-020](#d-020)** |

## Roadmap snapshot (2026-09-30)

| Phase | Status |
|---|---|
| 0 · Environment | Done: apt MoveIt 2 / ros2_control / BT.CPP / foxglove, Isaac ROS 4.5 image (D-022), `ros2/build.sh` |
| 1 · Sim ↔ ROS | Done: `sim/ros_cell.py` (I-020, I-021) |
| 2 · Motion | Done: MoveIt with the cell's boxes + perceived clutter (D-027) |
| 3 · Perception | Done: all nodes; C++ RANSAC matches numpy (D-019); FoundationPose live ~1 mm |
| 4 · Behavior tree | Done: look before you pick (D-023), rest poses (D-031); 8 episodes: can 6/8, mustard 3/8 |
| 5 · Robustness | Next: hand-eye calibration (I-018), regrasp (I-027), depth noise (I-015), better prompts (I-025) |

## Roadmap snapshot (2026-09-29, updated for D-019/D-020)

| Phase | Work | Done when |
|---|---|---|
| 0 · Environment | apt: MoveIt 2, controllers, vision_msgs, Panda description + MoveIt config, BT.CPP, foxglove_bridge. Isaac ROS 4.5 container + FoundationPose + engines. ROS-aware venv. `ros2/` workspace + `ppp_interfaces` | `colcon build` passes; FoundationPose container loads its engines |
| 1 · Sim ↔ ROS | `scene.py --ros`: cameras, `/clock`, ros2_control, `/sim/gt/*` | Point cloud aligns with the robot model in Foxglove; a CLI trajectory moves the arm |
| 2 · Motion | robot_state_publisher, extrinsics TF, MoveIt with table + shelf boxes, gripper | MoveIt reaches both targets and all 3 shelf levels |
| 3 · Perception | `ppp_geometry` (C++ RANSAC + pybind11), `detect_node`, 2× FoundationPose, C++ `spatial_node`, `grasp_node`, `place_node`, `eval_node` | Live pose error ≈ offline results; C++ RANSAC matches Python (D-019 tolerances) |
| 4 · Behavior tree | `task_manager` + trees, incl. "look at the shelf" → `GetPlacements` | Mustard → shelf, lying tomato can → shelf, confirmed by `eval_node` |
| 5 · Robustness | Hand-eye calibration (both cameras), depth noise, randomised episodes | Success rate over N episodes; calibration error vs ground truth |
