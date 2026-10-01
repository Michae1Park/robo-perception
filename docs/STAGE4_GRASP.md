# Stage 4 — Grasp playground

## Commands

Cursor terminal on the workstation, host `.venv` (numpy + OpenCV only). Run stage 2 first (`02_pose.py`) — this reads its pose.

```bash
source .venv/bin/activate                                 # once per terminal
python pipeline/04_grasp.py                               # config.yaml defaults
python pipeline/04_grasp.py --max-tilt 100 --tag side     # allow side grasps
python pipeline/04_grasp.py --max-opening 0.11 --tag wide # a wider gripper
python pipeline/04_grasp.py --pose it1 --tag it1          # grasps from another stage 2 run
python pipeline/04_grasp.py --help
```
Prints a candidate table, saves `output/grasp/<tag>.png` (2 panels), opens it as a Cursor tab.

## Data

Same scene as stages 1–3: `data/multi_object_scene/`. Mustard bottle only.

| Data | File / source |
|---|---|
| RGB image | `data/multi_object_scene/scene_rgb.png` (only for drawing) |
| Depth + `K` | `scene_depth.png`, `camera_K.txt` → stage 3 (table plane, scene points) |
| Pose | stage 2's `output/pose/<TAG>.json` (`--pose`, default `play`) |
| Mesh | `assets/ycb/006_mustard_bottle/textured.obj` (from the pose json) → its bounding box |
| **Table frame + obstacles** | stage 3, recomputed here (`analyze_scene`, ~0.6 s) |
| **Grasps** | terminal table + `output/grasp/<tag>.png` |

## Knobs

Defaults: `config.yaml` → `gripper:` and `grasp:`. Flags override per run. This playground draws the stock Franka fingers;
the robot cell's are 5 cm longer (`gripper.finger_extension`, [SIM.md](SIM.md)).

| Knob | Flag | What it does | Default |
|---|---|---|---|
| `max_opening` | `--max-opening` | Widest the fingers open; wider object sides get no candidates | 0.08 m |
| `finger_depth` | `--finger-depth` | How far the pads' centre (the TCP) goes past the grasped face | 0.045 m |
| `width_margin` | `--width-margin` | Free opening needed beyond the object's width | 0.008 m |
| `max_approach_tilt_deg` | `--max-tilt` | Max angle between the approach and straight down | 50° |
| `table_clearance` | `--clearance` | Fingertips must stay this far above the RANSAC table | 0.012 m |
| (fixed) | — | ≥ 10 obstacle points inside the gripper = collision | 10 |

## Outputs

| Output | Where |
|---|---|
| Scene points split: table / bottle / obstacles | terminal |
| Mesh box size, which closing directions fit the gripper | terminal |
| Every candidate: face, width, tilt, lowest point above the table, obstacle hits, result | terminal |
| A: all candidates as grippers (magenta best, cyan ok, red rejected), numbered | `output/grasp/<tag>.png` left |
| B: best grasp + the mesh box it came from (green) + object and gripper axes | right |
| Object + best gripper pose in camera, table and object frames (xyz mm, rpy °) | terminal |

## How it works

```
mesh box ─► candidates (faces × closing directions) ─► place with stage 2 pose ─► filter with stage 3 geometry ─► best
```

| Step | What happens | Here |
|---|---|---|
| 1. Box | Axis-aligned bounding box of the mesh, in the object's own frame | 96 × 58 × 191 mm |
| 2. Candidates | For each of the 6 faces: approach straight into it, fingers close along one of the other 2 axes (both signs). Axes wider than `max_opening − width_margin` are skipped | only y (58 mm) fits → 4 faces × 2 = **8** |
| 3. Place | Candidate (object frame) × stage 2 pose × stage 3 table frame → gripper pose on the table | — |
| 4. Filter | Three checks, in order (below). No IK: there is no robot | **2 / 8** pass |
| 5. Rank | Least tilted first | top grasp, 2° from vertical |

**The filter** — all geometry from stage 3's RANSAC:

| Check | Rejects when | Uses |
|---|---|---|
| Tilt | approach > `max_tilt` from straight down | table normal (RANSAC plane) |
| Table | TCP or a fingertip < `table_clearance` above the table | table height (RANSAC plane) |
| Collision | ≥ 10 obstacle points inside the fingers (whole open→closed stroke) or the palm | scene points that RANSAC didn't call table, minus the bottle's own points (its box + 1 cm) |

**Candidate table** (default run):

| # | Face | Tilt | Lowest point | Hits | Result |
|---|---|---|---|---|---|
| 0–1 | `+z` (top) | 2° | 142 mm | 0 | **best** |
| 2–5 | `±x` (sides) | 88–92° | 91 mm | 0 | too tilted (pass with `--max-tilt 100`) |
| 6–7 | `−z` (bottom) | 178° | 40 mm | 0 | too tilted |

- **Pairs.** `+y` / `−y` are the same grasp with the hand turned 180° — identical fingers, drawn on top of each other.
- **Face names.** `+z/+y` = approach through the +z face (the cap), fingers close along ±y. The face letters are the *mesh's* axes (panel B's axes), not the table's.
- **Lowest point** 142 mm for the top grasp: bottle top ~188 mm − 45 mm finger depth.

## Poses & frames

Every pose is a 4×4 transform `T_a_b` = frame b expressed in frame a. The chain:

```
T_cam_obj  (stage 2, FoundationPose)      T_table_cam (stage 3, RANSAC)      T_obj_tcp (stage 4 candidate)
T_table_obj = T_table_cam @ T_cam_obj      T_table_grip = T_table_obj @ T_obj_tcp @ T_tcp_grip      T_cam_grip = inv(T_table_cam) @ T_table_grip
```

| Frame | Origin | Axes |
|---|---|---|
| camera | camera centre | x right, y down, z forward (OpenCV) |
| table | a point on the RANSAC plane | z up (table normal), x = image right, y = away from the camera |
| object | mesh origin = centre of its box | the mesh's own axes (z = up the bottle) |
| gripper | between the fingers, halfway up (22.5 mm back from the tips) | z = approach, y = finger closing, x = z × y |

Printed for the object and the best grasp (default run):

| Pose | xyz (mm) | roll, pitch, yaw (°) | Reads as |
|---|---|---|---|
| object in camera | 164, −23, 769 | −151, −59, −129 | stage 2's output, 77 cm in front of the camera |
| object in table | 129, 680, 92 | 1, 2, 110 | upright (roll/pitch ≈ 0), centre 92 mm up, turned 110° |
| gripper in table | 129, 683, 165 | 179, −2, −71 | pointing straight down (roll 180°), jaw turned with the bottle (110° − 180°) |
| gripper in camera | 157, −88, 735 | −29, 59, 51 | what a robot would get after hand-eye calibration |
| gripper in object | 0, 0, 73 | 180, 0, 180 | the candidate itself: on the cap axis, 23 mm below the top (TCP 45 mm below, frame 22.5 mm above it) |

rpy = `R = Rz(yaw) · Ry(pitch) · Rx(roll)`. The filter's table + collision checks use the fingertips (the TCP), not this frame. The candidate table's last column is each grasp's TCP xyz in the table frame.

## Things to try

| Try | Question |
|---|---|
| `--max-tilt 100` | Side grasps pass. Which one is closest to the soup can? How many hits? |
| `--max-opening 0.11` | The 96 mm side fits (96 + 8 mm margin): 16 candidates, 4 pass. Tilt ties at 2°, so the 96 mm grip ranks first — is that the one you'd pick? |
| `--finger-depth 0.09` | The top grasp's lowest point drops 142 → 97 mm. Why does it stop there? (depth is capped at half the box) |
| `--clearance 0.2` | Everything fails the table check. Why the top grasp too? |
| `--pose it1` | Same grasps from a 1-iteration pose? |

## Gotchas

- **No IK / reachability.** There's no robot here; a "feasible" grasp only means the geometry works.
- **Final pose only.** The collision check looks at where the gripper ends up, not the path it takes getting there.
- **Only visible surfaces are obstacles.** Depth sees the front of each object; the back is unknown (grey in stage 3) and never collides.
- **The bottle's 1 cm margin also hides neighbours.** Obstacle points within 1 cm of the bottle's box count as "bottle". That is why the soup can, right behind it, produces almost no hits.
- **Box, not shape.** Candidates come from the bounding box, so a grasp on the bottle's narrow neck is never proposed.
