# Stage 5 — Shelf placement playground

## Commands

Cursor terminal on the workstation, host `.venv` (numpy + OpenCV only). Needs `data/sim/shelf/` — make it once with the sim:

```bash
.venv-sim/bin/python sim/scene.py                                   # Isaac Sim: writes data/sim/shelf/ (~15 s)
source .venv/bin/activate                                           # once per terminal
python pipeline/05_place.py                                         # where does the mustard bottle fit?
python pipeline/05_place.py --object 005_tomato_soup_can --tag soup # a smaller object
python pipeline/05_place.py --hand-clearance 0.15 --tag bighand     # a bulkier hand above the object
python pipeline/05_place.py --margin 0.04 --tag margin              # keep 4 cm from everything
python pipeline/05_place.py --help
```
Prints supports + candidates, saves `output/place/<tag>.png` (3 panels), opens it as a Cursor tab.

## Data

The shelf in the Isaac Sim cell ([SIM.md](SIM.md)), seen by a **wrist camera**: the arm moves to a "look at the shelf" pose and the camera on its hand takes the picture.

| Data | File |
|---|---|
| RGB image | `data/sim/shelf/rgb/000000.png` (only for drawing) |
| Depth | `data/sim/shelf/depth/000000.png` (uint16 mm) |
| Camera intrinsics `K` | `data/sim/shelf/cam_K.txt` |
| Camera pose in the robot base `T_base_cam` | `data/sim/shelf/T_base_cam.txt` — on a real robot: joint angles (forward kinematics) + hand-eye calibration. **Gives "up"** |
| Object to place | `assets/ycb/<--object>/textured.obj` → footprint radius + height (placed upright) |
| Ground truth (only for checking) | `data/sim/shelf/shelf.json`: board heights −0.35 / 0.05 / 0.45 / 0.95 / 1.30 m |
| **Supports, free space, candidates** | terminal + `output/place/<tag>.png` |

No shelf model is used: every horizontal surface the camera sees is a possible place to put things.

## Knobs

Defaults: `config.yaml` → `place:`. Flags override per run.

| Knob | Flag | What it does | Default |
|---|---|---|---|
| `max_tilt_deg` | — | A surface is "horizontal" if its normal is within this of up | 10° |
| `ransac_dist` | `--ransac-dist` | Point-to-plane distance that counts as "on the surface" | 8 mm |
| `min_inliers` / `min_area` | — | Smaller planes / pieces are ignored | 800 pts / 0.01 m² |
| `resolution` | `--resolution` | Grid cell size | 1 cm |
| `height_thresh` | `--height-thresh` | Higher than this above a support = obstacle | 12 mm |
| `margin` | `--margin` | Free space kept around the object's footprint | 1 cm |
| `hand_clearance` | `--hand-clearance` | Space needed above the object for the hand while setting it down | 8 cm |
| `max_reach` | `--max-reach` | Place point must be this close to the Franka shoulder (0, 0, 0.333) | 0.85 m |
| `per_support` | — | Best candidates kept per support | 3 |

## Outputs

| Output | Where |
|---|---|
| Points, planes → supports, time | terminal |
| Object footprint radius, height, headroom it needs | terminal |
| Per support: height (+ nearest ground-truth board), area, free / occupied / unknown cells, headroom, largest free circle, candidates (or "out of reach") | terminal |
| Per candidate: place point in the **base** frame, clearance, headroom, reach, same point in the **camera** frame | terminal |
| A: every support tinted + its height | `output/place/<tag>.png` top left |
| B: free / occupied / unknown drawn on each reachable support + candidate footprints (magenta = best) | top right |
| C: each reachable support from above, oriented like the camera view (back of the shelf at the top) | bottom |

## How it works

```
depth + up ─► points + normals ─► horizontal points ─► sequential RANSAC ─► planes ─► split into pieces (supports)
          ─► per support: ceiling, free / occupied / unknown, clearance ─► slide the object's footprint ─► candidates
```

| Step | What happens | Here |
|---|---|---|
| 1. Points + up | Depth → 3D points, moved into the robot base frame (z = up) with `T_base_cam` | 307k points |
| 2. Normals | Surface direction at each pixel from its neighbours (±3 px). Keep points facing up (within 10°) | 250k |
| 3. Planes | RANSAC on those points: biggest horizontal plane, remove it, repeat (sequential RANSAC) | 5 planes |
| 4. Supports | Each plane cut into connected pieces on a 1 cm grid: one shelf level, the floor... | 8 supports |
| 5. Ceiling | The lowest support above that overlaps this one = its ceiling. Headroom = ceiling − height | 400 mm on the −0.35 and 0.05 boards; 0.45 board: open (the board above it is out of view) |
| 6. Cells | Like stage 3: **free** (only support points), **occupied** (≥ 3 points between the support and its ceiling), **unknown** (not seen) | — |
| 7. Candidates | Clearance = distance to the nearest non-free cell. A cell fits if clearance ≥ footprint radius and headroom ≥ object height + hand. Best first, no overlaps | 6 |

**Result for the mustard bottle** (radius 59 mm incl. margin, 191 mm tall → needs 271 mm headroom):

| Support | Height (GT) | Headroom | Candidates | Note |
|---|---|---|---|---|
| S0–S4 | −0.75 (floor) | open | out of reach | seen past and under the pantry |
| S5 | −0.349 (−0.35) | 400 mm | 0 | lowest board, mostly hidden under the 0.05 board: no spot with enough clearance |
| S6 | 0.050 (0.05) | 400 mm | 3 | around the sugar box; the back is hidden under the 0.45 board |
| S7 | 0.450 (0.45) | open | 3 | either side of the Rubik's cube and foam brick: most room (145 mm) → **best** |

All three board heights match the ground truth to the millimetre. The 0.45 board shows as "open" because the board
above it (0.95 m) is out of the wrist camera's view; MoveIt still has it, so a placement that needs more than its
0.48 m is rejected when the motion is planned.

## Things to try

| Try | Question |
|---|---|
| `--object 005_tomato_soup_can` | Smaller and shorter: where does it fit that the bottle doesn't? |
| `--hand-clearance 0.15` | Only the 0.45 board is left. How much hand room do the lower levels really have? |
| `--margin 0.04` | Which levels are too crowded now? |
| `--resolution 0.02` | Coarser grid: do candidates move? |
| Panel C of the bottom level | Why is the back half missing? (Where could the camera go to see it?) |

## Gotchas

- **"Up" comes from the robot, not the image.** `T_base_cam` from the arm's joint angles tells which way gravity points; without it, you'd infer up from the scene (e.g. the direction most surfaces share).
- **Only what the camera sees can be offered.** The back of the bottom board is hidden under the top one; it never becomes a candidate. More views (move the wrist camera) would fix that.
- **Headroom is measured to the top of the board above.** We only see board tops, so headroom includes the board's own thickness (400 mm measured, 380 mm real). Keep `hand_clearance` generous.
- **No IK / path check.** "Within reach" is a distance, not a motion plan; placing on the bottom board means reaching in under the top one.
- **Upright only, round footprint.** The playground models the object as a cylinder around its upright footprint, so any
  yaw fits. The ROS `place_node` also offers the other stable rest poses of the mesh ([D-031](DECISIONS.md#d-031)).
