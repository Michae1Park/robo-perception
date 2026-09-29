# Stage 3 — Table plane + occupancy playground

## Commands

Cursor terminal on the workstation, host `.venv` (numpy + OpenCV only, no GPU, no container).

```bash
source .venv/bin/activate                                                       # once per terminal
python pipeline/interactive_spatial.py                                          # config.yaml defaults
python pipeline/interactive_spatial.py --resolution 0.02 --tag coarse           # 2 cm cells
python pipeline/interactive_spatial.py --height-thresh 0.03 --tag thresh3cm     # ignore anything < 3 cm tall
python pipeline/interactive_spatial.py --ransac-dist 0.002 --tag tight          # stricter "on the table"
python pipeline/interactive_spatial.py --ransac-iters 5 --tag few               # fewer RANSAC tries
python pipeline/interactive_spatial.py --help
```
Prints a report, saves `output/spatial/<tag>.png` (4 panels), opens it as a Cursor tab.

## Data

Same scene as stages 1 and 2: `data/multi_object_scene/`.

| Data | File |
|---|---|
| RGB image | `data/multi_object_scene/scene_rgb.png` (only for drawing) |
| Depth | `data/multi_object_scene/scene_depth.png` (uint16 × 0.1 mm → metres) |
| Camera intrinsics `K` | `data/multi_object_scene/camera_K.txt` |
| Mustard pose (optional) | stage 2's `output/pose/<TAG>.json` (`--pose`, default `play`) → where the bottle stands on the table |
| **Table plane + frame** | printed; `T_world_cam` returned by `vision.spatial.analyze_scene()` |
| **Occupancy grid** | panels C and D of `output/spatial/<tag>.png` |

No mask, no mesh, no network: stage 3 is pure geometry on the depth image.

## Knobs

Defaults: `config.yaml` → `spatial:`. Flags override per run.

| Knob | Flag | What it does | Try |
|---|---|---|---|
| `ransac_dist` | `--ransac-dist` | A point within this distance of the plane counts as "table" | 0.002 → 44% inliers, 0.006 → 52% |
| `ransac_iters` | `--ransac-iters` | Random 3-point planes tried | 5 still finds the table here |
| `resolution` | `--resolution` | Grid cell size (m) | 0.01 → 51 × 54 cells, 0.02 → 26 × 27 |
| `height_thresh` | `--height-thresh` | Taller than this above the table = obstacle | 0.012 → 0.03 |
| `max_height` | `--max-height` | Ignore points higher than this (ceiling, far walls) | 0.6 |
| `min_inliers` | (config only) | Fewer table points than this = "no table" error | 500 |

## Outputs

| Output | Where |
|---|---|
| Point count, time, plane equation, inlier %, camera height + tilt | terminal |
| Grid size + free / occupied / unknown counts, max clearance | terminal |
| Bottle position on the table (from stage 2), height of its centre, tilt from vertical | terminal |
| A: RANSAC inliers tinted green | `output/spatial/<tag>.png` top left |
| B: height above the table (turbo colour map, 0–25 cm; table = grey, no depth = black) | top right |
| C: grid cells drawn on the table in the image, + at the bottle's centre | bottom left |
| D: grid from above (far side at the top), + at the bottle | bottom right |

## How it works

`vision.spatial.analyze_scene()`:

```
depth ─► 3D points ─► RANSAC plane ─► table frame ─► occupancy grid ─► clearance
```

| Step | What happens | Here |
|---|---|---|
| 1. Points | Every depth pixel → 3D point: `x = (u − cx)·z / fx`, `y = (v − cy)·z / fy` | 240,671 points; 66,529 px have no depth |
| 2. RANSAC | Repeat `ransac_iters`×: pick 3 random points → plane → count points within `ransac_dist`. Keep the best, refit it to all its inliers | table = 52% of all points |
| 3. Table frame | New frame: table normal = +Z, table height = 0, X = image right, Y = away from the camera. No robot calibration needed | camera 0.464 m above the table, looking 29° down |
| 4. Grid | Table area (2nd–98th percentile of table points) cut into `resolution` cells. Each cell gets one label | 51 × 54 cells of 1 cm |
| 5. Clearance | For every free cell: distance to the nearest non-free cell | largest free circle: 95 mm radius |

**The cell labels:**

| Label | Rule | Colour |
|---|---|---|
| free | only table-height points (`\|h\| < height_thresh`) | green |
| occupied | ≥ 3 points between `height_thresh` and `max_height` above the table | red |
| unknown | no points at all: hidden behind an object (depth shadow) or no depth return | grey |

- **Occupied is only the part the camera sees.** The grid marks where object *points* land when dropped straight down onto the table. The back of an object is never observed, so the grey shadow behind it is unknown, not occupied.
- **Checking against stage 2.** The bottle's centre comes out **92 mm** above the table (mesh half-height: 96 mm) and **2°** from vertical, so the pose and the plane agree.

**Time** (~0.65 s total on the CPU): RANSAC (300 planes × 240k points) is most of it; clearance ~0.25 s; points and grid ~10 ms each.

## Things to try

| Try | Question |
|---|---|
| `--ransac-dist` 0.002 / 0.006 / 0.02 | How noisy is the depth? When do the banana and drill start counting as "table"? |
| `--ransac-iters` 1 / 5 / 300 | How many tries does it really need? (Hint: the table is half the points.) |
| `--height-thresh` 0.005 / 0.03 / 0.1 | Which objects disappear from the grid first? |
| `--resolution` 0.005 / 0.02 / 0.05 | Detail vs. how blocky the free space gets |
| Panel B vs panel D | Why is the cracker box's footprint a thin bar, not a rectangle? |

## Gotchas

- The table frame is built from the table itself. X/Y has no meaning beyond "image right" and "away from the camera"; there is no robot base.
- `max_height` 0.6 m keeps the background above the table out of the grid; points *below* the table (the floor past its edge) are neither table nor obstacle.
- The zone only covers the table points seen. Objects standing off the table's visible area are outside the grid.
