# Stage 2 — FoundationPose pose playground

## Commands

Cursor terminal on the workstation. The script re-runs itself inside the `foundationpose` container — no `docker exec` needed.

```bash
source .venv/bin/activate                                                        # once per terminal
python pipeline/interactive_pose.py                                              # defaults
python pipeline/interactive_pose.py --iterations 1 --tag it1                     # fewer refinement steps
python pipeline/interactive_pose.py --iterations 10 --tag it10                   # more refinement steps
python pipeline/interactive_pose.py --n-views 10 --inplane-step 120 --tag coarse # fewer hypotheses
python pipeline/interactive_pose.py --top 20                                     # print more hypotheses
python pipeline/interactive_pose.py --help
```
Prints a report, saves `output/pose/<tag>.{png,json}`, opens the image as a Cursor tab.

## Render the mesh

`pipeline/render_mesh.py`: the mesh as FoundationPose sees it (its own renderer). Runs itself in the container.

```bash
python pipeline/render_mesh.py                   # 4 views: texture | shape | vertices | triangles | triangles x5
python pipeline/render_mesh.py --views 8 --elev 60
python pipeline/render_mesh.py --pose play       # mesh at output/pose/play.json's pose, over the scene
```

| Mode | Output | Columns |
|---|---|---|
| views | `output/mesh/views.png` | texture (+ axes x red / y green / z blue) \| shape (grey, no texture) \| vertices (all 8,423, front + back) \| triangle edges facing the camera \| same, 5× zoom |
| `--pose TAG` | `output/mesh/pose_<TAG>.png` | scene \| texture over scene \| shape |

**What's in the mesh file** (`textured.obj`, YCB google_16k): 8,423 vertices + 16,384 triangles + one 4096×4096 texture.
Each triangle = 3 vertex indices (`mesh.faces`); edges are ~2.5 mm (median), irregular, denser where the surface curves.
The vertices are the *simplified surface*, not the raw scanner points — YCB ships the fused scan clouds separately.

## Data

Same scene as stage 1: `data/multi_object_scene/`. Mustard bottle only.

| Data | File |
|---|---|
| RGB image | `data/multi_object_scene/scene_rgb.png` |
| Depth | `data/multi_object_scene/scene_depth.png` (uint16 × 0.1 mm → metres) |
| Camera intrinsics `K` | `data/multi_object_scene/camera_K.txt` |
| Mask | YOLOE, visual prompt `refs/006_mustard_bottle.png` (stage 1 code + `config.yaml` `detect:`) → `output/pose/<tag>_mask.png` |
| Mesh | `assets/ycb/006_mustard_bottle/textured.obj` (metres) |
| **Pose** | `output/pose/<tag>.json` (`ob_in_cam`, 4×4) + overlay `output/pose/<tag>.png` |

## Model weights

In `third_party/FoundationPose/weights/` (downloaded during FoundationPose setup). Loading both: ~5.5 s per run.

| Folder | Size | Network | Job |
|---|---|---|---|
| `2023-10-28-18-33-37/` | 66 MB | refiner | nudges a pose hypothesis closer to the image |
| `2024-01-11-20-02-45/` | 182 MB | scorer | ranks all refined hypotheses |

Stage 2 also uses stage 1's YOLOE weights for the mask (see [STAGE1_DETECT.md](STAGE1_DETECT.md)).

## Knobs

| Knob | Flag | What it does | Measured (L40S) |
|---|---|---|---|
| `iterations` | `--iterations` (default `config.yaml` `pose.iterations` = 5) | Refinement passes on **every** hypothesis. ≥ 1 | 1 → 371 ms, 5 → 880 ms, 10 → 1505 ms |
| `n_views` | `--n-views` (40) | Viewpoints on a sphere, rounded **up** to 12 / 42 / 162 | 10 or 40 → 42 |
| `inplane_step` | `--inplane-step` (60) | Degrees between camera-roll steps per viewpoint | 60 → 6 rolls, 120 → 3 |
| `top` | `--top` (5) | Hypotheses printed | — |

| Setting | Hypotheses | Register time |
|---|---|---|
| default (42 views × 6 rolls) | 252 | 880 ms |
| `coarse` (42 × 3) | 126 | 433 ms |

Rule of thumb: ~250 ms fixed + ~125 ms per iteration (252 hypotheses).

## Outputs

| Output | Where |
|---|---|
| Mask px, hypothesis count, build + register time | terminal |
| Best pose `ob_in_cam` (4×4, metres) + distance moved from the start point | terminal |
| Top-N: score, rotation + translation vs #0 | terminal |
| Overlay: 3D box, axes (x red / y green / z blue), mask outline (cyan) | `output/pose/<tag>.png` |
| Settings + pose + top scores | `output/pose/<tag>.json` |

## How it works

`FoundationPose.register()` (`third_party/FoundationPose/estimater.py`):

```
mask + depth ─► start point (mask box centre, median depth)
252 rotations at that point ─► refiner × iterations ─► scorer ranks 252 ─► #0 = pose
```

| Step | What happens | Here |
|---|---|---|
| 1. Start point | Depth cleaned (erode + bilateral filter). One 3D point: mask box centre at median masked depth | on the bottle's front surface |
| 2. Hypotheses | Every grid rotation placed at that point. **No rotation guess** | 252 |
| 3. Refiner | Render mesh at each hypothesis, compare with RGB-D crop, correct rotation + translation. Repeat `iterations` times | moves the bottle 28.5 mm back to its centre |
| 4. Scorer | Rank all refined hypotheses; #0 wins | — |

The mask only sets the start point and crop — it doesn't define the pose.

**Reading the ranking** (`rot vs #0` = rotation difference from the winner):

| Iterations | Top 5 | Best score |
|---|---|---|
| 1 | ranks 2–3 = bottle **flipped ~180°** (66.91 vs 67.06) | 67.06 |
| 5 | all within 2° / 0.5 mm: converged | 62.22 |
| 10 | all within 0.2° | 62.56 |

- Same winning pose in all three (within 1 mm).
- Scores only rank within one run. **Not a confidence**; they shift with `iterations`.

## Things to try

| Try | Question |
|---|---|
| `--iterations` 1 / 2 / 3 / 5 | When do the flipped poses leave the top 5? |
| `--n-views 10 --inplane-step 120` | Same pose at half the time? |
| `--top 20` | How far down is the first flipped bottle at 5 iterations? |
| `detect.conf` / `detect.imgsz` in `config.yaml` | How much do mask and pose change? |

## Gotchas

| Symptom | Cause / fix |
|---|---|
| `--iterations 0` refused | FoundationPose crashes on 0 (`UnboundLocalError: trans_delta`) |
| Warp banner, torch warning, `num original candidates = 252` | container libraries; harmless |
| `free(): double free` on exit, or no output | container flake — re-run |
| `--help` printed twice | once on the host, once in the container; harmless |
| Other datasets | `--data DIR` (FoundationPose layout, e.g. Isaac Sim frames — [SIM.md](SIM.md)) |
