# prompt-pick-place — vision pipeline

Show it **one example image** (or type a phrase) → it finds the object, estimates its 6-DoF pose,
maps the table, and proposes grasps. Real RGB-D images, no robot, no simulator, no ROS.

![Pipeline on one frame: detection, pose, occupancy, grasps](docs/pipeline_demo/pipeline_overview.png)

## The pipeline

| # | Stage | Script | Runs in | Model / method |
|---|---|---|---|---|
| 1 | Detect + mask | [`pipeline/01_detect.py`](pipeline/01_detect.py) | container | YOLOE-seg, visual prompt (example image + box) |
| 2 | 6-DoF pose | [`pipeline/02_pose.py`](pipeline/02_pose.py) | container | FoundationPose (uses stage 1's mask) |
| 3 | Table + free space | [`pipeline/03_spatial.py`](pipeline/03_spatial.py) | host | RANSAC plane + occupancy grid |
| 4 | Grasp candidates | [`pipeline/04_grasp.py`](pipeline/04_grasp.py) | host | Box-face parallel-jaw grasps + geometric filter |

- Stages pass files through `output/` (each script reads the previous one's output) — run them in order.
- **container** = the `foundationpose` Docker container (GPU, torch, ultralytics). **host** = plain numpy, `.venv`.

## Run

```bash
docker start foundationpose
pipeline/run_all.sh                 # stages 1-4 on the mustard0 sequence -> output/
```

| Other entry points | What it does | Runs in |
|---|---|---|
| [`pipeline/interactive_detect.py`](pipeline/interactive_detect.py) | Stage 1 only: your own visual prompts on a 5-object scene | container |
| [`pipeline/talk_and_pick.py`](pipeline/talk_and_pick.py) | Text command → stages 1-4 → 4-panel composite | host (calls the container) |
| `pytest tests` | Unit tests for `vision/` (no GPU) | host |

```bash
# in the container:  docker exec -it foundationpose bash -lc 'cd <repo> && /opt/conda/envs/my/bin/python3 <script>'
pipeline/interactive_detect.py --object 005_tomato_soup_can --object 006_mustard_bottle
pipeline/interactive_detect.py --prompt my_obj ref.png 120,80,340,410 --scene scene.png   # your own images

# on the host:
.venv/bin/python pipeline/talk_and_pick.py --command "pick up mustard sauce"
.venv/bin/python pipeline/talk_and_pick.py --repl
```

## Where things are

| Path | What |
|---|---|
| `vision/` | **The algorithms** (import these when experimenting) |
| `vision/detect.py` | Stage 1: YOLOE visual + text prompting, `detect()` |
| `vision/pose.py` | Stage 2: FoundationPose wrapper (container only) |
| `vision/spatial.py` | Stage 3: depth → points, RANSAC plane, occupancy grid, `analyze_scene()` |
| `vision/grasp.py` | Stage 4: candidate generation + filter |
| `vision/viz.py`, `transforms.py` | Overlays / plots, 4×4 pose math |
| `config.yaml` | **Every tunable parameter** (conf, RANSAC, grid, gripper, grasp filter) |
| `pipeline/` | Runnable scripts (above) |
| `scripts/` | `prepare_ycb.py`, `yoloe_trt.py`, `build_fp_engines.sh` (see [TensorRT](#tensorrt-not-yet-run)) |
| `tests/` | Unit tests |
| `docs/pipeline_demo/` | Committed result screenshots (shown in this README) |
| `output/` | Results of your runs (gitignored) |

## Data

| Data | Location | Used by | How to get it |
|---|---|---|---|
| mustard0 RGB-D sequence + mesh | `third_party/FoundationPose/demo_data/mustard0/` | stages 1-4 (`run_all.sh`) | FoundationPose setup (below) |
| Multi-object scene (BOP YCB-Video, 5 objects) + reference crops | `data/multi_object_scene/` | `interactive_detect`, `talk_and_pick` | tracked in git |
| YCB meshes (6 objects) | `assets/ycb/<obj>/textured.obj` | `talk_and_pick` (pose + grasp) | `python scripts/prepare_ycb.py` |
| YOLOE weights | `yoloe-11l-seg.pt` in the repo root | stage 1 | auto-downloaded on first run |
| FoundationPose weights | `third_party/FoundationPose/weights/` | stage 2 | FoundationPose setup |
| `assets/prompts/`, `assets/ycb/*/*.usd` | `assets/` | **nothing** (leftovers from the removed Isaac Sim work) | safe to delete |

## Results

| Stage | Input | Result |
|---|---|---|
| 1 | mustard0 frame 736, prompt from frame 0 | `006_mustard_bottle` score **0.51** ([image](docs/pipeline_demo/01_detection.png)) |
| 2 | stage 1 mask | 6-DoF pose overlay ([image](docs/pipeline_demo/02_pose_overlay.png)) |
| 3 | depth | plane 88k / 132k inliers; grid 140×50 cells, 4632 free / 322 occupied / 2046 unknown ([image](docs/pipeline_demo/03_occupancy.png)) |
| 4 | pose + mesh | **2 / 8** candidates feasible; best is 8.5° from vertical ([image](docs/pipeline_demo/04_grasp_candidates.png)) |
| 1 (multi-object) | tomato soup can / mustard bottle prompts | 0.71 / 0.55, no false positives on cracker box, banana, drill ([image](docs/pipeline_demo/multi_object/overview.png)) |

`talk_and_pick.py` on the multi-object scene:

| Command | Result |
|---|---|
| `pick up mustard sauce` | detected 0.68 → pose → **2/8** grasps feasible ([image](docs/pipeline_demo/multi_object/full_pipeline_mustard_sauce.png)) |
| `grab the tomato soup can` | detected 0.40 → pose → **4/16** feasible ([image](docs/pipeline_demo/multi_object/full_pipeline_tomato_soup_can.png)) |
| `pick up the cheez-it box` | detected 0.75 → stops (no mesh for it) ([image](docs/pipeline_demo/multi_object/full_pipeline_cheez_it_box.png)) |

## Things to know

- **No real world frame.** Stage 3 builds one from the table itself: table normal = +Z, table height = 0.
- **Stage 4 mesh = stage 2 mesh.** On mustard0 it uses FoundationPose's own mesh, not `assets/ycb`, so pose and geometry agree.
- **Stage 4 has no IK/reachability check** — no robot here; it only filters by approach tilt and table clearance.
- **Bottle on its side → 0/8 grasps.** Correct: no top-down grasp exists. `run_all.sh` uses the final (upright) frame.
- **Text prompts are wording-sensitive.** `"mustard bottle"` scores 0.68, `"mustard sauce"` 0.008. `talk_and_pick` swaps in a canonical phrase for the 6 known objects.
- **`talk_and_pick` is not an LLM.** Prefix stripping + keyword match to a mesh; the detection itself is YOLOE.
- **Container teardown crash.** The container's Python occasionally prints `free(): double free` on exit (after the results are written). Re-run.

## Setup (one time)

Tested: Ubuntu 24.04, NVIDIA L40S, driver ≥ 580, Docker.

**Host env** (stages 3-4, tests):
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**FoundationPose container** (stages 1-2; torch + ultralytics live in its `my` conda env):
- `git clone https://github.com/NVlabs/FoundationPose third_party/FoundationPose`
- Follow its README for Docker. On Ada-class GPUs (L40S) use the image `shingarey/foundationpose_custom_cuda121:latest`.
- Inside the container run its `build_all.sh`, then download its weights and `demo_data` (has `mustard0`).
- Name the container `foundationpose`. It must mount **`/home:/home`** so the repo has the same path inside and outside.
- `pip install ultralytics` inside the `my` env (needs `>=8.3.150`).

## TensorRT (not yet run)

Written but never executed — this is the next piece of work.

| Script | Purpose |
|---|---|
| [`scripts/yoloe_trt.py`](scripts/yoloe_trt.py) | `export` the visual-prompt YOLOE to a TensorRT engine; `bench` PyTorch FP32 / FP16 / TensorRT latency |
| [`scripts/build_fp_engines.sh`](scripts/build_fp_engines.sh) | Build TensorRT engines for FoundationPose's refiner + scorer from NVIDIA's ONNX models |

| Backend (L40S) | inference p50 | end-to-end p50 |
|---|---|---|
| PyTorch FP32 | – | – |
| PyTorch FP16 | – | – |
| TensorRT FP16 | – | – |

## History

Earlier versions had a ROS 2 + Isaac Sim + MoveIt pick-and-place stack. It was removed to focus on the
vision pipeline; it is still in git at commit `255a4aa` (`git checkout 255a4aa -- ros2 isaac_sim docker docs/SETUP.md`).
