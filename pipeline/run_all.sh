#!/usr/bin/env bash
# Run stages 1-4 on the mustard0 sequence. Run from the repo root: pipeline/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/conda/envs/my/bin/python3   # the container's env (torch, ultralytics, cv2)

docker start foundationpose >/dev/null
docker exec foundationpose bash -lc "cd $PWD && $PY pipeline/01_detect.py"   # 1 detect  (container)
docker exec foundationpose bash -lc "cd $PWD && $PY pipeline/02_pose.py"     # 2 pose    (container)
env -u PYTHONPATH .venv/bin/python pipeline/03_spatial.py                                      # 3 spatial (host)
env -u PYTHONPATH .venv/bin/python pipeline/04_grasp.py                                        # 4 grasp   (host)
echo "results in output/"
