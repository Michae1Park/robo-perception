#!/usr/bin/env bash
# Run stages 1-4 on the mustard0 sequence. Run from the repo root: pipeline/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/conda/envs/my/bin/python3   # the container's env (FoundationPose CUDA extensions)
HOST="env -u PYTHONPATH .venv/bin/python"

docker start foundationpose >/dev/null
$HOST pipeline/01_detect.py                                                   # 1 detect  (host)
docker exec foundationpose bash -lc "cd $PWD && $PY pipeline/02_pose.py"     # 2 pose    (container)
$HOST pipeline/03_spatial.py                                                  # 3 spatial (host)
$HOST pipeline/04_grasp.py                                                    # 4 grasp   (host)
docker exec foundationpose chown -R "$(id -u):$(id -g)" "$PWD/output" 2>/dev/null || true   # container runs as root
echo "results in output/"
