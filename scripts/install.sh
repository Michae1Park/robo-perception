#!/bin/bash
# Install everything the project needs on Ubuntu 24.04 (x86_64, NVIDIA RTX GPU). Idempotent: finished steps are skipped,
# so re-run it after a failure or a reboot. Uses sudo for apt and Docker.
#
#   scripts/install.sh                 # every step below, in order
#   scripts/install.sh venv build      # only these steps
#
# Steps:
#   driver   NVIDIA driver 580 (open kernel modules); needs a reboot, then run the script again
#   apt      build tools, git-lfs, Python venv
#   ros      ROS 2 Jazzy + MoveIt 2, pick_ik, ros2_control, BehaviorTree.CPP, vision_msgs, foxglove_bridge
#   docker   Docker Engine + NVIDIA Container Toolkit (FoundationPose runs in a container)
#   venv     .venv: torch (CUDA 13.0), ultralytics (YOLOE), numpy, OpenCV, pybind11, uv
#   sim      .venv-sim: Isaac Sim 6.1 (pip, ~26 GB). Installing it accepts the NVIDIA Omniverse EULA
#   assets   YCB meshes (assets/ycb/) and YOLOE weights (models/)
#   fp       FoundationPose image ppp-isaac-ros:4.5 (~100 GB with its base, ~30 min), ONNX models, TensorRT engines
#   build    ROS workspace (ros2/build.sh) + camera extrinsics
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
ALL=(driver apt ros docker venv sim assets fp build)
STEPS=("${@:-${ALL[@]}}")
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
APT="$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends"
ROS_PKGS=(ros-jazzy-ros-base ros-jazzy-moveit ros-jazzy-moveit-resources-panda-moveit-config
          ros-jazzy-moveit-resources-panda-description ros-jazzy-pick-ik ros-jazzy-ros2-control
          ros-jazzy-ros2-controllers ros-jazzy-behaviortree-cpp ros-jazzy-vision-msgs ros-jazzy-foxglove-bridge
          ros-jazzy-tf2-eigen ros-jazzy-sensor-msgs-py python3-colcon-common-extensions)

say() { printf '\n\033[1m[install] %s\033[0m\n' "$*"; }
have_pkgs() { dpkg-query -W -f='${Status}\n' "$@" 2>/dev/null | grep -c "ok installed" | grep -qx "$#"; }
docker_cmd() { if docker info >/dev/null 2>&1; then docker "$@"; else $SUDO docker "$@"; fi; }

check() {
  . /etc/os-release
  [ "$VERSION_ID" = 24.04 ] && [ "$(uname -m)" = x86_64 ] || { echo "Needs Ubuntu 24.04 x86_64 (found $PRETTY_NAME, $(uname -m))"; exit 1; }
  local free; free=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 200 ] || echo "[install] warning: ${free} GB free here; a full install needs ~200 GB (Docker's are counted where /var/lib/docker is)"
}

step_driver() {
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    local v; v=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    [ "${v%%.*}" -ge 580 ] && { echo "driver $v"; return; }
    echo "driver $v is older than 580 (CUDA 13): upgrading"
  fi
  $SUDO apt-get update && $APT nvidia-driver-580-open
  say "NVIDIA driver installed: reboot, then run scripts/install.sh again"
  exit 0
}

step_apt() {
  have_pkgs build-essential cmake git git-lfs curl ca-certificates gnupg python3-venv python3-dev && return
  $SUDO apt-get update && $APT build-essential cmake git git-lfs curl ca-certificates gnupg python3-venv python3-dev
}

step_ros() {
  have_pkgs "${ROS_PKGS[@]}" && { echo "ROS 2 Jazzy + packages present"; return; }
  if ! have_pkgs ros2-apt-source; then   # the official apt source package (docs.ros.org, Jazzy install)
    $APT software-properties-common && $SUDO add-apt-repository -y universe
    local v; v=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F'"' '{print $4}')
    curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/$v/ros2-apt-source_$v.noble_all.deb"
    $SUDO dpkg -i /tmp/ros2-apt-source.deb
  fi
  $SUDO apt-get update && $APT "${ROS_PKGS[@]}"
}

step_docker() {
  if ! command -v docker >/dev/null; then   # Docker's apt repo (docs.docker.com/engine/install/ubuntu)
    $SUDO install -m 0755 -d /etc/apt/keyrings
    $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" \
      | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
    $SUDO apt-get update && $APT docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi
  if ! have_pkgs nvidia-container-toolkit; then   # NVIDIA's apt repo (docs.nvidia.com/datacenter/cloud-native)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update && $APT nvidia-container-toolkit
    $SUDO nvidia-ctk runtime configure --runtime=docker && $SUDO systemctl restart docker
  fi
  if [ "$(id -u)" != 0 ] && ! id -nG | grep -qw docker; then
    $SUDO usermod -aG docker "$USER"
    echo "[install] added $USER to the docker group: log out and back in (this run uses sudo docker)"
  fi
  docker_cmd run --rm --gpus all ubuntu:24.04 nvidia-smi -L   # the GPU is visible in containers
}

step_venv() {
  [ -x .venv/bin/python ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
  .venv/bin/pip install -r requirements.txt
  if command -v nvidia-smi >/dev/null; then   # not while building the Docker image: no GPU there
    .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, torch.cuda.get_device_name())"
  fi
}

step_sim() {
  [ -x .venv-sim/bin/python ] && .venv-sim/bin/python -c "import isaacsim" 2>/dev/null && { echo "Isaac Sim present"; return; }
  .venv/bin/uv venv --allow-existing .venv-sim --python /usr/bin/python3.12
  .venv/bin/uv pip install --python .venv-sim/bin/python "isaacsim[all,extscache]==6.1.0.0" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
}

step_assets() {
  [ -f assets/ycb/077_rubiks_cube/textured.obj ] || .venv/bin/python scripts/prepare_ycb.py
  mkdir -p models && (cd models && ../.venv/bin/python -c "
from ultralytics.utils.downloads import attempt_download_asset as get
for f in ('yoloe-11l-seg.pt', 'yoloe-11l-seg-pf.pt', 'mobileclip_blt.ts'):
    get(f)")
}

step_fp() {
  docker_cmd image inspect ppp-isaac-ros:4.5 >/dev/null 2>&1 \
    || docker_cmd build -f docker/Dockerfile --target isaac_ros -t ppp-isaac-ros:4.5 .
  if docker info >/dev/null 2>&1; then docker/foundationpose.sh --prepare; else $SUDO -E docker/foundationpose.sh --prepare; fi
}

step_build() {
  ros2/build.sh
  .venv/bin/python sim/write_calibration.py
}

check
for s in "${STEPS[@]}"; do
  declare -F "step_$s" >/dev/null || { echo "unknown step '$s' (steps: ${ALL[*]})"; exit 1; }
  say "$s"
  "step_$s"
done
say "done. Run it: README.md, 'Run the robot'"
