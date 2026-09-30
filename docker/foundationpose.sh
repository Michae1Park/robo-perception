#!/bin/bash
# Isaac ROS 4.5 FoundationPose (D-012), one instance for every object (D-043), in the ppp-isaac-ros:4.5 container
# (scripts/install.sh builds it). The repo is mounted at its own path; the container runs docker/fp_run.sh.
#   docker/foundationpose.sh [mustard_bottle]      (the mesh loaded at start; perception.launch.py runs this)
#   docker/foundationpose.sh --prepare             (only fetch the models and build the engines: scripts/install.sh)
# Topics: /ppp/fp/{image, depth_image, camera_info, segmentation} -> /ppp/fp/output. pose_node switches the mesh
# (parameter /ppp/fp/foundationpose mesh_file_path) before each request.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME=ppp_foundationpose
# the container runs as this user: files it writes stay ours, and DDS shared memory works with host processes
RUN=(docker run --rm --gpus all --network host --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp
     -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" -v "$REPO:$REPO" -w "$REPO")

if [ "$1" = --prepare ]; then
  exec "${RUN[@]}" ppp-isaac-ros:4.5 "$REPO/docker/fp_run.sh" --prepare
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
trap 'docker stop -t 5 "$NAME" >/dev/null 2>&1' INT TERM EXIT
"${RUN[@]}" --name "$NAME" ppp-isaac-ros:4.5 "$REPO/docker/fp_run.sh" "$@" &
wait $!
