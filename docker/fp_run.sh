#!/bin/bash
# Runs INSIDE the ppp-isaac-ros:4.5 container: fetch FoundationPose's ONNX models from NGC (NVIDIA's model license
# applies), build their TensorRT engines for this GPU once (~2 min), then start the pipeline.
#   docker/fp_run.sh [mustard_bottle]    (the mesh loaded at start)
#   docker/fp_run.sh --prepare           (models and engines only)
# Started by docker/foundationpose.sh (dev) or the foundationpose service of docker/compose.yaml (deployment).
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
M="$REPO/models/isaac_ros/foundationpose"
URL=https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files

mkdir -p "$M"
for m in refine score; do
  [ -f "$M/${m}_model.onnx" ] || curl -sfL -o "$M/${m}_model.onnx" "$URL/${m}_model.onnx"
done
build() {  # name, max batch
  [ -f "$M/$1_trt_engine.plan" ] && return
  echo "[foundationpose] building the $1 TensorRT engine (once)..."
  /usr/src/tensorrt/bin/trtexec --onnx="$M/$1_model.onnx" --saveEngine="$M/$1_trt_engine.plan" \
    --minShapes=input1:1x160x160x6,input2:1x160x160x6 --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --maxShapes=input1:$2x160x160x6,input2:$2x160x160x6 > "$M/$1_trtexec.log" 2>&1 \
    || { echo "trtexec failed, see $M/$1_trtexec.log"; exit 1; }
}
build refine 42
build score 252
[ "$1" = --prepare ] && exit 0
exec ros2 launch "$REPO/ros2/src/ppp_bringup/launch/foundationpose.launch.py" repo:="$REPO" mesh:="${1:-mustard_bottle}"
