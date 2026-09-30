#!/bin/bash
# Isaac ROS 4.5 FoundationPose, one instance per target (D-012, D-016), in the ppp-isaac-ros:4.5 container.
#   docker/foundationpose.sh mustard_bottle tomato_soup_can      (perception.launch.py runs this)
# Topics per target: /ppp/fp/<target>/{image, depth_image, camera_info, segmentation} -> /ppp/fp/<target>/output
# First run: downloads the two ONNX models from NGC (NVIDIA's model license applies) and builds the TensorRT
# engines for this GPU into models/isaac_ros/foundationpose/ (~2 min). Needs the image:
#   docker build -f docker/Dockerfile.isaac_ros -t ppp-isaac-ros:4.5 docker/
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=ppp-isaac-ros:4.5
NAME=ppp_foundationpose
M="$REPO/models/isaac_ros/foundationpose"
TARGETS="${*:-mustard_bottle tomato_soup_can}"
URL=https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files

# the container runs as this user: files it writes stay ours, and DDS shared memory works with host processes
RUN=(docker run --rm --gpus all --network host --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp
     -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" -v "$REPO:$REPO" -w "$REPO")

mkdir -p "$M"
for m in refine score; do
  [ -f "$M/${m}_model.onnx" ] || curl -sfL -o "$M/${m}_model.onnx" "$URL/${m}_model.onnx"
done
build() {  # name, max batch
  [ -f "$M/$1_trt_engine.plan" ] && return
  echo "[foundationpose] building $1 TensorRT engine (once)..."
  "${RUN[@]}" --entrypoint /usr/src/tensorrt/bin/trtexec "$IMAGE" --onnx="$M/$1_model.onnx" \
    --saveEngine="$M/$1_trt_engine.plan" --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
    --optShapes=input1:1x160x160x6,input2:1x160x160x6 --maxShapes=input1:$2x160x160x6,input2:$2x160x160x6 \
    > "$M/$1_trtexec.log" 2>&1 || { echo "trtexec failed, see $M/$1_trtexec.log"; exit 1; }
}
build refine 42
build score 252

docker rm -f "$NAME" >/dev/null 2>&1 || true
trap 'docker stop -t 5 "$NAME" >/dev/null 2>&1' INT TERM EXIT
"${RUN[@]}" --name "$NAME" "$IMAGE" ros2 launch "$REPO/ros2/src/ppp_bringup/launch/foundationpose.launch.py" \
  repo:="$REPO" targets:="$(echo $TARGETS | tr ' ' ',')" &
wait $!
