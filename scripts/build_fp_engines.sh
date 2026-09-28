#!/usr/bin/env bash
# Build the FoundationPose refine/score TensorRT engines once (inside the Isaac ROS container).
# All per-object FoundationPose instances share these engines.
set -euo pipefail
MODELS=${1:-${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/isaac_ros_assets/models/foundationpose}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

for f in refine_model.onnx score_model.onnx; do
  [[ -f "$MODELS/$f" ]] || { echo "missing $MODELS/$f (download the FoundationPose models from NGC first)"; exit 1; }
done

"$TRTEXEC" --onnx="$MODELS/refine_model.onnx" --saveEngine="$MODELS/refine_trt_engine.plan" --fp16 \
  --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --maxShapes=input1:42x160x160x6,input2:42x160x160x6

"$TRTEXEC" --onnx="$MODELS/score_model.onnx" --saveEngine="$MODELS/score_trt_engine.plan" --fp16 \
  --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
  --maxShapes=input1:252x160x160x6,input2:252x160x160x6

echo "engines written to $MODELS"
