#!/usr/bin/env bash
# [NOT YET RUN] Build FoundationPose refine/score TensorRT engines from NVIDIA's ONNX models (needs trtexec).
# Goal: replace the PyTorch refiner/scorer in vision/pose.py with TensorRT. Pass the dir holding
# refine_model.onnx + score_model.onnx (NGC "foundationpose" 1.0.0_onnx) as $1.
set -euo pipefail
MODELS=${1:?usage: build_fp_engines.sh <dir with refine_model.onnx and score_model.onnx>}
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
