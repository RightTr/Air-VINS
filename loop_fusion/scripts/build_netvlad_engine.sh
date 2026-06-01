#!/usr/bin/env bash
set -euo pipefail

# Build a TensorRT NetVLAD engine and place it in VINS-Fusion/output/netvlad.trt
# Model choice:
# - Patch-NetVLAD mapillary_WPCA4096.pth.tar (publicly downloadable)
# - 4096-D descriptor, good fit for online loop retrieval
#
# Prerequisites inside docker:
# - python3, torch, torchvision, onnx
# - TensorRT tools installed (trtexec)
# - netvlad_tensorrt repo available at /home/pi/netvlad_tensorrt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
NETVLAD_REPO="/home/pi/netvlad_tensorrt"

MODEL_DIR="${PROJECT_DIR}/output/netvlad_model"
ENGINE_OUT="${PROJECT_DIR}/output/netvlad.trt"
ONNX_OUT="${PROJECT_DIR}/output/netvlad.onnx"
WEIGHT_OUT="${MODEL_DIR}/mapillary_WPCA4096.pth.tar"

mkdir -p "${MODEL_DIR}"
mkdir -p "${PROJECT_DIR}/output"

echo "[1/3] Downloading Patch-NetVLAD weight to ${WEIGHT_OUT}"
if [ ! -f "${WEIGHT_OUT}" ]; then
  wget -O "${WEIGHT_OUT}" \
    "https://huggingface.co/TobiasRobotics/Patch-NetVLAD/resolve/main/mapillary_WPCA4096.pth.tar"
fi

echo "[2/3] Exporting ONNX to ${ONNX_OUT}"
python3 "${NETVLAD_REPO}/scripts/netvlad2onnx.py" \
  -m "${WEIGHT_OUT}" \
  -o "${ONNX_OUT}" \
  -c "${NETVLAD_REPO}/config/speed.ini"

echo "[3/3] Building TensorRT engine to ${ENGINE_OUT}"
trtexec --saveEngine="${ENGINE_OUT}" --onnx="${ONNX_OUT}"

echo "Done."
