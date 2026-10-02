#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE_NAME="stir_slam:latest"
readonly CACHE_DIR="${SCRIPT_DIR}/.cache"

mkdir -p "${CACHE_DIR}"
curl -fL --retry 8 --retry-delay 5 --retry-all-errors \
  "https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.0.0.tar.gz" \
  --output "${CACHE_DIR}/ceres.tar.gz"
curl -fL --retry 8 --retry-delay 5 --retry-all-errors \
  "https://github.com/RainerKuemmerle/g2o/archive/refs/tags/20230223_git.tar.gz" \
  --output "${CACHE_DIR}/g2o.tar.gz"
curl -fL --retry 8 --retry-delay 5 --retry-all-errors \
  "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" \
  --output "${CACHE_DIR}/ros.key"

echo "Building ${IMAGE_NAME}" 

docker build \
  --network host \
  --tag "${IMAGE_NAME}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"
