#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CACHE_DIR="${SCRIPT_DIR}/.cache"

usage() {
  echo "Usage: $0 x86|jetson" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  x86)
    expected_arch=x86_64
    image_name=stir_slam:x86
    dockerfile="${SCRIPT_DIR}/x86/Dockerfile"
    ;;
  jetson)
    expected_arch=aarch64
    image_name=stir_slam:jetson
    dockerfile="${SCRIPT_DIR}/jetson/Dockerfile"
    if [[ ! -f /etc/nv_tegra_release ]]; then
      echo "Jetson build requires a Jetson host with L4T installed." >&2
      exit 2
    fi
    if ! grep -q 'R36 (release), REVISION: 5\.' /etc/nv_tegra_release; then
      echo "This Jetson Dockerfile targets L4T R36.5; check the host release first." >&2
      exit 2
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ "$(uname -m)" != "${expected_arch}" ]]; then
  echo "$1 image must be built on ${expected_arch}; current host is $(uname -m)." >&2
  exit 2
fi

mkdir -p "${CACHE_DIR}"

download_if_missing() {
  local url="$1"
  local destination="$2"
  local temporary

  if [[ -s "${destination}" ]]; then
    return
  fi

  temporary="$(mktemp "${CACHE_DIR}/.download.XXXXXX")"
  if ! curl -fL --retry 8 --retry-delay 5 --retry-all-errors \
      "${url}" --output "${temporary}"; then
    rm -f "${temporary}"
    return 1
  fi
  mv "${temporary}" "${destination}"
}

download_if_missing \
  "https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.0.0.tar.gz" \
  "${CACHE_DIR}/ceres.tar.gz"
download_if_missing \
  "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" \
  "${CACHE_DIR}/ros.key"

echo "Building ${image_name} from ${dockerfile}"

docker build \
  --network host \
  --tag "${image_name}" \
  --file "${dockerfile}" \
  "${SCRIPT_DIR}"
