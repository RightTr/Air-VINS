#!/usr/bin/env bash

set -euo pipefail

readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE_DIR="$(cd "${REPO_DIR}/../.." && pwd)"
readonly PACKAGES=(stir_slam loop_fusion)

usage() {
  echo "Usage: $0 ROS1|ROS2|humble|jazzy"
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

selection="$1"
edition=""
distro=""
case "${selection}" in
  ROS1)
    edition="ROS1"
    ;;
  ROS2)
    edition="ROS2"
    distro="${ROS_DISTRO:-}"
    if [[ -z "${distro}" || "${ROS_VERSION:-}" != "2" ]]; then
      echo "ROS2 requires a sourced ROS2 environment (ROS_DISTRO and ROS_VERSION=2)." >&2
      exit 2
    fi
    ;;
  humble|jazzy)
    edition="ROS2"
    distro="${selection}"
    setup_file="/opt/ros/${distro}/setup.bash"
    if [[ ! -f "${setup_file}" ]]; then
      echo "ROS2 ${distro} is not installed at ${setup_file}." >&2
      exit 2
    fi
    set +u
    # shellcheck disable=SC1090
    source "${setup_file}"
    set -u
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ "${edition}" == "ROS1" && "${ROS_VERSION:-}" != "1" ]]; then
  echo "ROS1 requires a sourced ROS1 environment (ROS_VERSION=1)." >&2
  exit 2
fi

for package_dir in vins_estimator loop_fusion; do
  cp -f "${REPO_DIR}/${package_dir}/package_${edition}.xml" \
        "${REPO_DIR}/${package_dir}/package.xml"
done

# Force CMake to reconfigure only these packages when switching ROS editions.
for package_name in "${PACKAGES[@]}"; do
  rm -rf "${WORKSPACE_DIR}/build/${package_name}" \
         "${WORKSPACE_DIR}/install/${package_name}"
done

echo "Building ${PACKAGES[*]} for ${edition}${distro:+ (${distro})}"
if [[ "${edition}" == "ROS1" ]]; then
  cd "${WORKSPACE_DIR}"
  catkin_make -DROS_EDITION=ROS1
else
  cd "${WORKSPACE_DIR}"
  colcon build \
    --packages-select "${PACKAGES[@]}" \
    --cmake-args -DROS_EDITION=ROS2
fi
