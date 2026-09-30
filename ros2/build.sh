#!/bin/bash
# Build the ROS 2 workspace (Workstation):  ros2/build.sh [extra colcon args, e.g. --packages-select ppp_task]
# The pybind11 module (ppp_geometry_py) is built with the .venv's pybind11 (>= 2.12): the Python nodes run on the
# .venv's numpy 2, which Ubuntu's pybind11 2.11 doesn't support (it segfaults).
set -e
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
VENV_PY=../.venv/bin/python
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$($VENV_PY -m pybind11 --cmakedir)" -DPython3_EXECUTABLE=/usr/bin/python3 "$@"
