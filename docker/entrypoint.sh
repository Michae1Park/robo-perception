#!/bin/bash
# app image: ROS 2 Jazzy + the built workspace sourced, then the command (docker/compose.yaml).
source /opt/ppp/ros2/install/setup.bash
exec "$@"
