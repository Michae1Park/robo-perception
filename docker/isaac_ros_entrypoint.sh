#!/bin/bash
# Source Jazzy (+ the repo's ROS workspace if built for this container) and run the command.
source /opt/ros/jazzy/setup.bash
exec "$@"
