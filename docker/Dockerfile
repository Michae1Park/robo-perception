# Three images from one file; the build context is the repo root.
#
#   isaac_ros        Isaac ROS 4.5 FoundationPose (D-012). Development: FoundationPose runs in it with the repo mounted
#                      docker build -f docker/Dockerfile --target isaac_ros -t ppp-isaac-ros:4.5 .   (scripts/install.sh)
#   foundationpose   deployment: isaac_ros + the meshes and its launch file, at /opt/ppp
#   app              deployment: everything else (ROS 2 + MoveIt, .venv, Isaac Sim, code, built workspace), at /opt/ppp
#                      docker compose -f docker/compose.yaml build
#
# Both deployment images keep the repo at /opt/ppp: pose_node hands FoundationPose mesh paths.

# NVIDIA's Jazzy base image (CUDA 13.0, TensorRT 10.13, driver 580+) with the apt repo moved from release-4.0 to
# release-4.5. Installing at build time avoids I-011: the GL/EGL files that clash are only bind-mounted by the NVIDIA
# runtime in a running container.
FROM nvcr.io/nvidia/isaac/ros:noble-ros2_jazzy_d3e84470d576702a380478a513fb3fc6-amd64 AS isaac_ros
RUN rm -f /etc/apt/sources.list.d/yarn.list \
 && sed -i 's#isaac-ros/release-4.0#isaac-ros/release-4.5#' /etc/apt/sources.list.d/nvidia-isaac-ros.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends isaac-ros-cli
# 4.5's pip-shim packages (torch, onnx) only install once the Isaac ROS CLI says which environment this is;
# NVIDIA's own container layer marks itself "docker-activated" (pip installs into the system Python).
RUN echo 'ISAAC_ROS_ENVIRONMENT=docker-activated' > /etc/isaac-ros-cli/environment.conf \
 && apt-get install -y --no-install-recommends ros-jazzy-isaac-ros-foundationpose \
 && rm -rf /var/lib/apt/lists/*
COPY docker/isaac_ros_entrypoint.sh /usr/local/bin/ppp_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/ppp_entrypoint.sh"]


FROM ubuntu:24.04 AS app
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    OMNI_KIT_ACCEPT_EULA=YES
# Isaac Sim's runtime libraries (Vulkan loader, GL/EGL, the X libraries it links even headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 libegl1 libgl1 libglu1-mesa libxt6 libxrandr2 libxi6 libxcursor1 libxinerama1 libxkbcommon0 \
      libsm6 libice6 libgomp1 libatomic1 libfontconfig1 libnss3 libdbus-1-3 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ppp
# the same install script as development, one layer per step so a code change rebuilds only the last layers
COPY scripts/install.sh scripts/
RUN scripts/install.sh apt ros && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN scripts/install.sh venv && rm -rf /root/.cache
RUN scripts/install.sh sim && rm -rf /root/.cache
COPY vision/ vision/
COPY scripts/prepare_ycb.py scripts/
RUN scripts/install.sh assets
COPY . .
RUN scripts/install.sh build
COPY docker/entrypoint.sh /usr/local/bin/ppp_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/ppp_entrypoint.sh"]


FROM isaac_ros AS foundationpose
COPY --from=app /opt/ppp/assets/ycb /opt/ppp/assets/ycb
COPY docker/fp_run.sh /opt/ppp/docker/
COPY ros2/src/ppp_bringup/launch/foundationpose.launch.py /opt/ppp/ros2/src/ppp_bringup/launch/
WORKDIR /opt/ppp
