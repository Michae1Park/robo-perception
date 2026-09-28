cmake_minimum_required(VERSION 3.8)
project(ppp_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectTarget.srv"
  "srv/EstimatePose.srv"
  "srv/AnalyzeScene.srv"
  "srv/PlanGrasp.srv"
  "srv/PlanPlace.srv"
  "action/PickPlace.action"
  DEPENDENCIES builtin_interfaces geometry_msgs nav_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
