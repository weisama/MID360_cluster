cmake_minimum_required(VERSION 3.8)
project(cluster_detection)

# ------------------- 编译选项 -------------------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------- 查找依赖 -------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)

# PCL 及 ROS 封装
find_package(PCL REQUIRED COMPONENTS common io filters segmentation kdtree)
find_package(pcl_conversions REQUIRED)

# ------------------- 可执行文件 -------------------
add_executable(ball_cluster_node src/ball_cluster_node.cpp)

# 一句话搞定头文件、库、link flags
ament_target_dependencies(ball_cluster_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  PCL
  pcl_conversions
  nlohmann_json
)

# ------------------- 安装 -------------------
install(TARGETS ball_cluster_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

# ------------------- 导出依赖（供 find_package 使用） ----------
ament_export_dependencies(
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  PCL
  pcl_conversions
)

ament_package()
