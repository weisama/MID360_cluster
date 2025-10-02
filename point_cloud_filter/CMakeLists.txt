cmake_minimum_required(VERSION 3.8)
project(point_cloud_filter)

# 检查编译器兼容性
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

# 查找 nlohmann_json 包
find_package(nlohmann_json REQUIRED)

# 包含 PCL 头文件目录
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# 创建可执行文件
add_executable(point_cloud_filter_node src/point_cloud_filter_node.cpp)

# 链接库
ament_target_dependencies(point_cloud_filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(point_cloud_filter_node
  ${PCL_LIBRARIES}
  nlohmann_json::nlohmann_json
)

# 安装可执行文件
install(TARGETS
  point_cloud_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装启动文件
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# 导出依赖
ament_export_dependencies(rclcpp sensor_msgs pcl_conversions)

# 生成环境钩子
ament_package()
