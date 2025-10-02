import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取功能包目录
    pkg_dir = get_package_share_directory('point_cloud_filter')
    
    # 创建节点配置
    point_cloud_filter_node = Node(
        package='point_cloud_filter',
        executable='point_cloud_filter_node',
        name='point_cloud_filter',
        output='screen',
        parameters=[]
    )
    
    return LaunchDescription([
        point_cloud_filter_node
    ])
