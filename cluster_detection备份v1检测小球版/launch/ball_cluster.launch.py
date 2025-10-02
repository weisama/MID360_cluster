from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cluster_detection',
            executable='ball_cluster_node',
            name='ball_cluster_node',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        ),
    ])
