import os
from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    
    ld = LaunchDescription()
    
    start_training = Node(
        package='rl_racing',
        executable='start_learning',
        output='screen'
        )
    
    ld.add_action(start_training)
    
    return ld
