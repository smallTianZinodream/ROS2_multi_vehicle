import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def vehicle(name, kind, x, y, yaw, peers):
    return Node(
        package='fleet_bringup',
        executable='vehicle_node',
        namespace=name,
        name='vehicle',
        output='screen',
        parameters=[{
            'vehicle_id': name,
            'kind': kind,
            'start_x': x,
            'start_y': y,
            'start_yaw': yaw,
            'home_x': x,
            'home_y': y,
            'peer_topics': peers,
        }],
    )


def generate_launch_description():
    robots = [
        vehicle('tb1', 'robot', -3.5, -2.2, 0.0, ['/tb2/state', '/uav1/state']),
        vehicle('tb2', 'robot', -3.5, 0.2, 0.0, ['/tb1/state', '/uav1/state']),
        vehicle('uav1', 'uav', -3.5, 2.2, 0.0, ['/tb1/state', '/tb2/state']),
    ]
    vehicles = ['tb1', 'tb2', 'uav1']
    dispatcher = Node(
        package='fleet_bringup',
        executable='dispatcher_node',
        name='dispatcher',
        output='screen',
        parameters=[{
            'vehicles': vehicles,
            'min_battery': 40.0,
        }],
    )
    scene = Node(
        package='fleet_bringup',
        executable='scene_node',
        name='scene',
        output='screen',
        parameters=[{'vehicles': vehicles}],
    )
    capture = Node(
        package='fleet_bringup',
        executable='capture_node',
        name='capture',
        output='screen',
    )
    rviz_config = os.path.join(
        get_package_share_directory('fleet_bringup'), 'rviz', 'fleet.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )
    return LaunchDescription([
        *robots,
        scene,
        capture,
        TimerAction(period=1.0, actions=[dispatcher, rviz]),
    ])
