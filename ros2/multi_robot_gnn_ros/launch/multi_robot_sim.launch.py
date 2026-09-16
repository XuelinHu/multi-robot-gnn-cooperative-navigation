"""Spawn a namespaced multi-Wheeltec Gazebo experiment."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition


def setup(context, *args, **kwargs):
    count = int(LaunchConfiguration('robot_count').perform(context))
    package_share = Path(get_package_share_directory('wheeltec_4wd_bringup'))
    model_source = (package_share / 'models' / 'wheeltec_4wd' / 'model.sdf').read_text()
    world = package_share / 'worlds' / 'wheeltec_4wd_world.sdf'
    temp_dir = Path(tempfile.mkdtemp(prefix='multi_robot_gnn_'))
    bridge_lines = ['- ros_topic_name: "clock"', '  gz_topic_name: "/clock"', '  ros_type_name: "rosgraph_msgs/msg/Clock"', '  gz_type_name: "gz.msgs.Clock"', '  direction: GZ_TO_ROS']
    actions = []
    for index in range(count):
        name = f'robot_{index}'
        model = model_source
        for topic in ('scan', 'odom', 'cmd_vel', 'joint_states'):
            model = model.replace(f'<topic>/{topic}</topic>', f'<topic>/{name}/{topic}</topic>')
        model = model.replace('<odom_topic>/odom</odom_topic>', f'<odom_topic>/{name}/odom</odom_topic>')
        model_path = temp_dir / f'{name}.sdf'
        model_path.write_text(model, encoding='utf-8')
        x = -4.5 + index * 1.8
        y = -3.8 if index % 2 == 0 else 3.8
        actions.append(Node(package='ros_gz_sim', executable='create', name=f'{name}_spawn', arguments=['-name', name, '-file', str(model_path), '-x', str(x), '-y', str(y), '-z', '0.2'], output='screen'))
        for topic, ros_type, gz_type, direction in [
            ('scan', 'sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan', 'GZ_TO_ROS'),
            ('odom', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry', 'GZ_TO_ROS'),
            ('cmd_vel', 'geometry_msgs/msg/Twist', 'gz.msgs.Twist', 'ROS_TO_GZ'),
        ]:
            bridge_lines += [f'- ros_topic_name: "/{name}/{topic}"', f'  gz_topic_name: "/{name}/{topic}"', f'  ros_type_name: "{ros_type}"', f'  gz_type_name: "{gz_type}"', f'  direction: {direction}']
    bridge = temp_dir / 'bridge.yaml'
    bridge.write_text('\n'.join(bridge_lines) + '\n', encoding='utf-8')
    gz_launch = Path(get_package_share_directory('ros_gz_sim')) / 'launch' / 'gz_sim.launch.py'
    actions = [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gz_launch)), launch_arguments={'gz_args': ['-r -s ', str(world)]}.items()),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='multi_robot_bridge', parameters=[{'config_file': str(bridge), 'expand_gz_topic_names': True}], output='screen'),
        *actions,
        Node(package='multi_robot_gnn_ros', executable='multi_gnn_controller', name='multi_gnn_controller', parameters=[{'robot_count': count, 'checkpoint': LaunchConfiguration('checkpoint').perform(context)}], output='screen'),
        Node(package='multi_robot_gnn_ros', executable='multi_robot_markers', name='multi_robot_markers', parameters=[{'robot_count': count}], output='screen'),
    ]
    if LaunchConfiguration('use_rviz').perform(context).lower() == 'true':
        rviz = Path(get_package_share_directory('multi_robot_gnn_ros')) / 'rviz' / 'multi_robot.rviz'
        actions.append(Node(package='rviz2', executable='rviz2', name='multi_robot_rviz', arguments=['-d', str(rviz)], condition=IfCondition(LaunchConfiguration('use_rviz')), output='screen'))
    return actions


def generate_launch_description():
    actions = [
        DeclareLaunchArgument('robot_count', default_value='5'),
        DeclareLaunchArgument('checkpoint', default_value=''),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        OpaqueFunction(function=setup),
    ]
    # Headless runs need an offscreen Qt backend; desktop RViz must keep the
    # user's normal display backend.
    actions.insert(3, SetEnvironmentVariable(
        'QT_QPA_PLATFORM', 'offscreen',
        condition=UnlessCondition(LaunchConfiguration('use_rviz'))
    ))
    return LaunchDescription(actions)
