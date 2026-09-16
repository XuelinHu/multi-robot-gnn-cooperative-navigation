from setuptools import setup

package_name = 'multi_robot_gnn_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/multi_robot_sim.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/multi_robot.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'gnn_controller = multi_robot_gnn_ros.gnn_controller:main',
        'multi_gnn_controller = multi_robot_gnn_ros.multi_controller:main',
        'multi_robot_markers = multi_robot_gnn_ros.marker_visualizer:main',
    ]},
)
