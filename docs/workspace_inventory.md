# Existing Workspace Inventory

The local workspace already contains a suitable robotics stack:

| Resource | Location | Planned use |
| --- | --- | --- |
| ROS 2 Jazzy learning workspace | `/ds1/workspace/robot` | ROS conventions, bag recording, web console pattern |
| Wheeltec 4WD ROS 2 workspace | `/ds1/workspace/wheeltec_4wd` | Gazebo Harmonic robot model, world, LiDAR, odometry, RViz2 |
| Wheeltec world | `/ds1/workspace/wheeltec_4wd/src/wheeltec_4wd_bringup/worlds/wheeltec_4wd_world.sdf` | Obstacle geometry and later multi-robot scene design |
| Wheeltec model | `/ds1/workspace/wheeltec_4wd/src/wheeltec_4wd_bringup/models/wheeltec_4wd/model.sdf` | Differential-drive and sensor reference |
| Existing web console | `/ds1/workspace/robot/web` | Optional ROS bridge dashboard integration |

The research implementation will live in this independent directory so that
existing robot workspaces are not modified accidentally.

