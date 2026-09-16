# Research Work Plan

## Working title

基于图神经网络的多机器人协同路径规划与避碰方法研究

## Research question

Can a graph policy use time-varying robot interaction relations to produce
short, collision-free and scalable local actions when the number of robots,
obstacle layout and communication range change?

## Fixed experimental definition

- Robot: planar differential-drive mobile robot abstraction.
- Node: one robot, with position, velocity, goal-relative position and local
  obstacle features.
- Edge: robot pair within a communication radius; edge attributes are relative
  position, relative velocity and distance.
- Action: linear velocity and angular velocity, or a 2D local velocity in the
  first simplified experiment.
- Main model: message-passing GNN with a shared policy head for every robot.
- Safety layer: velocity clipping and emergency separation constraint.
- Metrics: success rate, collision rate, path length, arrival time, minimum
  separation, inference time and communication degree.

## Milestones

### M0 - Environment audit and reproducible baseline

Status: implemented (development data and training pipeline verified).

Deliverables: workspace inventory, experiment protocol, standalone visualizer,
cooperative potential-field baseline and screenshot-capable run.

### M1 - Scenario and data generator

Generate fixed-seed episodes with 5, 10 and 20 robots. Include open space,
random obstacles, narrow passages, crossing traffic and limited communication.
The current generator exposes the four map families as `open`, `random`,
`narrow` and `crossing`, with rejection sampling for valid random starts.
Store observations, graph edges, actions, rewards, obstacle metadata, layout
metadata and episode indices in a versioned `.npz` or PyTorch dataset format.

### M2 - Classical and neural baselines

Status: implemented in the shared simulator; the final 600-episode evaluation
is complete, including a dependency-free RVO-style candidate-velocity baseline.

Implement or wrap A*, ORCA/RVO and an independent per-robot MLP policy. Use
these as comparison baselines before claiming a GNN improvement; a centralized
MLP is outside the current fixed experimental scope.

### M3 - GNN policy

Status: implemented as a dependency-light message-passing model; GPU smoke
training verified on the local RTX 3090.

Implement a dependency-light message-passing model first. Then optionally add
PyTorch Geometric after the data contract is stable. Train with imitation
learning from ORCA/MAPF trajectories, followed by reinforcement learning or a
collision-penalized fine-tuning stage.

### M4 - Evaluation and ablations

Status: final comparison, event-level collision metrics, per-step control-time
metrics, safety ablation and communication-radius ablation are complete. The
learned checkpoint is trained at 10 robots, so cross-scale and cross-map
results remain generalization evidence rather than a deployment guarantee.

Test unseen robot counts, unseen maps, obstacle density, sensor noise and
communication radius. Ablate graph edges, edge attributes, recurrent state,
safety layer and training method.

### M5 - ROS 2 and Gazebo integration

Status: single Wheeltec and three-robot namespaced Gazebo runs are verified.
RViz2 MarkerArray output for robots, goals, trails and graph edges is verified;
the desktop GUI itself requires a valid Ubuntu `DISPLAY`.

Reuse the Wheeltec model and world. Start with a single robot topic contract,
then namespace multiple robot instances. Connect the learned policy to
`/robot_i/cmd_vel`, `/robot_i/odom` and LiDAR-derived local observations.

### M6 - Visualization and paper evidence

Status: desktop policy switcher, RViz2 view, final baseline/generalization,
ablation and trajectory figures are generated under `figures/generated/`.

The same visual interface should show the selected method, graph edges,
trajectories and live metrics. Export deterministic figures and CSV results for
the paper. RViz2/Gazebo screenshots can be used as the 3D/ROS evidence.

### M7 - Manuscript

Write Introduction, Related Work, Problem Formulation, Method, Experimental
Setup, Results, Ablation, Limitations and Conclusion only after the result
tables are frozen.

## First paper-grade experiment matrix

| Factor | Values |
| --- | --- |
| Robot count | 5, 10, 20 |
| Map | open, random, narrow passage, crossing |
| Communication radius | 2.0 m, 4.0 m, global |
| Method | A*, ORCA/RVO, independent MLP, GNN |
| Seed | 0-4 for development, 10-19 for final test |

No numerical improvement should be claimed until all methods use the same
initial states, goals, obstacle maps, action limits and termination rules.
