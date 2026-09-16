# Experiment Log

## Verified on 2026-09-15

- Hardware: NVIDIA GeForce RTX 3090, approximately 23 GiB visible to PyTorch.
- Offline training: CUDA PyTorch, hidden width 128, batch size 128, 30 epochs.
- Development data: 5, 10 and 20 robots, 20 episodes and 160 steps per robot
  count, deterministic seeds and compressed NumPy storage.
- ROS build: `multi_robot_gnn_ros` built successfully with ROS 2 Jazzy.
- Gazebo: Wheeltec server-only simulation spawned successfully after making the
  existing `headless:=true` argument effective.
- Topics observed: `/scan` about 80 Hz, `/odom` about 28 Hz, `/cmd_vel`, `/tf`,
  `/joint_states` and stereo topics.
- Controller: exported NumPy GNN checkpoint loaded by the ROS system Python;
  `/cmd_vel` observed at 10 Hz.
- Multi-robot ROS smoke test: three namespaced Wheeltec models spawned and
  `/robot_0/odom` plus `/robot_0/cmd_vel` were observed.
- RViz visualization adapter: the three-robot run published
  `/multi_robot/markers` as a `visualization_msgs/MarkerArray`, including robot
  spheres, goal cylinders, bounded trajectory line strips, and communication
  graph edges. The installed RViz configuration is
  `ros2/multi_robot_gnn_ros/rviz/multi_robot.rviz`.
- The one-command entrypoint `ros2/run_multi_robot_gnn.sh` was smoke-tested
  with three robots; it exposed all namespaced odom/scan/cmd_vel topics and
  received a MarkerArray message before clean shutdown.
- ROS adapter build script now uses a regular merged install so the ament
  package-index marker is present with the current setuptools environment.
- Tests: seven core tests pass with the system `pytest` command.
- Joint loss training: a 30-epoch GNN run with collision weight 2.0 completed
  on CUDA; the resulting comparison is saved in `results/joint_gnn_comparison.csv`.
- Full evaluation: 120 episodes across crossing/random layouts, 5 seeds and
  5/10/20 robot counts; summary with 95% confidence intervals is saved in
  `results/baselines_summary.csv`.
- Final evaluation: 600 episodes across crossing/open/random/narrow layouts,
  ORCA-style, RVO-style, A*, MLP and GNN methods, 10 fixed test seeds and
  5/10/20 robot counts. Results and 95% confidence intervals are saved in
  `results/baselines_final.csv` and `results/baselines_final_summary.csv`.
- Final ablations: safety-layer comparison has 60 episodes and radius
  comparison has 120 episodes; outputs are `results/ablation_safety_final.csv`
  and `results/ablation_radius_final.csv`.
- Graph ablation: 360 episodes compare no-edge, local-graph and global-graph
  variants; results are saved in `results/ablation_graph_final.csv` and
  `figures/generated/ablation_graph_final.png`.
- Training-seed robustness: three independent 30-epoch GNN trainings completed
  on CUDA and were evaluated over 360 identical episodes; outputs are
  `results/training_seed_comparison.csv` and
  `results/training_seed_summary.csv`.
- Paper tables: `scripts/export_paper_tables.py` generated
  `manuscript/tables.md` directly from the frozen CSV files.
- Control-time instrumentation records mean controller time per simulation
  step. The finite-lattice RVO-style baseline is substantially more expensive
  than learned policies, so it is reported separately from task metrics.

## Interpretation boundary

The current labels come from an ORCA-style local controller and the learned
policies are trained by imitation. The current results establish a reproducible
pipeline, not a demonstrated research improvement. The RVO controller is a
dependency-free finite-lattice implementation, and the GNN still requires
safety-constrained training before any deployment claim.

## Multiscale edge-feature experiment on 2026-09-16

- Expanded training data to 5, 10, 20, 30 and 40 robots on crossing, open,
  random and narrow layouts. Each configuration contains 8 episodes and 100
  steps for the initial multiscale training package.
- Each sample now stores directed five-dimensional edge features: normalized
  relative position, normalized relative velocity and normalized closing speed.
- Mixed expert labels alternate ORCA-style, RVO-style and safety-projected
  actions by episode, and are marked with `expert=mixed`.
- A shared 40-node padded representation with node masks trains one model over
  all robot counts. The loss combines action imitation, one-step separation,
  goal progress and velocity smoothness terms.
- New checkpoints are under `results/checkpoints_multiscale_gnn`,
  `results/checkpoints_multiscale_attention` and
  `results/checkpoints_multiscale_transformer`.
- Independent test: 1,800 episodes using 30 unseen seeds (100--129) across all
  four layouts and five robot counts. Safety-filtered test: another 1,800
  episodes.
- Without safety filtering, goal success remains high at 30/40 robots but
  collision-free success falls to zero at these densities. With the safety
  filter, GNN collision-free success is 0.708 at 20 robots, 0.250 at 30 and
  0.000 at 40; Graph Transformer is 0.767, 0.242 and 0.017 respectively.
- Gazebo pressure checks spawned 5, 10 and 20 robots successfully and also
  completed 30/40-robot entity creation before the bounded smoke-test timeout.
