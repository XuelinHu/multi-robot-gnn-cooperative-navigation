# Multi-Robot GNN Cooperative Navigation

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-3.10%2B-3776AB" />
  <img height="20" src="https://img.shields.io/badge/pytorch-%3E%3D2.0-EE4C2C" />
  <img height="20" src="https://img.shields.io/badge/numpy-%3E%3D2.0-013243" />
  <img height="20" src="https://img.shields.io/badge/matplotlib-%3E%3D3.7-11557C" />
  <img height="20" src="https://img.shields.io/badge/ROS_2-ament__python-22314E" />
  <img height="20" src="https://img.shields.io/badge/Gazebo-RViz2-FF6F00" />
</p>

This repository contains the reproducible implementation and manuscript for
the study **Graph Neural Network-Based Cooperative Path Planning and Collision
Avoidance for Multi-Robot Systems**.

The project studies dynamic-graph policies for cooperative local navigation.
It includes message-passing GNN, Attention-GNN, and Graph Transformer policies,
five-dimensional relative-motion edge features, joint imitation and safety
losses, an external safety filter, classical baselines, ablation experiments,
an interactive visualizer, and ROS 2/Gazebo/RViz2 integration.

The recorded multiscale evaluation covers four map families, 5/10/20/30/40
robots, and 30 unseen seeds per configuration. The strict success metric
requires goal completion without crossing the physical robot-distance
threshold; it is reported separately from ordinary goal-reaching success.

## Paper and data

- Manuscript PDF: `manuscript/paper_overall.pdf`
- LaTeX source: `manuscript/paper_overall.tex`
- DOI-verified bibliography: `references.bib`
- Main multiscale summary: `results/multiscale_models_test_safety_summary.csv`
- Publication figures: `figures/generated/`
- Research notes: `docs/experiment_log.md` and `docs/results_summary.md`

The bibliography contains 22 entries, including 21 explicit DOIs. Twenty-one
entries were checked against Crossref; 18 entries are from 2023--2025.

## Current baseline

The first milestone is a standalone Ubuntu desktop visualizer. It uses only
Python's standard `tkinter` module, so it can run before installing PyTorch
Geometric or ROS packages. The visualizer shows:

- robot nodes and goal nodes;
- local interaction edges used as graph edges;
- planned trails and obstacle geometry;
- live success, collision, distance, and episode-time metrics;
- classical ORCA/RVO and A* baselines, plus MLP/GNN policy switching.

Run it with:

```bash
cd /ds1/workspace/paper/multi_robot_gnn
python3 visualization/multi_robot_visualizer.py
```

The same window can switch between ORCA, RVO, A* and the learned policy. To load a
checkpoint trained by `scripts/train_policy.py`:

```bash
python3 visualization/multi_robot_visualizer.py --checkpoint results/checkpoints/gnn_n10.pt
```

## Reproducible experiment commands

Generate a small development set (the default is intentionally modest for a
24-GB RTX 3090):

```bash
python3 scripts/generate_dataset.py --robots 5 10 20 --episodes 20 --steps 160

# Additional held-out map families
python3 scripts/generate_dataset.py --robots 5 10 20 --layout open --episodes 20 --steps 160
python3 scripts/generate_dataset.py --robots 5 10 20 --layout narrow --episodes 20 --steps 160
```

Generate safety-projected expert labels for the constrained-training branch:

```bash
python3 scripts/generate_dataset.py --robots 5 10 20 --episodes 20 --steps 160 --safe-labels
python3 scripts/train_policy.py --data data/generated/safe_orca_crossing_n10.npz --method gnn --output results/checkpoints_safe
python3 scripts/train_policy.py --data data/generated/orca_crossing_n10.npz --method gnn --collision-weight 2.0 --output results/checkpoints_joint
python3 scripts/train_policy.py --data data/generated/orca_crossing_n10.npz --method gnn --collision-weight 0.5 --progress-weight 0.5 --output results/checkpoints_progress
python3 scripts/plot_results.py --input results/progress_gnn_comparison.csv --output figures/generated/progress_gnn_comparison.png
```

Train one policy at a time. The scripts use CUDA automatically, keep the
default hidden width at 128, batch size at 128, and save only the best
validation checkpoint:

```bash
python3 scripts/train_policy.py --data data/generated/orca_crossing_n10.npz --method mlp
python3 scripts/train_policy.py --data data/generated/orca_crossing_n10.npz --method gnn
python3 scripts/train_policy.py --data data/generated/orca_crossing_n10.npz --method attention
python3 scripts/train_policy.py --data data/generated/orca_crossing_n10.npz --method transformer
python3 scripts/export_numpy_checkpoint.py --input results/checkpoints/gnn_n10.pt --output results/checkpoints/gnn_n10.npz
```

Evaluate all requested methods on the same seeded episodes and create the first
paper figure:

```bash
python3 scripts/evaluate_baselines.py --mlp-checkpoint results/checkpoints/mlp_n10.pt --gnn-checkpoint results/checkpoints/gnn_n10.pt
python3 scripts/evaluate_baselines.py --robots 5 10 20 --seeds 10 11 12 13 14 15 16 17 18 19 --layouts crossing open random narrow --mlp-checkpoint results/checkpoints/mlp_n10.pt --gnn-checkpoint results/checkpoints/gnn_n10.pt --output results/baselines_final.csv
python3 scripts/summarize_results.py --input results/baselines_final.csv --output results/baselines_final_summary.csv
python3 scripts/plot_generalization.py --input results/baselines_final_summary.csv --output figures/generated/generalization_final.png
python3 scripts/run_ablation.py --checkpoint results/checkpoints/gnn_n10.pt --output results/ablation_safety_final.csv
python3 scripts/plot_ablation.py --input results/ablation_safety_final.csv --output figures/generated/ablation_safety_final.png
python3 scripts/run_radius_ablation.py --checkpoint results/checkpoints/gnn_n10.pt --output results/ablation_radius_final.csv
python3 scripts/plot_radius_ablation.py --input results/ablation_radius_final.csv --output figures/generated/ablation_radius_final.png
python3 scripts/run_graph_ablation.py --checkpoint results/checkpoints/gnn_n10.pt --output results/ablation_graph_final.csv
python3 scripts/plot_graph_ablation.py --input results/ablation_graph_final.csv --output figures/generated/ablation_graph_final.png
python3 scripts/evaluate_training_seeds.py --checkpoints results/checkpoints_seed1/gnn_n10.pt results/checkpoints_seed2/gnn_n10.pt results/checkpoints_seed3/gnn_n10.pt --output results/training_seed_comparison.csv
python3 scripts/summarize_training_seeds.py
python3 scripts/export_paper_tables.py
python3 scripts/run_radius_ablation.py --checkpoint results/checkpoints/gnn_n10.pt
python3 scripts/plot_radius_ablation.py
python3 scripts/plot_results.py
python3 scripts/plot_trajectories.py --gnn-checkpoint results/checkpoints/gnn_n10.pt
python3 scripts/plot_results.py --input results/joint_gnn_comparison.csv --output figures/generated/joint_gnn_comparison.png

# Extended attention/Transformer comparison
python3 scripts/evaluate_baselines.py --robots 5 10 20 --seeds 10 11 12 13 14 15 16 17 18 19 --layouts crossing open random narrow \
  --mlp-checkpoint results/checkpoints/mlp_n10.pt --gnn-checkpoint results/checkpoints/gnn_n10.pt \
  --attention-checkpoint results/checkpoints_attention/attention_n10.pt \
  --transformer-checkpoint results/checkpoints_transformer/transformer_n10.pt \
  --output results/extended_models.csv --quiet
python3 scripts/evaluate_baselines.py --robots 5 10 20 --seeds 10 11 12 13 14 15 16 17 18 19 --layouts crossing open random narrow \
  --mlp-checkpoint results/checkpoints/mlp_n10.pt --gnn-checkpoint results/checkpoints/gnn_n10.pt \
  --attention-checkpoint results/checkpoints_attention/attention_n10.pt \
  --transformer-checkpoint results/checkpoints_transformer/transformer_n10.pt \
  --output results/extended_learned_physical.csv --quiet --learned-only
python3 scripts/summarize_results.py --input results/extended_learned_physical.csv --output results/extended_learned_physical_summary.csv
python3 scripts/plot_extended_comparison.py --input results/extended_learned_physical.csv --output figures/generated/extended_learned_models.png
```

Safety-layer ablation:

```bash
python3 scripts/run_ablation.py --checkpoint results/checkpoints/gnn_n10.pt
```

## ROS 2 / Wheeltec

Build the adapter in a separate ROS workspace, then run the existing Wheeltec
Gazebo model with RViz2 and the controller:

```bash
./ros2/build_ros_adapter.sh
./ros2/run_wheeltec_gnn.sh results/checkpoints/gnn_n10.npz
```

The adapter uses `odom`, `scan` and `cmd_vel` in the current namespace. The
default ROS Python environment can run the fallback controller without PyTorch;
checkpoint inference uses the exported `.npz` format and NumPy, so the ROS
system Python does not need the training PyTorch installation.

For the multi-robot namespaced Gazebo experiment with RViz2:

```bash
bash ros2/run_multi_robot_gnn.sh 5 results/checkpoints/gnn_n10.npz true
```

This creates `/robot_i/odom`, `/robot_i/scan` and `/robot_i/cmd_vel` topics and
one controller that builds a graph over all available robot odometry states.

Every long command is restartable. Use fewer `--episodes`, `--steps` or
`--epochs` for smoke tests; do not run all robot counts and all seeds in one
large GPU process.

The ROS 2 and Wheeltec integration reuses the existing resources documented in
`docs/workspace_inventory.md`.

## Recreate paper figures

The additional multiscale figures are generated directly from the recorded
CSV summary:

```bash
python3 scripts/plot_paper_extended.py
```

This writes vector PDF and 320-dpi PNG versions of the scaling curves and the
map/model strict-success heatmap to `figures/generated/`.

## Scope and limitations

The `ORCA-style` and `RVO-style` controllers are transparent dependency-free
benchmarks implemented for this repository and should not be interpreted as
formal reference-library implementations. The safety filter is an external
wrapper around the learned policy, not a claim that the neural network itself
provides a formal safety guarantee. Current results support reliable use in
low-to-medium density settings, approximately 5--20 robots in the tested
maps; 30--40 robots are reported as high-density stress tests.
