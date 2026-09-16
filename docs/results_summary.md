# Current Results Summary

The development tables are pipeline-validation results; the final package
below is the frozen comparison evidence. No table is interpreted as proof of
GNN superiority. All methods use the same seeded scenario generator and action
limits.

## Models

- `mlp_n10.pt`: independent per-robot MLP trained on ORCA-style actions.
- `gnn_n10.pt`: shared three-layer message-passing policy trained on ORCA-style
  actions.
- `checkpoints_safe/gnn_n10.pt`: message-passing policy trained on actions
  after the shared safety projection.
- `checkpoints_joint/gnn_n10.pt`: message-passing policy trained with an
  additional differentiable one-step separation penalty.
- `checkpoints_progress/gnn_n10.pt`: message-passing policy trained with both
  one-step separation and goal-progress terms.

## Final evaluation package

- `results/baselines_final.csv`: 600 episodes across four map families, five
  methods, three robot counts and ten fixed test seeds.
- `results/baselines_final_summary.csv`: mean, standard deviation and 95% CI
  for success, collision events, episode collision, separation, path length,
  final error and per-step control time.
- `results/ablation_safety_final.csv`: 60 episodes comparing raw GNN and the
  common safety filter.
- `results/ablation_radius_final.csv`: 120 episodes over four communication
  radii.
- `results/ablation_graph_final.csv`: 360 episodes comparing no edges, local
  communication graphs and global graphs.
- `results/training_seed_comparison.csv` and
  `results/training_seed_summary.csv`: evaluation of three independently
  trained GNN checkpoints and their cross-seed mean/std.
- `figures/generated/generalization_final.png`,
  `ablation_safety_final.png`, `ablation_radius_final.png` and
  `ablation_graph_final.png`, `trajectories_final.png` are the corresponding
  paper figures.

The final aggregate table is descriptive. It does not establish GNN
superiority: the n10-trained GNN loses performance on the open map at 10 and
20 robots, while safety filtering improves separation at the cost of task
completion.

## Observed behavior

- The GNN has lower imitation validation loss than the MLP on the development
  set.
- The raw GNN closes the task for many 10-robot and 20-robot episodes, but its
  collision count remains too high for a safety claim.
- Safety-projected labels improve the training target's separation behavior in
  some episodes, but the current model can fail to reach goals. This motivates
  joint progress, collision and control-smoothness objectives in the next
  training stage.
- The first joint-loss run confirms that a separation penalty alone can reduce
  goal progress. Its weight must be scheduled or paired with a progress term;
  it should not be reported as a final improvement.
- Adding the progress term restores some goal-reaching behavior, but the
  current high-density collision rate remains too large for deployment.

## Remaining limitations before manuscript submission

- the RVO controller is a dependency-free finite-lattice implementation rather
  than a formally verified reference implementation;
- the final GNN checkpoint is imitation-trained at 10 robots and needs a
  stronger safety-constrained training stage before deployment claims;
- an Ubuntu desktop screenshot of the RViz2 run still needs to be captured in
  a session with a valid `DISPLAY`.

## Extended attention comparison

- `results/extended_learned_physical.csv` contains 480 episodes comparing MLP, GNN, Attention-GNN and Graph Transformer on the same four layouts, three robot counts and ten seeds.
- `results/extended_learned_physical_summary.csv` contains aggregate metrics including physical collision events and collision-free success.
- `figures/generated/extended_learned_models.png` and `.pdf` are the publication-style comparison figures.
- Attention-GNN uses local masked multi-head neighbor attention. Graph Transformer uses adjacency-masked self-attention with residual feed-forward blocks.
- Overall goal success is 0.817 for MLP, 0.683 for GNN, 0.342 for Attention-GNN and 0.558 for Graph Transformer.
- Overall collision-free success under the physical diameter threshold is 0.017, 0.008, 0.025 and 0.058 respectively. The added attention mechanisms do not by themselves provide collision safety.
- `collision_events` denotes violation of the 0.72 m safety distance, while `physical_collision_events` uses the robot diameter threshold of 0.56 m. These metrics are reported separately.

## Multiscale edge-feature experiment

- `data/expanded/` contains mixed-expert datasets for 5, 10, 20, 30 and 40 robots across all four layouts, with five-dimensional directed edge features.
- `results/multiscale_models_test_summary.csv` and `results/multiscale_models_test_safety_summary.csv` contain 1,800-episode raw and safety-filtered tests, respectively, using 30 unseen seeds.
- Multiscale training raises goal success to 0.988 for GNN and 0.997 for Graph Transformer overall, but high-density collision-free success remains limited.
- Safety filtering gives GNN collision-free success of 0.708 at 20 robots, 0.250 at 30 robots and 0.000 at 40 robots. Graph Transformer gives 0.767, 0.242 and 0.017.
- The result supports 10--20 robots as the current practical operating range and 30--40 robots as stress-test conditions; larger robot counts do not automatically improve safety.
