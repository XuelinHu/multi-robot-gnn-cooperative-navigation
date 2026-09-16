#!/usr/bin/env bash
set -u
python3 -u scripts/run_ablation.py --checkpoint results/checkpoints/gnn_n10.pt --output results/ablation_safety_physical.csv
python3 -u scripts/run_radius_ablation.py --checkpoint results/checkpoints/gnn_n10.pt --output results/ablation_radius_physical.csv
python3 -u scripts/run_graph_ablation.py --checkpoint results/checkpoints/gnn_n10.pt --output results/ablation_graph_physical.csv
