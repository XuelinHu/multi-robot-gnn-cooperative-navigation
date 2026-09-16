import numpy as np
import torch

from core.environment import build_graph, make_scenario, rvo_style_action, safety_filter
from core.models import AttentionGNN, GraphTransformer
from core.numpy_policy import load_numpy_policy


def test_graph_has_no_self_edges():
    graph = build_graph(np.asarray([[0, 0], [1, 0]], dtype=np.float32), radius=2.0)
    assert graph.shape == (2, 2)
    assert graph[0, 0] == 0 and graph[1, 1] == 0
    assert graph[0, 1] == 1 and graph[1, 0] == 1


def test_safety_filter_separates_closing_actions():
    position = np.asarray([[0, 0], [0.5, 0]], dtype=np.float32)
    action = np.asarray([[1, 0], [-1, 0]], dtype=np.float32)
    filtered = safety_filter(position, action)
    assert filtered[0, 0] < filtered[1, 0]


def test_scenario_seed_changes_crossing_layout():
    assert not np.allclose(make_scenario(5, 1).starts, make_scenario(5, 2).starts)


def test_random_scenario_has_initial_clearance():
    scenario = make_scenario(20, 10, "random")
    distances = np.linalg.norm(scenario.starts[:, None] - scenario.starts[None, :], axis=-1)
    distances += np.eye(20) * 100.0
    assert float(distances.min()) >= 1.0


def test_rvo_baseline_returns_bounded_actions():
    scenario = make_scenario(5, 4, "open")
    action = rvo_style_action(scenario.starts, np.zeros_like(scenario.starts), scenario.goals, scenario.obstacles)
    assert action.shape == (5, 2)
    assert float(np.linalg.norm(action, axis=1).max()) <= 1.4 + 1e-5


def test_numpy_exported_policy_inference():
    policy = load_numpy_policy('results/checkpoints/gnn_n10.npz')
    scenario = make_scenario(10, 4)
    from core.environment import observation
    features, adjacency = observation(scenario.starts, np.zeros_like(scenario.starts), scenario.goals, scenario.obstacles)
    output = policy(features, adjacency)
    assert output.shape == (10, 2)
    assert np.isfinite(output).all()


def test_rollout_exposes_event_level_collision_metric():
    from core.environment import rollout
    result = rollout(make_scenario(5, 3), 'orca', steps=4)
    assert 'collision_events' in result
    assert result['control_time_ms'] >= 0.0
    assert result['collision_episode'] in (0, 1)
    assert result['strict_success'] in (0, 1)
    assert result['physical_collision_episode'] in (0, 1)


def test_attention_and_transformer_support_batched_graphs():
    x = torch.randn(2, 5, 8)
    adjacency = torch.zeros(2, 5, 5)
    adjacency[:, :, :] = 1.0
    for model in (AttentionGNN(), GraphTransformer()):
        output = model(x, adjacency)
        assert output.shape == (2, 5, 2)
        assert torch.isfinite(output).all()
