import pytest
import torch
from torch_geometric.data import Data


def _graph() -> Data:
    undirected = {
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 3),
        (3, 4),
        (4, 5),
        (4, 7),
        (5, 6),
        (6, 7),
    }
    directed = sorted(undirected | {(right, left) for left, right in undirected})
    return Data(
        x=torch.eye(8),
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
        y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )


def _edges(graph: Data) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(left), int(right))))
        for left, right in graph.edge_index.t().tolist()
        if int(left) != int(right)
    }


def _assert_simple_undirected(graph: Data) -> None:
    directed = [tuple(map(int, pair)) for pair in graph.edge_index.t().tolist()]
    assert all(left != right for left, right in directed)
    assert len(directed) == len(set(directed))
    assert set(directed) == {(right, left) for left, right in directed}


def test_bridge_add_is_exact_deterministic_and_topology_only() -> None:
    from gnn_attack_detector.attacks.fast_structural_adapter import run_bridge_add

    clean = _graph()
    first = run_bridge_add(clean, budget=3, seed=17)
    repeated = run_bridge_add(clean, budget=3, seed=17)
    other_seed = run_bridge_add(clean, budget=3, seed=18)

    assert torch.equal(first.edge_index, repeated.edge_index)
    assert not torch.equal(first.edge_index, other_seed.edge_index)
    assert len(_edges(first) ^ _edges(clean)) == 3
    assert torch.equal(first.x, clean.x)
    assert torch.equal(first.y, clean.y)
    _assert_simple_undirected(first)

    attributed = clean.clone()
    attributed.x = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    assert torch.equal(
        run_bridge_add(attributed, budget=3, seed=17).edge_index,
        first.edge_index,
    )


def test_hub_rewire_is_exact_deterministic_and_preserves_edge_count() -> None:
    from gnn_attack_detector.attacks.fast_structural_adapter import run_hub_rewire

    clean = _graph()
    first = run_hub_rewire(clean, budget=4, seed=19)
    repeated = run_hub_rewire(clean, budget=4, seed=19)

    assert torch.equal(first.edge_index, repeated.edge_index)
    assert len(_edges(first) ^ _edges(clean)) == 4
    assert len(_edges(first)) == len(_edges(clean))
    assert torch.equal(first.x, clean.x)
    assert torch.equal(first.y, clean.y)
    _assert_simple_undirected(first)


def test_fast_structural_attacks_reject_impossible_addition_budget() -> None:
    from gnn_attack_detector.attacks.fast_structural_adapter import run_bridge_add

    complete = _graph()
    pairs = {(left, right) for left in range(8) for right in range(left + 1, 8)}
    directed = sorted(pairs | {(right, left) for left, right in pairs})
    complete.edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()

    with pytest.raises(RuntimeError, match="unique non-edges"):
        run_bridge_add(complete, budget=1, seed=7)
