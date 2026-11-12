"""Regression checks for cached predictions and GPU-first attack dispatch."""

import pytest
import torch
from torch_geometric.data import Data

from gnn_attack_detector.attacks.deeprobust_adapter import (
    _DenseGCNSurrogateView,
    run_deeprobust_igattack,
    run_deeprobust_metattack,
)
from gnn_attack_detector.attacks.registry import attack_spec
from gnn_attack_detector.graph_attacks import _targeted_adapter
from gnn_attack_detector.victim import VictimGCN


def tiny_inputs():
    graph = Data(
        x=torch.eye(4),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                                 [1, 0, 2, 1, 3, 2, 0, 3]]),
        y=torch.tensor([0, 0, 1, 1]),
    )
    graph.train_ids = torch.tensor([0, 2])
    graph.val_ids = torch.tensor([1])
    graph.test_ids = torch.tensor([3])
    return graph, VictimGCN(4, 3, 2, 0.0)


def test_cached_output_moves_with_surrogate_without_cuda_hardware():
    graph, model = tiny_inputs()
    view = _DenseGCNSurrogateView(model, graph)
    torch.testing.assert_close(view.output, view.predict())
    assert "output" not in view.state_dict()
    # The meta device checks Module.to migration without requiring a GPU.
    view.to("meta")
    assert view.output.device == view.weight1.device
    assert view.output.device.type == "meta"


def test_igattack_receives_gpu_device_from_dispatch(monkeypatch):
    graph, model = tiny_inputs()
    calls = []

    def capture(graph, **kwargs):
        calls.append(kwargs)
        return graph

    monkeypatch.setattr("gnn_attack_detector.graph_attacks.run_deeprobust_igattack", capture)
    _targeted_adapter(attack_spec("igattack"), graph, model, 0, 1, 42, "cuda")
    assert calls[0].get("device") == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("name", ["metattack", "igattack"])
def test_real_cuda_attack_completes_without_cpu_fallback(name):
    pytest.importorskip("deeprobust")
    graph, model = tiny_inputs()
    view = _DenseGCNSurrogateView(model, graph).to("cuda")
    assert view.output.device.type == "cuda"
    runner = run_deeprobust_metattack if name == "metattack" else run_deeprobust_igattack
    kwargs = {} if name == "metattack" else {"target_node": 0}
    # Call the adapter directly: CUDA errors must fail, not be hidden by fallback.
    attacked = runner(graph, surrogate=model, budget=1, seed=42, device="cuda", **kwargs)
    assert attacked.num_nodes == graph.num_nodes
    torch.testing.assert_close(attacked.x, graph.x)
    torch.testing.assert_close(attacked.y, graph.y)
    before = set(map(tuple, graph.edge_index.t().tolist()))
    after = set(map(tuple, attacked.edge_index.t().tolist()))
    assert len(before.symmetric_difference(after)) == 2
