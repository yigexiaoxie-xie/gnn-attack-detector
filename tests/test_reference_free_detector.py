from types import SimpleNamespace

import pytest
import torch

from gnn_attack_detector.detector_features import prepare_graph_input
from gnn_attack_detector.detector_model import SharedGraphDetector
from gnn_attack_detector.detector_model import GraphDetectorBundle, predict_graph_probabilities


def graph(x=None):
    return SimpleNamespace(num_nodes=4, x=x, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))


def test_reference_free_inputs_ignore_labels_and_accept_variable_attributes():
    for width in (3, 17):
        g = graph(torch.rand(4, width))
        before = prepare_graph_input(g)
        g.y = torch.tensor([999, -42, 500, 1000])
        after = prepare_graph_input(g)
        assert before.x.shape == (4, 12)
        assert before.global_features.shape == (4,)
        assert torch.equal(before.x, after.x)
        assert torch.isfinite(before.x).all()


@pytest.mark.parametrize('x', [None, torch.eye(4), torch.zeros(4, 3)])
def test_missing_placeholder_attributes_are_masked(x):
    prepared = prepare_graph_input(graph(x))
    assert torch.count_nonzero(prepared.x[:, 6:]) == 0
    assert torch.count_nonzero(prepared.global_features) == 0


def test_permuted_identity_attributes_remain_masked():
    g = graph(torch.eye(4))
    before = prepare_graph_input(g)
    perm = torch.tensor([2, 0, 3, 1])
    g.x = g.x[perm]
    g.edge_index = torch.argsort(perm)[g.edge_index]
    after = prepare_graph_input(g)
    assert torch.equal(before.x[perm], after.x)
    assert torch.equal(before.global_features, after.global_features)


def test_invalid_edge_indices_rejected():
    g = graph()
    g.edge_index[0, 0] = 99
    with pytest.raises(ValueError):
        prepare_graph_input(g)


def test_gcn_is_jointly_trainable_and_permutation_invariant():
    torch.manual_seed(7)
    g = graph(torch.rand(4, 3))
    model = SharedGraphDetector(hidden_channels=32, dropout=0)
    prepared = prepare_graph_input(g)
    score = model(prepared)
    score.backward()
    assert model.conv1.lin.weight.grad.abs().sum() > 0
    perm = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(perm)
    g.x = g.x[perm]
    g.edge_index = inverse[g.edge_index]
    assert torch.allclose(score, model(prepare_graph_input(g)), atol=1e-6)


def test_inference_retries_on_cpu_after_cuda_failure(monkeypatch):
    import gnn_attack_detector.detector_model as module
    from gnn_attack_detector.device import run_with_cpu_fallback
    devices = []
    original_to = SharedGraphDetector.to
    def failing_cuda(self, device):
        devices.append(str(device))
        if str(device) == 'cuda':
            raise RuntimeError('CUDA out of memory (simulated)')
        return original_to(self, device)
    monkeypatch.setattr(SharedGraphDetector, 'to', failing_cuda)
    monkeypatch.setattr(module, 'run_with_cpu_fallback', lambda operation, **kwargs:
                        run_with_cpu_fallback(operation, cuda_available=lambda: True, **kwargs))
    bundle = GraphDetectorBundle(SharedGraphDetector(dropout=0), 42, .5, ('tiny',), {'tiny': 'auto'}, (), ())
    probabilities = predict_graph_probabilities(bundle, [prepare_graph_input(graph())])
    assert devices == ['cuda', 'cpu']
    assert probabilities.shape == (1,)
    assert torch.isfinite(probabilities).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_shared_gcn_cuda_forward_backward_matches_cpu():
    import copy
    prepared = prepare_graph_input(graph(torch.rand(4, 3)))
    model = SharedGraphDetector(dropout=0)
    cuda_model = copy.deepcopy(model).cuda()
    result = cuda_model(prepared.to('cuda'))
    result.backward()
    assert torch.allclose(model(prepared), result.cpu(), atol=1e-5)
    assert cuda_model.conv1.lin.weight.grad is not None
