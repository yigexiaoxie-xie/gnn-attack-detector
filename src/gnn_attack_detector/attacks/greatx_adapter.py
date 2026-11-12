"""Lazy GreatX boundary for the two catalogued attack protocols."""

from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Any

import numpy as np
import torch


@contextmanager
def _isolated_random_seed(seed: int):
    """Seed every RNG used by GreatX, then restore the caller's state."""

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


def _load_greatx_attack(name: str):
    """Import GreatX only when one of its attacks is actually selected."""

    try:
        if name == "gfattack":
            from greatx.attack.targeted import GFAttack

            return GFAttack
        if name == "fgattack_untargeted":
            from greatx.attack.untargeted import FGAttack

            return FGAttack
    except ImportError as exc:
        raise RuntimeError(
            "GreatX is required for this selected attack. Run "
            "scripts\\setup_windows_cpu.bat first."
        ) from exc
    raise ValueError(f"unsupported GreatX attack: {name}")


def _undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> set[tuple[int, int]]:
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise RuntimeError("GreatX returned an invalid edge_index")
    directed: set[tuple[int, int]] = set()
    for left, right in edge_index.detach().cpu().long().t().tolist():
        left, right = int(left), int(right)
        if left < 0 or right < 0 or left >= num_nodes or right >= num_nodes:
            raise RuntimeError("GreatX returned a node index outside the graph")
        if left == right:
            raise RuntimeError("GreatX returned a self-loop")
        directed.add((left, right))
    if any((right, left) not in directed for left, right in directed):
        raise RuntimeError("GreatX returned an asymmetric graph")
    return {tuple(sorted(edge)) for edge in directed}


def _validated_graph(clean: Any, candidate: Any) -> Any:
    if int(candidate.num_nodes) != int(clean.num_nodes):
        raise RuntimeError("GreatX changed the graph node count")
    if not torch.equal(candidate.x.detach().cpu(), clean.x.detach().cpu()):
        raise RuntimeError("GreatX structure protocol changed node features")
    if not torch.equal(candidate.y.detach().cpu(), clean.y.detach().cpu()):
        raise RuntimeError("GreatX changed node labels")
    attacked_edges = _undirected_edges(candidate.edge_index, int(clean.num_nodes))
    ordered = sorted(attacked_edges)
    directed = [(left, right) for left, right in ordered for left, right in ((left, right), (right, left))]
    attacked = clean.clone().cpu()
    attacked.edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    return attacked


class _DenseGreatXSurrogate(torch.nn.Module):
    """Copy VictimGCN weights into GreatX's dense-adjacency call shape."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        try:
            first = model.conv1.lin.weight.detach().cpu().t().contiguous()
            second = model.conv2.lin.weight.detach().cpu().t().contiguous()
            first_bias = model.conv1.bias
            second_bias = model.conv2.bias
        except AttributeError as exc:
            raise TypeError(
                "GreatX FGAttack requires the project's two-layer VictimGCN"
            ) from exc
        self.register_buffer("weight1", first)
        self.register_buffer("weight2", second)
        self.register_buffer(
            "bias1",
            torch.zeros(first.shape[1])
            if first_bias is None
            else first_bias.detach().cpu().clone(),
        )
        self.register_buffer(
            "bias2",
            torch.zeros(second.shape[1])
            if second_bias is None
            else second_bias.detach().cpu().clone(),
        )

    @staticmethod
    def _normalize(adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.is_sparse:
            adjacency = adjacency.to_dense()
        adjacency = adjacency.float()
        adjacency = adjacency + torch.eye(
            int(adjacency.shape[0]),
            dtype=adjacency.dtype,
            device=adjacency.device,
        )
        degree = adjacency.sum(dim=1).clamp_min(1.0)
        scale = degree.pow(-0.5)
        return scale[:, None] * adjacency * scale[None, :]

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_weight
        normalized = self._normalize(adjacency)
        hidden = (normalized @ (features @ self.weight1) + self.bias1).relu()
        return normalized @ (hidden @ self.weight2) + self.bias2


def _seeded_victim_nodes(graph: Any, seed: int, max_nodes: int = 512) -> torch.Tensor:
    """Choose a reproducible victim subset for untargeted FGAttack.

    FGAttack keeps the complete graph and changes global edges, but its loss is
    aggregated over the supplied victim nodes.  Using every test node makes a
    Cora run unnecessarily expensive and tends to produce the exact same
    gradient ranking for every retry.  A seed-specific subset keeps each run
    reproducible, gives retries a different objective, and bounds the CPU
    cost without changing the graph's node count or attack scope.
    """

    victim_nodes = getattr(graph, "test_ids", None)
    if victim_nodes is None:
        victim_nodes = torch.arange(int(graph.num_nodes), dtype=torch.long)
    elif not torch.is_tensor(victim_nodes):
        victim_nodes = torch.tensor(victim_nodes, dtype=torch.long)
    else:
        victim_nodes = victim_nodes.detach().cpu().long()
    victim_nodes = torch.unique(victim_nodes, sorted=False)
    if int(victim_nodes.numel()) == 0:
        victim_nodes = torch.arange(int(graph.num_nodes), dtype=torch.long)
    limit = max(1, int(max_nodes))
    if int(victim_nodes.numel()) <= limit:
        return victim_nodes
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(int(victim_nodes.numel()), generator=generator)
    return victim_nodes[permutation[:limit]]


def run_greatx_gfattack(
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    target_node: int,
    budget: int,
    seed: int,
    attacker_class=None,
) -> Any:
    """Run GreatX GFAttack against one target node."""

    del surrogate
    attack_type = attacker_class or _load_greatx_attack("gfattack")
    with _isolated_random_seed(seed):
        attacker = attack_type(
            graph.clone().cpu(),
            K=2,
            T=min(128, max(1, int(graph.num_nodes) - 1)),
            device="cpu",
            seed=int(seed),
        )
        attacker.set_max_perturbations(int(budget), verbose=False)
        attacker.reset()
        attacker.attack(
            int(target_node),
            num_budgets=int(budget),
            direct_attack=True,
            structure_attack=True,
            feature_attack=False,
            ll_constraint=False,
            disable=True,
        )
    return _validated_graph(graph, attacker.data())


def run_greatx_fgattack_untargeted(
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    budget: int,
    seed: int,
    device: str = "cpu",
    attacker_class=None,
) -> Any:
    """Run GreatX's full-graph, structure-only FGAttack."""

    attack_type = attacker_class or _load_greatx_attack("fgattack_untargeted")
    with _isolated_random_seed(seed):
        device_graph = graph.clone().to(device)
        attacker = attack_type(device_graph, device=device, seed=int(seed))
        attacker.set_max_perturbations(int(budget), verbose=False)
        victim_nodes = _seeded_victim_nodes(graph, seed)
        attacker.setup_surrogate(
            _DenseGreatXSurrogate(surrogate).to(device).eval(),
            victim_nodes.to(device),
            device_graph.y.detach().long()[victim_nodes.to(device)],
        )
        attacker.reset()
        attacker.attack(
            num_budgets=int(budget),
            structure_attack=True,
            feature_attack=False,
            disable=True,
        )
    return _validated_graph(graph, attacker.data())
