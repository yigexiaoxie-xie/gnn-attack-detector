"""DeepRobust complete-graph attack dispatch with auditable budgets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from gnn_attack_detector.attacks.deeprobust_adapter import (
    load_precomputed_adjacency,
    run_deeprobust_dice,
    run_deeprobust_degree_attack,
    run_deeprobust_fga,
    run_deeprobust_igattack,
    run_deeprobust_metattack,
    run_deeprobust_nettack,
    run_deeprobust_pgdattack,
    run_deeprobust_prbcd,
    run_deeprobust_random,
    run_deeprobust_rnd,
    run_deeprobust_sgattack,
)
from gnn_attack_detector.attacks.greatx_adapter import (
    run_greatx_fgattack_untargeted,
    run_greatx_gfattack,
)
from gnn_attack_detector.attacks.fast_structural_adapter import (
    run_bridge_add,
    run_hub_rewire,
)
from gnn_attack_detector.attacks.registry import attack_spec
from gnn_attack_detector.attacks.structack_adapter import run_structack
from gnn_attack_detector.integrity import evaluate_model


Edge = tuple[int, int]
Feature = tuple[int, int]


class AttackCandidateRejected(RuntimeError):
    """A generated candidate violates its declared protocol contract."""


@dataclass(frozen=True)
class CompleteGraphAttackOutcome:
    graph: Any
    attack_name: str
    target_nodes: tuple[int, ...]
    added_edges: tuple[Edge, ...]
    removed_edges: tuple[Edge, ...]
    changed_features: tuple[Feature, ...]
    requested_modifications: int
    actual_modifications: int
    requested_rate: float | None = None
    actual_rate: float | None = None
    requested_target_count: int | None = None
    local_budget: int | None = None
    per_target_modifications: tuple[int, ...] = ()


def unique_undirected_edges(edge_index: torch.Tensor) -> set[Edge]:
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return {
        tuple(sorted((int(left), int(right))))
        for left, right in edge_index.detach().cpu().t().tolist()
        if int(left) != int(right)
    }


def edge_difference(clean_graph: Any, attacked_graph: Any) -> tuple[set[Edge], set[Edge]]:
    clean = unique_undirected_edges(clean_graph.edge_index)
    attacked = unique_undirected_edges(attacked_graph.edge_index)
    return attacked - clean, clean - attacked


def feature_difference(clean_graph: Any, attacked_graph: Any) -> set[Feature]:
    if tuple(clean_graph.x.shape) != tuple(attacked_graph.x.shape):
        raise RuntimeError("attack changed the node-feature shape")
    changed = torch.nonzero(
        clean_graph.x.detach().cpu() != attacked_graph.x.detach().cpu(),
        as_tuple=False,
    )
    return {(int(node), int(column)) for node, column in changed.tolist()}


def structural_budget(graph: Any, rate: float) -> int:
    value = float(rate)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("structural attack rate must be positive and finite")
    edge_count = len(unique_undirected_edges(graph.edge_index))
    if edge_count == 0:
        raise ValueError("structural attacks require at least one clean edge")
    # DeepRobust's published examples use ``int(rate * undirected_edges)``.
    # Matching that floor convention also keeps official Metattack artifacts
    # auditable at rates whose products are not integers.
    budget = max(1, int(edge_count * value))
    possible = int(graph.num_nodes) * (int(graph.num_nodes) - 1) // 2
    if budget > possible:
        raise ValueError(f"structural budget {budget} exceeds {possible} node pairs")
    return budget


def _outcome(
    clean_graph: Any,
    attacked_graph: Any,
    attack_name: str,
    *,
    requested_modifications: int,
    target_nodes: tuple[int, ...] = (),
    requested_rate: float | None = None,
    local_budget: int | None = None,
    per_target_modifications: tuple[int, ...] = (),
) -> CompleteGraphAttackOutcome:
    if int(clean_graph.num_nodes) != int(attacked_graph.num_nodes):
        raise RuntimeError("core attack protocols must preserve node count")
    if not torch.equal(clean_graph.y.detach().cpu(), attacked_graph.y.detach().cpu()):
        raise RuntimeError("attack changed node labels")
    added, removed = edge_difference(clean_graph, attacked_graph)
    features = feature_difference(clean_graph, attacked_graph)
    actual = len(added) + len(removed) + len(features)
    clean_edge_count = max(len(unique_undirected_edges(clean_graph.edge_index)), 1)
    return CompleteGraphAttackOutcome(
        graph=attacked_graph,
        attack_name=str(attack_name),
        target_nodes=target_nodes,
        added_edges=tuple(sorted(added)),
        removed_edges=tuple(sorted(removed)),
        changed_features=tuple(sorted(features)),
        requested_modifications=int(requested_modifications),
        actual_modifications=actual,
        requested_rate=requested_rate,
        actual_rate=(len(added) + len(removed)) / clean_edge_count,
        requested_target_count=len(target_nodes) if target_nodes else None,
        local_budget=local_budget,
        per_target_modifications=per_target_modifications,
    )


def _global_adapter(spec, graph: Any, surrogate: torch.nn.Module, budget: int, seed: int, device: str) -> Any:
    common = {"budget": int(budget), "seed": int(seed)}
    if spec.algorithm == "random":
        return run_deeprobust_random(graph, attack_type=spec.operation, **common)
    if spec.algorithm == "dice":
        return run_deeprobust_dice(graph, attack_type=spec.operation, **common)
    if spec.algorithm == "degree":
        return run_deeprobust_degree_attack(
            graph, attack_type=spec.operation, **common
        )
    if spec.algorithm == "prbcd":
        return run_deeprobust_prbcd(graph, surrogate=surrogate, device=device, **common)
    if spec.algorithm == "structack":
        return run_structack(graph, **common)
    if spec.algorithm == "fgattack":
        return run_greatx_fgattack_untargeted(
            graph, surrogate=surrogate, device=device, **common
        )
    if spec.algorithm == "pgdattack":
        return run_deeprobust_pgdattack(graph, surrogate=surrogate, device=device, **common)
    if spec.algorithm == "metattack":
        return run_deeprobust_metattack(graph, surrogate=surrogate, device=device, **common)
    if spec.algorithm == "bridge_add":
        return run_bridge_add(graph, **common)
    if spec.algorithm == "hub_rewire":
        return run_hub_rewire(graph, **common)
    raise ValueError(f"no live global adapter for {spec.identifier}")


def run_global_protocol(
    graph: Any,
    surrogate: torch.nn.Module,
    attack_name: str,
    rate: float,
    seed: int,
    *,
    adapter: Callable[..., Any] | None = None,
    device: str = "cpu",
) -> CompleteGraphAttackOutcome:
    spec = attack_spec(attack_name)
    if spec.family != "global":
        raise ValueError(f"not a live global protocol: {attack_name}")
    budget = structural_budget(graph, rate)
    attacked = (
        _global_adapter(spec, graph, surrogate, budget, seed, device)
        if adapter is None
        else adapter(
            graph,
            surrogate=surrogate,
            budget=budget,
            seed=int(seed),
            operation=spec.operation,
            device=device,
        )
    )
    outcome = _outcome(
        graph,
        attacked,
        spec.identifier,
        requested_modifications=budget,
        requested_rate=float(rate),
    )
    if outcome.actual_modifications != budget:
        raise AttackCandidateRejected(
            f"{spec.identifier} produced {outcome.actual_modifications} changes; expected {budget}"
        )
    return outcome


def run_precomputed_global_attack(
    graph: Any,
    attack_name: str,
    rate: float,
    path: str | Path,
) -> CompleteGraphAttackOutcome:
    budget = structural_budget(graph, rate)
    attacked = load_precomputed_adjacency(graph, path)
    outcome = _outcome(
        graph,
        attacked,
        attack_name,
        requested_modifications=budget,
        requested_rate=float(rate),
    )
    if outcome.actual_modifications != budget:
        raise AttackCandidateRejected(
            f"{attack_name} artifact has {outcome.actual_modifications} changes; expected {budget}"
        )
    return outcome


def _targeted_adapter(spec, graph: Any, surrogate: torch.nn.Module, target: int, budget: int, seed: int, device: str) -> Any:
    common = {
        "surrogate": surrogate,
        "target_node": int(target),
        "budget": int(budget),
        "seed": int(seed),
    }
    if spec.algorithm == "nettack":
        return run_deeprobust_nettack(graph, attack_mode=spec.operation, **common)
    if spec.algorithm == "gfattack":
        return run_greatx_gfattack(graph, **common)
    adapters = {
        "rnd": run_deeprobust_rnd,
        "fga": run_deeprobust_fga,
        "sgattack": run_deeprobust_sgattack,
        "igattack": run_deeprobust_igattack,
    }
    if spec.cuda_capable:
        common["device"] = device
    return adapters[spec.algorithm](graph, **common)


def run_multi_target_protocol(
    graph: Any,
    surrogate: torch.nn.Module,
    attack_name: str,
    target_nodes: Iterable[int],
    local_budget: int,
    seed: int,
    *,
    adapter: Callable[..., Any] | None = None,
    device: str = "cpu",
) -> CompleteGraphAttackOutcome:
    spec = attack_spec(attack_name)
    if spec.family != "targeted":
        raise ValueError(f"not a targeted protocol: {attack_name}")
    targets = tuple(int(node) for node in target_nodes)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("target nodes must be non-empty and unique")
    if any(node < 0 or node >= int(graph.num_nodes) for node in targets):
        raise ValueError("target node is outside graph")
    budget = int(local_budget)
    if budget <= 0:
        raise ValueError("local budget must be positive")

    working = graph.clone().cpu()
    per_target: list[int] = []
    for ordinal, target in enumerate(targets):
        before = working.clone().cpu()
        working = (
            _targeted_adapter(
                spec, working, surrogate, target, budget, int(seed) + ordinal, device
            )
            if adapter is None
            else adapter(
                working,
                surrogate=surrogate,
                target_node=target,
                budget=budget,
                seed=int(seed) + ordinal,
                operation=spec.operation,
                device=device,
            )
        )
        added, removed = edge_difference(before, working)
        features = feature_difference(before, working)
        edge_changes = added | removed
        count = len(edge_changes) + len(features)
        if count != budget:
            raise AttackCandidateRejected(
                f"{spec.identifier} target {target} produced {count} changes; expected {budget}"
            )
        if spec.operation == "structure" and (
            features or not any(target in edge for edge in edge_changes)
        ):
            raise AttackCandidateRejected("structure protocol changed the wrong target state")
        if spec.operation == "feature" and (
            edge_changes or not any(node == target for node, _ in features)
        ):
            raise AttackCandidateRejected("feature protocol changed the wrong target state")
        if spec.operation == "hybrid" and not (
            any(target in edge for edge in edge_changes)
            or any(node == target for node, _ in features)
        ):
            raise AttackCandidateRejected("hybrid protocol did not affect its target")
        per_target.append(count)

    return _outcome(
        graph,
        working,
        spec.identifier,
        requested_modifications=len(targets) * budget,
        target_nodes=targets,
        local_budget=budget,
        per_target_modifications=tuple(per_target),
    )


def run_complete_graph_attack(
    *,
    graph: Any,
    surrogate: torch.nn.Module,
    attack_name: str,
    strength: float | int,
    target_nodes: Iterable[int],
    local_budget: int,
    seed: int,
    precomputed_graph_path: str | Path | None = None,
    global_adapter: Callable[..., Any] | None = None,
    targeted_adapter: Callable[..., Any] | None = None,
    device: str = "cpu",
) -> CompleteGraphAttackOutcome:
    spec = attack_spec(attack_name)
    if spec.identifier == "metattack":
        if precomputed_graph_path is not None:
            if not Path(precomputed_graph_path).is_file():
                raise FileNotFoundError(
                    f"Metattack artifact not found: {precomputed_graph_path}"
                )
            return run_precomputed_global_attack(
                graph, spec.identifier, float(strength), precomputed_graph_path
            )
    if spec.family == "global":
        return run_global_protocol(
            graph,
            surrogate,
            spec.identifier,
            float(strength),
            int(seed),
            adapter=global_adapter,
            device=device,
        )
    return run_multi_target_protocol(
        graph,
        surrogate,
        spec.identifier,
        target_nodes,
        int(local_budget),
        int(seed),
        adapter=targeted_adapter,
        device=device,
    )


def eligible_attack_targets(
    graph: Any,
    model: torch.nn.Module,
    *,
    minimum_degree: int,
    required_count: int,
) -> list[int]:
    signals = evaluate_model(model, graph)
    degrees = [0] * int(graph.num_nodes)
    for left, right in unique_undirected_edges(graph.edge_index):
        degrees[left] += 1
        degrees[right] += 1
    eligible = [
        node
        for node in range(int(graph.num_nodes))
        if int(signals["predictions"][node]) == int(graph.y[node])
        and degrees[node] >= int(minimum_degree)
    ]
    if len(eligible) < int(required_count):
        raise ValueError(
            f"only {len(eligible)} eligible targeted nodes; require {required_count}"
        )
    return eligible
