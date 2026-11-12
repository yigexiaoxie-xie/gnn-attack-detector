"""Fast dependency-free structural attack baselines."""

from __future__ import annotations

import random
from typing import Any

import torch


Edge = tuple[int, int]


def _edges(graph: Any) -> set[Edge]:
    edge_index = graph.edge_index.detach().cpu().long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return {
        tuple(sorted((int(left), int(right))))
        for left, right in edge_index.t().tolist()
        if int(left) != int(right)
    }


def _topology(edges: set[Edge], num_nodes: int) -> tuple[list[set[int]], list[int]]:
    neighbors = [set() for _ in range(int(num_nodes))]
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors, [len(items) for items in neighbors]


def _ranked_non_edges(
    clean_edges: set[Edge],
    num_nodes: int,
    count: int,
    seed: int,
) -> list[Edge]:
    n = int(num_nodes)
    wanted = int(count)
    possible = n * (n - 1) // 2 - len(clean_edges)
    if wanted < 0:
        raise ValueError("budget must be non-negative")
    if wanted > possible:
        raise RuntimeError(
            f"could not find {wanted} unique non-edges; only {possible} exist"
        )
    if wanted == 0:
        return []

    generator = random.Random(int(seed))
    pool_size = min(possible, wanted * 4)
    candidates: set[Edge] = set()
    attempts = 0
    attempt_limit = max(10_000, wanted * 200)
    while len(candidates) < pool_size and attempts < attempt_limit:
        left, right = sorted((generator.randrange(n), generator.randrange(n)))
        attempts += 1
        pair = (left, right)
        if left != right and pair not in clean_edges:
            candidates.add(pair)
    if len(candidates) < wanted:
        raise RuntimeError(
            f"could not sample {wanted} unique non-edges within {attempt_limit} attempts"
        )

    neighbors, degrees = _topology(clean_edges, n)
    ranked = []
    for left, right in sorted(candidates):
        shared = len(neighbors[left] & neighbors[right])
        union = degrees[left] + degrees[right] - shared
        overlap = shared / union if union else 0.0
        ranked.append(
            (
                shared,
                overlap,
                -abs(degrees[left] - degrees[right]),
                generator.random(),
                (left, right),
            )
        )
    ranked.sort()
    return [item[-1] for item in ranked[:wanted]]


def _hub_edges(edges: set[Edge], num_nodes: int, count: int, seed: int) -> list[Edge]:
    if count <= 0:
        return []
    _, degrees = _topology(edges, num_nodes)
    generator = random.Random(int(seed))
    ranked = [
        (
            -(degrees[left] + degrees[right]),
            -max(degrees[left], degrees[right]),
            generator.random(),
            (left, right),
        )
        for left, right in sorted(edges)
    ]
    ranked.sort()
    return [item[-1] for item in ranked[:count]]


def _with_edges(graph: Any, edges: set[Edge]) -> Any:
    directed = sorted(edges | {(right, left) for left, right in edges})
    attacked = graph.clone().cpu()
    attacked.edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    return attacked


def run_bridge_add(graph: Any, *, budget: int, seed: int) -> Any:
    """Add structurally distant non-edges without scanning all node pairs."""

    clean_edges = _edges(graph)
    additions = _ranked_non_edges(
        clean_edges,
        int(graph.num_nodes),
        int(budget),
        int(seed),
    )
    return _with_edges(graph, clean_edges | set(additions))


def run_hub_rewire(graph: Any, *, budget: int, seed: int) -> Any:
    """Remove hub-adjacent edges and add distant non-edges with an exact budget."""

    wanted = int(budget)
    if wanted < 0:
        raise ValueError("budget must be non-negative")
    clean_edges = _edges(graph)
    remove_count = wanted // 2
    if remove_count > len(clean_edges):
        raise RuntimeError(
            f"could not remove {remove_count} hub edges; only {len(clean_edges)} exist"
        )
    removals = _hub_edges(
        clean_edges,
        int(graph.num_nodes),
        remove_count,
        int(seed),
    )
    additions = _ranked_non_edges(
        clean_edges,
        int(graph.num_nodes),
        wanted - remove_count,
        int(seed) + 1,
    )
    return _with_edges(graph, (clean_edges - set(removals)) | set(additions))
