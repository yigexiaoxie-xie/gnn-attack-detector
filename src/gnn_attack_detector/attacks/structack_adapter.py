"""Deterministic low-degree, low-similarity Structack adapter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _clean_structure(graph: Any) -> tuple[set[tuple[int, int]], list[set[int]], list[int]]:
    num_nodes = int(graph.num_nodes)
    edges: set[tuple[int, int]] = set()
    neighbors = [set() for _ in range(num_nodes)]
    for left, right in graph.edge_index.detach().cpu().long().t().tolist():
        left, right = int(left), int(right)
        if left == right:
            raise RuntimeError("Structack requires a loop-free clean graph")
        if left < 0 or right < 0 or left >= num_nodes or right >= num_nodes:
            raise RuntimeError("Structack clean graph has an invalid node index")
        pair = tuple(sorted((left, right)))
        edges.add(pair)
        neighbors[left].add(right)
        neighbors[right].add(left)
    degrees = [len(items) for items in neighbors]
    return edges, neighbors, degrees


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _ranked_additions(graph: Any, budget: int, seed: int) -> list[tuple[int, int]]:
    edges, neighbors, degrees = _clean_structure(graph)
    num_nodes = int(graph.num_nodes)
    possible = num_nodes * (num_nodes - 1) // 2 - len(edges)
    if int(budget) > possible:
        raise ValueError("Structack budget exceeds the available non-edges")

    generator = np.random.default_rng(int(seed))
    tie_break = generator.random(num_nodes)
    low_degree_nodes = sorted(
        range(num_nodes),
        key=lambda node: (degrees[node], float(tie_break[node]), node),
    )
    pool_size = min(
        num_nodes,
        max(32, int(math.ceil(2.25 * math.sqrt(max(1, int(budget)))) * 2)),
    )

    while True:
        pool = low_degree_nodes[:pool_size]
        left_nodes = pool[::2]
        right_nodes = pool[1::2]
        scored: list[tuple[float, int, float, int, int]] = []
        for left in left_nodes:
            for right in right_nodes:
                pair = tuple(sorted((left, right)))
                if pair in edges:
                    continue
                scored.append(
                    (
                        _jaccard(neighbors[left], neighbors[right]),
                        degrees[left] + degrees[right],
                        float(tie_break[left] + tie_break[right]),
                        pair[0],
                        pair[1],
                    )
                )
        if len(scored) >= int(budget):
            scored.sort()
            return [(left, right) for _, _, _, left, right in scored[: int(budget)]]
        if pool_size >= num_nodes:
            raise RuntimeError("Structack could not rank enough low-similarity non-edges")
        pool_size = min(num_nodes, max(pool_size + 2, pool_size * 2))


def run_structack(
    graph: Any,
    *,
    budget: int,
    seed: int,
) -> Any:
    """Add exactly ``budget`` low-degree, low-Jaccard-similarity edges."""

    count = int(budget)
    if count <= 0:
        raise ValueError("Structack budget must be positive")
    additions = _ranked_additions(graph, count, int(seed))
    directed = [
        edge
        for left, right in additions
        for edge in ((left, right), (right, left))
    ]
    attacked = graph.clone().cpu()
    added_edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    attacked.edge_index = torch.cat(
        (attacked.edge_index.detach().cpu().long(), added_edge_index),
        dim=1,
    )
    return attacked
