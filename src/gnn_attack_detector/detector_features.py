"""Reference-free descriptors, inspired by ProGNN's attribute smoothness prior.

Not the full ProGNN algorithm: no reference subtraction, labels, graph repair,
low-rank decomposition or dense N-by-N optimization.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 4
PREPROCESSING_VERSION = 1
NODE_FEATURE_NAMES = (
    "log_degree", "relative_degree", "isolated", "neighbor_degree_mean",
    "neighbor_degree_std", "neighbor_degree_max", "edge_cosine_mean",
    "edge_cosine_std", "edge_cosine_min", "edge_cosine_max",
    "has_attribute_neighbor", "attribute_neighbor_fraction",
)
GLOBAL_FEATURE_NAMES = ("cosine_mean", "cosine_std", "cosine_q05", "valid_edge_fraction")


@dataclass(frozen=True)
class PreparedGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    global_features: torch.Tensor

    def to(self, device):
        return PreparedGraph(self.x.to(device), self.edge_index.to(device), self.global_features.to(device))


def _stats(values, indices, n):
    counts = torch.bincount(indices, minlength=n).float()
    mean = torch.zeros(n).index_add_(0, indices, values) / counts.clamp_min(1)
    squares = torch.zeros(n).index_add_(0, indices, values.square())
    std = (squares / counts.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
    low = torch.full((n,), float("inf")).scatter_reduce_(0, indices, values, reduce="amin")
    high = torch.full((n,), -float("inf")).scatter_reduce_(0, indices, values, reduce="amax")
    low[counts == 0] = 0
    high[counts == 0] = 0
    return mean, std, low, high, counts


def resolve_attribute_mode(graph, requested="auto"):
    mode = getattr(graph, "attribute_mode", "auto") if requested == "auto" else requested
    if mode not in {"auto", "content", "none"}:
        raise ValueError("attribute_mode must be auto, content, or none")
    return mode


def prepare_graph_input(graph: Any, *, attribute_mode: str = "auto", chunk_size: int = 4096) -> PreparedGraph:
    """Read edges/attributes only. Use `none` for known placeholder attributes.

    Auto masks constant, zero and identity attributes, but cannot infer all
    attribute semantics. Dataset names and node labels never enter this function.
    """
    attribute_mode = resolve_attribute_mode(graph, attribute_mode)
    n = int(graph.num_nodes)
    if n < 1 or chunk_size < 1:
        raise ValueError("num_nodes and chunk_size must be positive")
    edges = graph.edge_index.detach().cpu()
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be int64 with shape [2, E]")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError("edge_index contains a node outside the graph")
    left, right = edges
    codes = left * n + right
    if bool((left == right).any()) or codes.unique().numel() != codes.numel():
        raise ValueError("self-loops and duplicate edges are not accepted")
    if not torch.equal(codes.sort().values, (right * n + left).sort().values):
        raise ValueError("undirected edges must contain both directions")
    degree = torch.bincount(left, minlength=n).float()
    log_degree = degree.log1p()
    neighbor_mean, neighbor_std, _, neighbor_max, _ = _stats(log_degree[right], left, n)
    similarities = torch.zeros(left.numel())
    valid = torch.zeros(left.numel(), dtype=torch.bool)
    raw = getattr(graph, "x", None)
    if raw is not None:
        raw = raw.detach().cpu().float()
        if raw.layout != torch.strided or raw.ndim != 2 or raw.shape[0] != n:
            raise ValueError("x must be a dense [num_nodes, num_attributes] matrix")
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("x must contain finite values")
        if attribute_mode == "auto":
            # A node permutation turns identity into a permutation matrix.
            identity = (raw.shape[1] == n and int(torch.count_nonzero(raw)) == n
                        and bool((raw.max(dim=1).values == 1).all())
                        and bool((raw.sum(dim=0) == 1).all()))
            if identity or bool((raw == raw[:1]).all()):
                attribute_mode = "none"
        if attribute_mode != "none" and raw.shape[1]:
            norms = raw.norm(dim=1)
            normalized = raw / norms.clamp_min(1e-12)[:, None]
            valid = (norms[left] > 1e-12) & (norms[right] > 1e-12)
            step = min(chunk_size, max(1, 2_000_000 // raw.shape[1]))
            for start in range(0, left.numel(), step):
                end = start + step
                similarities[start:end] = (normalized[left[start:end]] * normalized[right[start:end]]).sum(1).clamp(-1, 1)
    mean, std, low, high, counts = _stats(similarities[valid], left[valid], n)
    x = torch.stack((log_degree, degree / degree.mean().clamp_min(1), (degree == 0).float(),
                     neighbor_mean, neighbor_std, neighbor_max, mean, std, low, high,
                     (counts > 0).float(), counts / degree.clamp_min(1)), dim=1)
    values = similarities[valid & (left < right)]
    global_features = torch.zeros(4)
    if values.numel():
        global_features = torch.stack((values.mean(), values.std(unbiased=False),
                                       torch.quantile(values, .05), valid.float().mean()))
    return PreparedGraph(x, edges.contiguous(), global_features)


def save_prepared_graph(path: Path, graph: PreparedGraph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"format_version": SCHEMA_VERSION, "x": graph.x.cpu(),
                "preprocessing_version": PREPROCESSING_VERSION,
                "edge_index": graph.edge_index.cpu(), "global_features": graph.global_features.cpu()}, temporary)
    temporary.replace(path)


def load_prepared_graph(path: str | Path) -> PreparedGraph:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format_version") != SCHEMA_VERSION
            or payload.get("preprocessing_version") != PREPROCESSING_VERSION):
        raise ValueError("obsolete detector inputs; run rebuild-detector")
    graph = PreparedGraph(payload["x"], payload["edge_index"], payload["global_features"])
    if graph.x.ndim != 2 or graph.x.shape[1] != 12 or graph.global_features.shape != (4,):
        raise ValueError("invalid prepared graph schema")
    if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.global_features).all():
        raise ValueError("nonfinite prepared graph")
    return graph
