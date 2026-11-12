"""Strict conversion boundary for DeepRobust graph attacks."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch


def _load_random_class():
    try:
        from deeprobust.graph.global_attack import Random
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust is required; run scripts\\setup_windows_cpu.bat first"
        ) from exc
    return Random


def _load_dice_class():
    try:
        from deeprobust.graph.global_attack import DICE
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust is required; run scripts\\setup_windows_cpu.bat first"
        ) from exc
    return DICE


def _load_nettack_class():
    try:
        from deeprobust.graph.targeted_attack import Nettack
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust is required; run scripts\\setup_windows_cpu.bat first"
        ) from exc
    return Nettack


def _load_targeted_class(name: str):
    try:
        from deeprobust.graph.targeted_attack import FGA, IGAttack, RND, SGAttack
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust targeted attacks are unavailable; run setup_windows_cpu.bat"
        ) from exc
    return {"rnd": RND, "fga": FGA, "igattack": IGAttack, "sgattack": SGAttack}[name]


def _load_global_class(name: str):
    try:
        from deeprobust.graph.global_attack import (
            Metattack,
            PGDAttack,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust global attacks are unavailable; run setup_windows_cpu.bat"
        ) from exc
    return {
        "metattack": Metattack,
        "pgdattack": PGDAttack,
    }[name]


def _load_prbcd_class():
    """Load PRBCD lazily because it also imports optional sparse extensions."""

    try:
        from deeprobust.graph.global_attack import PRBCD
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "DeepRobust PRBCD is unavailable. Run "
            "scripts\\setup_windows_cpu.bat and install the compatible "
            "torch-sparse wheel."
        ) from exc
    return PRBCD


def _load_degree_class():
    try:
        from deeprobust.graph.global_attack import OtherNodeEmbeddingAttack
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust degree attack is unavailable; run "
            "scripts\\setup_windows_cpu.bat"
        ) from exc
    return lambda: OtherNodeEmbeddingAttack("degree")


@contextmanager
def _isolated_random_seed(seed: int):
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


def _graph_adjacency(graph: Any) -> sp.csr_matrix:
    edge_index = graph.edge_index.detach().cpu().long()
    rows = edge_index[0].numpy()
    columns = edge_index[1].numpy()
    values = np.ones(rows.shape[0], dtype=np.float32)
    adjacency = sp.csr_matrix(
        (values, (rows, columns)),
        shape=(int(graph.num_nodes), int(graph.num_nodes)),
        dtype=np.float32,
    )
    adjacency.sum_duplicates()
    if adjacency.nnz:
        adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    return adjacency


def _graph_features(graph: Any) -> sp.csr_matrix:
    return sp.csr_matrix(
        graph.x.detach().cpu().float().numpy(),
        dtype=np.float32,
    )


def _dense_vector(values: Any) -> np.ndarray:
    """Flatten both SciPy sparse results and ordinary NumPy arrays."""

    if sp.issparse(values):
        values = values.toarray()
    return np.asarray(values).reshape(-1)


def _nettack_cooccurrence_constraint_compat(attacker: Any, nodes: np.ndarray) -> None:
    """DeepRobust 0.2.9 logic without its obsolete ``numpy.matrix.A1`` calls."""

    words_graph = attacker.cooc_matrix.copy()
    feature_count = int(attacker.modified_features.shape[1])
    words_graph.setdiag(0)
    words_graph = words_graph > 0
    word_degrees = _dense_vector(words_graph.sum(axis=0))
    inverse_degrees = np.reciprocal(word_degrees.astype(float) + 1e-8)
    scores_per_node = np.zeros(int(attacker.nnodes))
    for node in range(int(attacker.nnodes)):
        feature_ids = attacker.modified_features[node, :].nonzero()[1]
        scores_per_node[node] = np.sum(inverse_degrees[feature_ids.tolist()])
    scores_matrix = sp.lil_matrix((int(attacker.nnodes), feature_count))
    for node in nodes:
        common_words = words_graph.multiply(attacker.modified_features[int(node)])
        inverse = inverse_degrees[common_words.nonzero()[1]]
        row_ids = common_words.nonzero()[0]
        scores_matrix[int(node)] = np.array(
            [inverse[row_ids == index].sum() for index in range(feature_count)]
        )
    attacker.cooc_constraint = sp.csr_matrix(
        scores_matrix - 0.5 * scores_per_node[:, None] > 0
    )


def _nettack_feature_scores_compat(attacker: Any) -> tuple[np.ndarray, np.ndarray]:
    """Calculate Nettack feature scores with array-safe flattening."""

    if attacker.cooc_constraint is None:
        _nettack_cooccurrence_constraint_compat(attacker, attacker.influencer_nodes)
    logits = attacker.compute_logits()
    wrong_class = attacker.strongest_wrong_class(logits)
    surrogate_loss = logits[attacker.label_u] - logits[wrong_class]
    gradient = (
        attacker.gradient_wrt_x(attacker.label_u)
        - attacker.gradient_wrt_x(wrong_class)
    )
    flipped = sp.lil_matrix(gradient * -1)
    flipped[attacker.modified_features.nonzero()] *= -1
    influencer_features = sp.lil_matrix(attacker.modified_features.shape)
    influencer_features[attacker.influencer_nodes] = attacker.modified_features[
        attacker.influencer_nodes
    ]
    flipped = flipped.multiply((attacker.cooc_constraint + influencer_features) > 0)
    used = {
        tuple(int(value) for value in perturbation)
        for perturbation in attacker.feature_perturbations
        if len(perturbation) == 2
    }

    def available_indices(values: Any) -> np.ndarray:
        indices = np.array(values.nonzero()).T
        if used and int(indices.shape[0]):
            keep = [tuple(int(value) for value in row) not in used for row in indices]
            indices = indices[np.asarray(keep, dtype=bool)]
        return indices

    nonzero_indices = available_indices(flipped)
    if int(nonzero_indices.shape[0]) == 0:
        # Nettack co-occurrence constraint has no candidate on identity-feature
        # graphs such as Polblogs. Keep Nettack's gradient ranking, but relax
        # only that unavailable linguistic constraint for the target node.
        unconstrained = sp.lil_matrix(attacker.modified_features.shape)
        unconstrained[attacker.influencer_nodes] = 1.0
        flipped = sp.lil_matrix(gradient * -1)
        flipped[attacker.modified_features.nonzero()] *= -1
        flipped = flipped.multiply(unconstrained.tocsr())
        nonzero_indices = available_indices(flipped)
    if int(nonzero_indices.shape[0]) == 0:
        raise RuntimeError("Nettack found no unused feature perturbation")
    values = _dense_vector(flipped[tuple(nonzero_indices.T)])
    sorting = np.argsort(values)
    sorted_indices = nonzero_indices[sorting]
    gradients = _dense_vector(flipped[tuple(sorted_indices.T)])
    scores = _dense_vector(surrogate_loss - gradients)
    return sorted_indices[::-1], scores[::-1]


def _graph_labels(graph: Any) -> np.ndarray:
    return graph.y.detach().cpu().long().numpy()


def _split_ids(graph: Any, name: str) -> np.ndarray:
    attribute = f"{name}_ids"
    if not hasattr(graph, attribute):
        raise ValueError(
            f"DeepRobust {name} split is missing from the known dataset graph"
        )
    values = getattr(graph, attribute)
    if torch.is_tensor(values):
        values = values.detach().cpu().long().numpy()
    return np.asarray(values, dtype=np.int64)


def _normalized_dense_adjacency(graph: Any) -> torch.Tensor:
    adjacency = torch.from_numpy(_graph_adjacency(graph).toarray()).float()
    adjacency = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    scale = degree.pow(-0.5)
    return scale[:, None] * adjacency * scale[None, :]


class _DenseGCNSurrogateView(torch.nn.Module):
    """Independent differentiable copy of the fixed two-layer PyG GCN."""

    def __init__(self, model: torch.nn.Module, graph: Any):
        super().__init__()
        try:
            first = model.conv1.lin.weight.detach().cpu().t().contiguous()
            second = model.conv2.lin.weight.detach().cpu().t().contiguous()
            first_bias = model.conv1.bias
            second_bias = model.conv2.bias
        except AttributeError as exc:
            raise TypeError(
                "DeepRobust gradient attacks require the project's two-layer VictimGCN"
            ) from exc
        self.weight1 = torch.nn.Parameter(first.clone())
        self.weight2 = torch.nn.Parameter(second.clone())
        self.bias1 = torch.nn.Parameter(
            torch.zeros(first.shape[1])
            if first_bias is None else first_bias.detach().cpu().clone()
        )
        self.bias2 = torch.nn.Parameter(
            torch.zeros(second.shape[1])
            if second_bias is None else second_bias.detach().cpu().clone()
        )
        self.nfeat = int(first.shape[0])
        self.nclass = int(second.shape[1])
        self.hidden_sizes = [int(first.shape[1])]
        self.with_relu = True
        self.register_buffer(
            "_clean_features",
            torch.from_numpy(_graph_features(graph).toarray()).float(),
        )
        self.register_buffer("_clean_adjacency", _normalized_dense_adjacency(graph))
        with torch.no_grad():
            output = self(self._clean_features, self._clean_adjacency)
        # Metattack reads this cache directly; it must follow Module.to(device).
        self.register_buffer("output", output, persistent=False)

    @staticmethod
    def _matmul(adjacency: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adjacency, values) if adjacency.is_sparse else adjacency @ values

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if features.is_sparse:
            features = features.to_dense()
        hidden = (self._matmul(adjacency, features @ self.weight1) + self.bias1).relu()
        logits = self._matmul(adjacency, hidden @ self.weight2) + self.bias2
        return torch.nn.functional.log_softmax(logits, dim=1)

    def predict(self) -> torch.Tensor:
        self.eval()
        return self(self._clean_features, self._clean_adjacency)


class _PRBCDSurrogateView(torch.nn.Module):
    """Sparse, weighted view of the fixed victim model for PRBCD."""

    name = "VictimGCN-PRBCD-view"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        try:
            first = model.conv1.lin.weight.detach().cpu().t().contiguous()
            second = model.conv2.lin.weight.detach().cpu().t().contiguous()
            first_bias = model.conv1.bias
            second_bias = model.conv2.bias
        except AttributeError as exc:
            raise TypeError(
                "DeepRobust PRBCD requires the project's two-layer VictimGCN"
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
    def _propagate(
        values: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        from torch_geometric.nn.conv.gcn_conv import gcn_norm

        normalized_edges, normalized_weights = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=int(values.shape[0]),
            add_self_loops=True,
            dtype=values.dtype,
        )
        adjacency = torch.sparse_coo_tensor(
            torch.stack((normalized_edges[1], normalized_edges[0])),
            normalized_weights,
            (int(values.shape[0]), int(values.shape[0])),
            device=values.device,
        ).coalesce()
        return torch.sparse.mm(adjacency, values)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self._propagate(
            features @ self.weight1,
            edge_index,
            edge_weight,
        )
        hidden = (hidden + self.bias1).relu()
        logits = self._propagate(
            hidden @ self.weight2,
            edge_index,
            edge_weight,
        ) + self.bias2
        return torch.nn.functional.log_softmax(logits, dim=1)

    def predict(self, *args) -> torch.Tensor:
        self.eval()
        return self.forward(*args)


class _SGCSurrogateView:
    """Expose the linearized fixed GCN through SGAttack's small SGC interface."""

    def __init__(self, model: torch.nn.Module, graph: Any, device: str = "cpu"):
        first = model.conv1.lin.weight.detach().to(device)
        second = model.conv2.lin.weight.detach().to(device)
        combined = second @ first
        bias = model.conv2.bias
        self.conv1 = SimpleNamespace(
            K=2,
            lin=SimpleNamespace(
                weight=combined.contiguous(),
                bias=None if bias is None else bias.detach().to(device).clone(),
            ),
        )
        model_copy = deepcopy(model).to(device).eval()
        with torch.no_grad():
            logits, _ = model_copy(graph.x.detach().to(device), graph.edge_index.detach().to(device))
        self._output = torch.nn.functional.log_softmax(logits, dim=1)

    def predict(self) -> torch.Tensor:
        return self._output.clone()


def _validate_modified_adjacency(
    adjacency: Any,
    *,
    num_nodes: int,
) -> sp.csr_matrix:
    if torch.is_tensor(adjacency):
        adjacency = adjacency.detach().cpu().numpy()
    if not sp.issparse(adjacency):
        adjacency = sp.csr_matrix(adjacency)
    matrix = adjacency.tocsr(copy=True).astype(np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.shape != (int(num_nodes), int(num_nodes)):
        raise RuntimeError("DeepRobust attack changed the adjacency shape")
    if matrix.diagonal().any():
        raise RuntimeError("DeepRobust attack produced a self-loop")
    if (matrix != matrix.T).nnz:
        raise RuntimeError("DeepRobust attack produced an asymmetric graph")
    if matrix.nnz and not np.all(matrix.data == 1):
        raise RuntimeError("DeepRobust attack produced a weighted graph")
    return matrix


def _graph_with_adjacency(graph: Any, adjacency: Any) -> Any:
    matrix = _validate_modified_adjacency(
        adjacency,
        num_nodes=int(graph.num_nodes),
    ).tocoo()
    order = np.lexsort((matrix.col, matrix.row))
    edge_array = np.vstack((matrix.row[order], matrix.col[order])).astype(
        np.int64,
        copy=False,
    )
    attacked = graph.clone().cpu()
    attacked.edge_index = torch.from_numpy(edge_array).long().contiguous()
    return attacked


def _graph_with_attack_state(
    graph: Any,
    adjacency: Any,
    features: Any | None = None,
) -> Any:
    attacked = _graph_with_adjacency(graph, adjacency)
    if features is None:
        return attacked
    values = features.toarray() if sp.issparse(features) else np.asarray(features)
    if values.shape != tuple(graph.x.shape):
        raise RuntimeError("DeepRobust attack changed the node-feature shape")
    if not np.isfinite(values).all():
        raise RuntimeError("DeepRobust attack produced non-finite node features")
    attacked.x = torch.from_numpy(np.asarray(values, dtype=np.float32)).float()
    return attacked


def _graph_with_weighted_edge_index(
    graph: Any,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
) -> Any:
    edges = edge_index.detach().cpu().long()
    if edges.ndim != 2 or int(edges.shape[0]) != 2:
        raise RuntimeError("DeepRobust attack returned an invalid edge_index")
    if edge_weight is None:
        weights = torch.ones(int(edges.shape[1]), dtype=torch.float32)
    else:
        weights = edge_weight.detach().cpu().float().reshape(-1)
    if int(weights.numel()) != int(edges.shape[1]):
        raise RuntimeError("DeepRobust attack returned mismatched edge weights")
    if not torch.isfinite(weights).all():
        raise RuntimeError("DeepRobust attack returned non-finite edge weights")
    keep = weights > 0.5
    rows = edges[0, keep].numpy()
    columns = edges[1, keep].numpy()
    adjacency = sp.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, columns)),
        shape=(int(graph.num_nodes), int(graph.num_nodes)),
        dtype=np.float32,
    )
    return _graph_with_adjacency(graph, adjacency)


def _graph_with_boolean_masks(graph: Any) -> Any:
    data = graph.clone().cpu()
    for name in ("train", "val", "test"):
        mask = torch.zeros(int(data.num_nodes), dtype=torch.bool)
        mask[torch.from_numpy(_split_ids(data, name)).long()] = True
        setattr(data, f"{name}_mask", mask)
    return data


def load_precomputed_adjacency(graph: Any, path: str | Path) -> Any:
    """Load one trusted SciPy sparse attacked adjacency against ``graph``."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"precomputed attack adjacency not found: {source}")
    try:
        adjacency = sp.load_npz(source)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid precomputed attack adjacency: {source}") from exc
    return _graph_with_adjacency(graph, adjacency)


@dataclass(frozen=True)
class _WeightView:
    weight: torch.Tensor


class _NettackSurrogateView:
    """Expose the fixed PyG GCN weights in DeepRobust's expected orientation."""

    def __init__(self, model: torch.nn.Module):
        try:
            first = model.conv1.lin.weight.detach().cpu().t().contiguous()
            second = model.conv2.lin.weight.detach().cpu().t().contiguous()
            in_channels = int(model.conv1.in_channels)
            hidden_channels = int(model.conv1.out_channels)
            out_channels = int(model.conv2.out_channels)
        except AttributeError as exc:
            raise TypeError(
                "DeepRobust Nettack requires the project's two-layer VictimGCN"
            ) from exc
        self.nfeat = in_channels
        self.nclass = out_channels
        self.hidden_sizes = [hidden_channels]
        self.gc1 = _WeightView(first)
        self.gc2 = _WeightView(second)


def run_deeprobust_random(
    graph: Any,
    *,
    budget: int,
    seed: int,
    attack_type: str = "flip",
    attacker_class=None,
) -> Any:
    """Run DeepRobust Random with its native add/remove/flip switch."""

    operation = str(attack_type).lower()
    if operation not in {"add", "remove", "flip"}:
        raise ValueError(f"unsupported Random operation: {attack_type}")
    attacker = (attacker_class or _load_random_class())()
    adjacency = _graph_adjacency(graph)
    with _isolated_random_seed(seed):
        attacker.attack(adjacency, n_perturbations=int(budget), type=operation)
    return _graph_with_adjacency(graph, attacker.modified_adj)


def run_deeprobust_dice(
    graph: Any,
    *,
    budget: int,
    seed: int,
    attack_type: str = "mix",
    attacker_class=None,
) -> Any:
    """Run DICE or one explicit add-only/remove-only DICE rule variant."""

    operation = str(attack_type).lower()
    if operation not in {"add", "remove", "mix"}:
        raise ValueError(f"unsupported DICE operation: {attack_type}")
    attack_type = attacker_class or _load_dice_class()
    adjacency = _graph_adjacency(graph)
    labels = graph.y.detach().cpu().long().numpy()
    if operation != "mix":
        matrix = adjacency.tolil(copy=True)
        generator = np.random.default_rng(int(seed))
        if operation == "remove":
            upper = sp.triu(adjacency, k=1).tocoo()
            candidates = [
                (int(left), int(right))
                for left, right in zip(upper.row, upper.col)
                if labels[int(left)] == labels[int(right)]
            ]
            if len(candidates) < int(budget):
                raise RuntimeError("DICE remove-only has too few same-class edges")
            for index in generator.choice(len(candidates), size=int(budget), replace=False):
                left, right = candidates[int(index)]
                matrix[left, right] = 0
                matrix[right, left] = 0
        else:
            selected: set[tuple[int, int]] = set()
            attempts = 0
            limit = max(10_000, int(budget) * 500)
            while len(selected) < int(budget) and attempts < limit:
                left, right = generator.integers(0, int(graph.num_nodes), size=2)
                left, right = sorted((int(left), int(right)))
                attempts += 1
                if (
                    left == right
                    or labels[left] == labels[right]
                    or adjacency[left, right] != 0
                ):
                    continue
                selected.add((left, right))
            if len(selected) != int(budget):
                raise RuntimeError("DICE add-only could not find enough cross-class non-edges")
            for left, right in selected:
                matrix[left, right] = 1
                matrix[right, left] = 1
        return _graph_with_adjacency(graph, matrix.tocsr())

    attacker = None
    for attempt in range(20):
        candidate = attack_type()
        try:
            with _isolated_random_seed(int(seed) + attempt):
                candidate.attack(
                    adjacency,
                    labels,
                    n_perturbations=int(budget),
                )
        except IndexError as exc:
            if "too many indices for array" not in str(exc):
                raise
            continue
        attacker = candidate
        break
    if attacker is None:
        raise RuntimeError(
            "DeepRobust DICE repeatedly selected an empty removal set"
        )
    modified = attacker.modified_adj.tolil(copy=True)
    clean_upper = sp.triu(adjacency, k=1).tocsr()
    changed_upper = sp.triu((adjacency != modified.tocsr()).astype(np.int8), k=1).tocoo()
    changed = {(int(left), int(right)) for left, right in zip(changed_upper.row, changed_upper.col)}
    if len(changed) > int(budget):
        for left, right in sorted(changed)[int(budget):]:
            value = int(adjacency[left, right] != 0)
            modified[left, right] = value
            modified[right, left] = value
        changed = set(sorted(changed)[: int(budget)])
    if len(changed) < int(budget):
        generator = np.random.default_rng(int(seed) + 10_000)
        removals = [
            (int(left), int(right))
            for left, right in zip(clean_upper.tocoo().row, clean_upper.tocoo().col)
            if labels[int(left)] == labels[int(right)] and (int(left), int(right)) not in changed
        ]
        additions: list[tuple[int, int]] = []
        attempts = 0
        while len(additions) < int(budget) and attempts < max(10_000, int(budget) * 500):
            left, right = sorted(generator.integers(0, int(graph.num_nodes), size=2).tolist())
            attempts += 1
            pair = (int(left), int(right))
            if (
                left == right
                or labels[left] == labels[right]
                or adjacency[left, right] != 0
                or pair in changed
                or pair in additions
            ):
                continue
            additions.append(pair)
        candidates = removals + additions
        generator.shuffle(candidates)
        for left, right in candidates:
            if len(changed) >= int(budget):
                break
            value = 0 if adjacency[left, right] != 0 else 1
            modified[left, right] = value
            modified[right, left] = value
            changed.add((left, right))
    if len(changed) != int(budget):
        raise RuntimeError("DICE could not satisfy the requested unique-change budget")
    return _graph_with_adjacency(graph, modified.tocsr())


def run_deeprobust_nettack(
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    target_node: int,
    budget: int,
    seed: int,
    attack_mode: str = "structure",
    attacker_class=None,
) -> Any:
    """Run one DeepRobust Nettack structure, feature, or hybrid operation."""

    mode = str(attack_mode).lower()
    if mode not in {"structure", "feature", "hybrid"}:
        raise ValueError(f"unsupported Nettack mode: {attack_mode}")
    view = _NettackSurrogateView(surrogate)
    attack_type = attacker_class or _load_nettack_class()
    attacker = attack_type(
        view,
        nnodes=int(graph.num_nodes),
        attack_structure=mode in {"structure", "hybrid"},
        attack_features=mode in {"feature", "hybrid"},
        device="cpu",
    ).to("cpu")
    if mode in {"feature", "hybrid"}:
        attacker.feature_scores = lambda: _nettack_feature_scores_compat(attacker)
    if mode == "feature":
        attacker.influencer_nodes = np.array([int(target_node)], dtype=np.int64)
        # DeepRobust 0.2.9 casts this field unconditionally even when
        # structure attacks are disabled.  Its default is a Python list.
        attacker.potential_edges = np.empty((0, 2), dtype=np.int32)
    adjacency = _graph_adjacency(graph)
    features = sp.csr_matrix(
        graph.x.detach().cpu().float().numpy(),
        dtype=np.float32,
    )
    labels = graph.y.detach().cpu().long().numpy()
    with _isolated_random_seed(seed):
        attacker.attack(
            features,
            adjacency,
            labels,
            int(target_node),
            n_perturbations=int(budget),
            verbose=False,
        )
    return _graph_with_attack_state(
        graph,
        attacker.modified_adj,
        attacker.modified_features if mode in {"feature", "hybrid"} else None,
    )


def run_deeprobust_rnd(
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    target_node: int,
    budget: int,
    seed: int,
    attacker_class=None,
) -> Any:
    """Run DeepRobust RND as a structure-only targeted attack."""

    del surrogate
    attacker = (attacker_class or _load_targeted_class("rnd"))()
    with _isolated_random_seed(seed):
        attacker.attack(
            _graph_adjacency(graph),
            _graph_labels(graph),
            _split_ids(graph, "train"),
            int(target_node),
            n_perturbations=int(budget),
        )
    return _graph_with_adjacency(graph, attacker.modified_adj)


def _run_gradient_targeted(
    name: str,
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    target_node: int,
    budget: int,
    seed: int,
    device: str = "cpu",
    attacker_class=None,
) -> Any:
    attack_type = attacker_class or _load_targeted_class(name)
    view = _DenseGCNSurrogateView(surrogate, graph).to(device)
    attacker = attack_type(
        view,
        nnodes=int(graph.num_nodes),
        feature_shape=tuple(graph.x.shape),
        attack_structure=True,
        attack_features=False,
        device=device,
    ).to(device)
    kwargs: dict[str, Any] = {}
    if name == "fga":
        kwargs["verbose"] = False
    elif name == "igattack":
        kwargs["steps"] = 10
    with _isolated_random_seed(seed):
        attacker.attack(
            _graph_features(graph),
            _graph_adjacency(graph),
            _graph_labels(graph),
            _split_ids(graph, "train"),
            int(target_node),
            n_perturbations=int(budget),
            **kwargs,
        )
    return _graph_with_adjacency(graph, attacker.modified_adj)


def run_deeprobust_fga(graph: Any, **kwargs) -> Any:
    return _run_gradient_targeted("fga", graph, **kwargs)


def run_deeprobust_igattack(graph: Any, **kwargs) -> Any:
    return _run_gradient_targeted("igattack", graph, **kwargs)


def run_deeprobust_sgattack(
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    target_node: int,
    budget: int,
    seed: int,
    device: str = "cpu",
    attacker_class=None,
) -> Any:
    """Run DeepRobust SGAttack through a fixed, linearized SGC view."""

    view = _SGCSurrogateView(surrogate, graph, device=device)
    attack_type = attacker_class or _load_targeted_class("sgattack")
    attacker = attack_type(
        view,
        nnodes=int(graph.num_nodes),
        attack_structure=True,
        attack_features=False,
        device=device,
    )
    with _isolated_random_seed(seed):
        attacker.attack(
            _graph_features(graph),
            _graph_adjacency(graph),
            _graph_labels(graph),
            int(target_node),
            n_perturbations=int(budget),
            direct=True,
        )
    return _graph_with_adjacency(graph, attacker.modified_adj)


def _run_meta_attack(
    name: str,
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    budget: int,
    seed: int,
    device: str = "cpu",
    attacker_class=None,
) -> Any:
    attack_type = attacker_class or _load_global_class(name)
    view = _DenseGCNSurrogateView(surrogate, graph).to(device)
    unlabeled = np.union1d(_split_ids(graph, "val"), _split_ids(graph, "test"))
    with _isolated_random_seed(seed):
        # Metattack initializes and repeatedly reinitializes its inner model.
        # Constructing it inside the seeded context makes independent corpus
        # slots both reproducible and seed-dependent.
        attacker = attack_type(
            view,
            nnodes=int(graph.num_nodes),
            feature_shape=tuple(graph.x.shape),
            attack_structure=True,
            attack_features=False,
            device=device,
            with_bias=True,
            train_iters=20,
        ).to(device)
        attacker.attack(
            _graph_features(graph),
            _graph_adjacency(graph),
            _graph_labels(graph),
            _split_ids(graph, "train"),
            unlabeled,
            n_perturbations=int(budget),
            ll_constraint=False,
        )
    return _graph_with_adjacency(graph, attacker.modified_adj)


def run_deeprobust_metattack(graph: Any, **kwargs) -> Any:
    return _run_meta_attack("metattack", graph, **kwargs)


def _initialize_pgd_changes(attacker: Any, *, budget: int, seed: int) -> None:
    """Place PGD at a reproducible, seed-dependent feasible starting point."""

    changes = getattr(attacker, "adj_changes", None)
    if not torch.is_tensor(changes) or changes.ndim != 1:
        raise RuntimeError(
            "DeepRobust PGDAttack does not expose a one-dimensional "
            "adj_changes tensor"
        )
    count = int(budget)
    if count <= 0 or count > int(changes.numel()):
        raise ValueError("PGD random-start budget is outside the feasible edge space")
    generator = np.random.default_rng(int(seed))
    indices = generator.choice(int(changes.numel()), size=count, replace=False)
    values = generator.uniform(0.25, 0.50, size=count).astype(np.float32)
    index_tensor = torch.from_numpy(indices).long().to(changes.device)
    value_tensor = torch.from_numpy(values).to(
        device=changes.device,
        dtype=changes.dtype,
    )
    with torch.no_grad():
        changes.zero_()
        changes[index_tensor] = value_tensor


# randomized discrete projection: exact budget with reproducible sampling
def _select_pgd_flip_indices(scores: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Select exactly ``budget`` lower-triangle flips from PGD scores.

    DeepRobust's stock ``random_sample`` keeps only Bernoulli samples whose
    count is *at most* the requested budget, so its final graph can contain
    fewer changes.  We keep the PGD scores but use a seeded randomized discrete
    projection to select exactly the requested number of feasible flips.
    Seeded Gumbel sampling keeps high-score candidates preferred while making
    independent corpus slots genuinely different.
    """

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = int(budget)
    if count <= 0 or count > int(values.size):
        raise ValueError("PGD discrete projection budget is outside the feasible edge space")
    finite_mask = np.isfinite(values)
    if int(finite_mask.sum()) < count:
        raise ValueError("PGD scores contain too few finite edge candidates")
    finite = np.where(finite_mask, values, -np.inf)
    rng = np.random.default_rng(int(seed))
    pool_size = min(int(finite_mask.sum()), max(count, count * 4))
    tie_jitter = rng.random(values.size) * np.finfo(np.float64).eps
    ranked = finite + tie_jitter
    if pool_size == int(values.size):
        candidates = np.arange(values.size, dtype=np.int64)
    else:
        candidates = np.argpartition(ranked, -pool_size)[-pool_size:]

    candidate_scores = finite[candidates]
    candidate_scores = candidate_scores - candidate_scores.min()
    spread = float(candidate_scores.max())
    if spread > 0.0:
        candidate_scores = candidate_scores / spread
    # Seeded Gumbel-top-k is an exact-budget randomized rounding of the PGD
    # scores.  The score term keeps high-gradient flips preferred, while the
    # random term prevents every independent run collapsing to the same graph.
    uniform = np.clip(
        rng.random(pool_size),
        np.finfo(np.float64).tiny,
        1.0 - np.finfo(np.float64).eps,
    )
    keys = candidate_scores * 4.0 - np.log(-np.log(uniform))
    chosen = np.argpartition(keys, -count)[-count:]
    order = np.argsort(keys[chosen], kind="stable")[::-1]
    return candidates[chosen[order]].astype(np.int64, copy=False)


def _run_pgd_family(
    name: str,
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    budget: int,
    seed: int,
    device: str = "cpu",
    attacker_class=None,
) -> Any:
    attack_type = attacker_class or _load_global_class(name)
    view = _DenseGCNSurrogateView(surrogate, graph).to(device)
    attacker = attack_type(
        model=view,
        nnodes=int(graph.num_nodes),
        loss_type="CE",
        attack_structure=True,
        attack_features=False,
        device=device,
    ).to(device)
    kwargs = {"epochs": 50}
    dense_adjacency = _graph_adjacency(graph).toarray().astype(
        np.float32,
        copy=False,
    )
    with _isolated_random_seed(seed):
        _initialize_pgd_changes(attacker, budget=int(budget), seed=int(seed))
        optimized_scores: np.ndarray | None = None

        def capture_optimized_scores(*_args, **_sample_kwargs):
            nonlocal optimized_scores
            optimized_scores = (
                attacker.adj_changes.detach().cpu().numpy().copy()
            )
            selected = _select_pgd_flip_indices(
                optimized_scores,
                int(budget),
                int(seed),
            )
            projected = torch.zeros_like(attacker.adj_changes)
            projected[torch.from_numpy(selected).to(projected.device)] = 1.0
            with torch.no_grad():
                attacker.adj_changes.copy_(projected)

        # DeepRobust calls random_sample after optimizing the continuous PGD
        # scores.  Its stock Bernoulli routine can return fewer changes and can
        # fail when every sample exceeds the budget.  Replace only that final
        # rounding step; the PGD optimization itself remains unchanged.
        attacker.random_sample = capture_optimized_scores
        attacker.attack(
            _graph_features(graph),
            dense_adjacency,
            _graph_labels(graph),
            _split_ids(graph, "train"),
            n_perturbations=int(budget),
            **kwargs,
        )
        if optimized_scores is None:
            raise RuntimeError("DeepRobust PGDAttack did not expose optimized scores")
        modified = attacker.modified_adj.detach()
    return _graph_with_adjacency(graph, modified)


def run_deeprobust_pgdattack(graph: Any, **kwargs) -> Any:
    return _run_pgd_family("pgdattack", graph, **kwargs)


def run_deeprobust_prbcd(
    graph: Any,
    *,
    surrogate: torch.nn.Module,
    budget: int,
    seed: int,
    device: str = "cpu",
    attacker_class=None,
    epochs: int = 30,
) -> Any:
    """Run scalable PRBCD with an exact edge-budget-derived rate.

    DeepRobust's public PRBCD API accepts a perturbation rate rather than an
    integer.  The half-step numerator avoids floating-point flooring to one
    fewer perturbation while still resolving to the requested integer.
    """

    count = int(budget)
    if count <= 0:
        raise ValueError("PRBCD budget must be positive")
    data = _graph_with_boolean_masks(graph).to(device)
    clean_edges = int(_graph_adjacency(data).nnz // 2)
    if clean_edges <= 0:
        raise ValueError("PRBCD requires at least one undirected edge")
    possible = int(data.num_nodes) * (int(data.num_nodes) - 1) // 2
    search_space = min(possible, max(10_000, count * 20))
    attack_type = attacker_class or _load_prbcd_class()
    view = _PRBCDSurrogateView(surrogate).to(device).eval()
    with _isolated_random_seed(seed):
        torch_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            attacker = attack_type(
                data,
                model=view,
                make_undirected=True,
                search_space_size=search_space,
                max_final_samples=5,
                fine_tune_epochs=min(10, max(1, int(epochs) // 3)),
                epochs=int(epochs),
                do_synchronize=False,
                device=device,
            )
            result = attacker.attack(
                ptb_rate=(count + 0.5) / float(clean_edges)
            )
        finally:
            torch.random.set_rng_state(torch_state)
    if isinstance(result, tuple) and len(result) == 2:
        edge_index, edge_weight = result
    else:
        edge_index = getattr(attacker, "modified_edge_index", None)
        edge_weight = getattr(attacker, "perturbed_edge_weight", None)
    if not torch.is_tensor(edge_index):
        modified_adj = getattr(attacker, "modified_adj", None)
        if modified_adj is None:
            raise RuntimeError("DeepRobust PRBCD did not return a modified graph")
        return _graph_with_adjacency(graph, modified_adj)
    return _graph_with_weighted_edge_index(
        graph,
        edge_index,
        edge_weight,
    )


def _run_deeprobust_centrality(
    graph: Any,
    *,
    budget: int,
    seed: int,
    attack_type: str = "remove",
    attacker_class=None,
) -> Any:
    operation = str(attack_type).lower()
    if operation not in {"add", "remove"}:
        raise ValueError(f"unsupported centrality operation: {attack_type}")
    attack_class = attacker_class or _load_degree_class()
    attacker = attack_class() if callable(attack_class) else attack_class
    attacker.attack(
        _graph_adjacency(graph),
        n_perturbations=int(budget),
        attack_type=operation,
        n_candidates=(
            max(int(budget) * 2, int(budget) + 1)
            if operation == "add"
            else None
        ),
        seed=int(seed),
    )
    return _graph_with_adjacency(graph, attacker.modified_adj)


def run_deeprobust_degree_attack(graph: Any, **kwargs) -> Any:
    """Add/remove edges ranked by DeepRobust's degree-based objective."""

    return _run_deeprobust_centrality(graph, **kwargs)
