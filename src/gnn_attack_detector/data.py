"""Known graph dataset loading and reproducible node-label splits."""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from gnn_attack_detector.profiles import load_profiles


def _structural_features(adjacency) -> np.ndarray:
    """Return portable features for topology-only datasets such as Polblogs."""

    degrees = np.asarray(adjacency.sum(axis=1), dtype=np.float32).reshape(-1)
    maximum_degree = float(degrees.max()) if degrees.size else 0.0
    normalized = degrees / maximum_degree if maximum_degree > 0.0 else degrees
    return np.column_stack(
        (normalized, np.ones(int(adjacency.shape[0]), dtype=np.float32))
    ).astype(np.float32, copy=False)


def graph_has_self_loops(graph) -> bool:
    edges = graph.edge_index
    return bool(edges.numel() and torch.any(edges[0] == edges[1]).item())


def load_planetoid(root: str | Path, name: str):
    """Load a Planetoid graph, downloading it under ``root`` when absent."""

    dataset = Planetoid(root=str(Path(root)), name=name, split="public")
    return dataset[0]


def _load_deeprobust_dataset_class():
    try:
        from deeprobust.graph.data import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust is required; run scripts\\setup_windows_cpu.bat first"
        ) from exc
    return Dataset


def load_deeprobust_graph(
    root: str | Path,
    name: str,
    *,
    setting: str = "nettack",
    seed: int = 42,
    expected_num_nodes: int | None = None,
    dataset_class=None,
):
    """Load a DeepRobust citation graph and convert it to one PyG graph.

    The ``nettack`` setting selects the largest connected component and supplies
    the official 10%/10%/80% node split used by DeepRobust.
    """

    dataset_root = Path(root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    loader = dataset_class or _load_deeprobust_dataset_class()
    dataset = loader(
        root=str(dataset_root),
        name=str(name).lower(),
        setting=str(setting).lower(),
        seed=int(seed),
    )
    adjacency = dataset.adj.tocsr(copy=True)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency = adjacency.tocoo()
    features = dataset.features
    if features is None:
        feature_array = _structural_features(adjacency)
    else:
        feature_array = (
            features.toarray() if hasattr(features, "toarray") else np.asarray(features)
        )
        if feature_array.ndim == 2 and int(feature_array.shape[1]) == 0:
            feature_array = _structural_features(adjacency)
    labels = np.asarray(dataset.labels)
    num_nodes = int(adjacency.shape[0])
    if expected_num_nodes is not None and num_nodes != int(expected_num_nodes):
        raise ValueError(
            f"expected {int(expected_num_nodes)} nodes; loaded {num_nodes}"
        )
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("DeepRobust adjacency matrix must be square")
    if feature_array.ndim != 2 or int(feature_array.shape[0]) != num_nodes:
        raise ValueError("DeepRobust features must contain one row per node")
    if labels.ndim != 1 or int(labels.shape[0]) != num_nodes:
        raise ValueError("DeepRobust labels must contain one value per node")

    edge_index = torch.from_numpy(
        np.vstack((adjacency.row, adjacency.col)).astype(np.int64, copy=False)
    ).long().contiguous()
    graph = Data(
        x=torch.from_numpy(
            np.asarray(feature_array, dtype=np.float32)
        ).float().contiguous(),
        edge_index=edge_index,
        y=torch.from_numpy(np.asarray(labels, dtype=np.int64)).long().contiguous(),
    )
    graph.train_ids = torch.as_tensor(dataset.idx_train, dtype=torch.long)
    graph.val_ids = torch.as_tensor(dataset.idx_val, dtype=torch.long)
    graph.test_ids = torch.as_tensor(dataset.idx_test, dtype=torch.long)
    graph.dataset_source = "deeprobust"
    graph.dataset_setting = str(setting).lower()
    return graph


def split_from_graph(graph) -> dict[str, list[int]]:
    """Return the dataset-provided node split attached by the known loader."""

    required = ("train_ids", "val_ids", "test_ids")
    if not all(hasattr(graph, name) for name in required):
        raise ValueError("graph does not contain a dataset-provided node split")
    return {
        "train": [int(value) for value in graph.train_ids.tolist()],
        "val": [int(value) for value in graph.val_ids.tolist()],
        "test": [int(value) for value in graph.test_ids.tolist()],
    }


def load_graph_for_config(project_root: str | Path, config: dict):
    """Load the active known-dataset profile."""

    profile = load_profiles(config["profiles"])[str(config["active_profile"])]
    source = profile.source
    if source != "deeprobust":
        raise ValueError(f"unsupported profile source: {source}")
    return load_deeprobust_graph(
        Path(project_root) / str(config["paths"]["raw_data"]) / "deeprobust",
        profile.name,
        setting=profile.setting,
        seed=int(config["seed"]),
        expected_num_nodes=profile.expected_num_nodes,
    )


def split_node_ids(
    num_nodes: int,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, list[int]]:
    """Return deterministic, disjoint train/val/test node identifiers."""

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")

    generator = np.random.default_rng(seed)
    node_ids = generator.permutation(num_nodes)
    train_count = int(num_nodes * ratios[0])
    val_count = int(num_nodes * ratios[1])
    val_end = train_count + val_count

    return {
        "train": node_ids[:train_count].tolist(),
        "val": node_ids[train_count:val_end].tolist(),
        "test": node_ids[val_end:].tolist(),
    }
