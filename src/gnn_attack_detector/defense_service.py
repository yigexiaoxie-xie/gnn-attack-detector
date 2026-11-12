"""Per-input DeepRobust defense benchmarks, isolated from attack detection."""

from __future__ import annotations

from importlib import import_module
import inspect
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from gnn_attack_detector.config import load_config
from gnn_attack_detector.data import split_from_graph
from gnn_attack_detector.defenses.registry import SUPPORTED_DEFENSES, defense_spec
from gnn_attack_detector.device import run_with_cpu_fallback
from gnn_attack_detector.graph_io import load_graph_file
from gnn_attack_detector.probes import _scipy_graph
from gnn_attack_detector.profiles import load_profiles


_PROGNN_MAX_NODES = 5000
_SCIPY_DEFENSES = {
    "gcn", "gcn_jaccard", "gcn_svd", "rgcn", "simpgcn", "gnnguard",
}
_PYG_DATASET_DEFENSES = {"median_gcn", "sgc"}


class _SingleGraphDataset:
    """Minimal dataset shape expected by DeepRobust's SGC and MedianGCN."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getitem__(self, index: int) -> Any:
        if index != 0:
            raise IndexError(index)
        return self.data

    def __len__(self) -> int:
        return 1


def _resolve(root: Path, value: str | Path) -> Path:
    path = (root / Path(value)).resolve()
    path.relative_to(root)
    return path


def defense_profile_status(
    project_root: str | Path,
    profile: Any,
    *,
    clean_ready: bool,
) -> dict[str, dict[str, Any]]:
    """Report whether each per-input benchmark can start for this profile."""

    del project_root
    rows: dict[str, dict[str, Any]] = {}
    for name in SUPPORTED_DEFENSES:
        spec = defense_spec(name)
        ready = bool(clean_ready)
        reason = None
        if name == "prognn" and int(profile.expected_num_nodes) > _PROGNN_MAX_NODES:
            ready = False
            reason = (
                f"ProGNN uses a dense N×N adjacency matrix and is disabled above "
                f"{_PROGNN_MAX_NODES} nodes"
            )
        row = {
            "execution_mode": spec.execution_mode,
            "ready": ready,
            "prepared": False,
        }
        if reason:
            row["reason"] = reason
        rows[name] = row
    return rows


def _load_defense_class(name: str) -> type:
    targets = {
        "gcn": (("deeprobust.graph.defense.gcn", "GCN"),),
        "gcn_jaccard": (
            ("deeprobust.graph.defense.gcn_preprocess", "GCNJaccard"),
        ),
        "gcn_svd": (("deeprobust.graph.defense.gcn_preprocess", "GCNSVD"),),
        "rgcn": (("deeprobust.graph.defense.r_gcn", "RGCN"),),
        "median_gcn": (
            ("deeprobust.graph.defense_pyg.median_gcn", "MedianGCN"),
            ("deeprobust.graph.defense", "MedianGCN"),
        ),
        "simpgcn": (("deeprobust.graph.defense.simpgcn", "SimPGCN"),),
        "prognn": (("deeprobust.graph.defense.prognn", "ProGNN"),),
        "airgnn": (("deeprobust.graph.defense_pyg.airgnn", "AirGNN"),),
        "sgc": (
            ("deeprobust.graph.defense_pyg.sgc", "SGC"),
            ("deeprobust.graph.defense", "SGC"),
        ),
        "gnnguard": (
            ("deeprobust.graph.defense", "GNNGuard"),
            ("deeprobust.graph.defense.gnnguard", "GNNGuard"),
            ("gnnguard", "GNNGuard"),
        ),
    }
    errors: list[str] = []
    for module_name, class_name in targets.get(name, ()):
        try:
            model_class = getattr(import_module(module_name), class_name)
            if isinstance(model_class, type):
                return model_class
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if name == "gnnguard":
        raise RuntimeError(
            "GNNGuard compatible implementation is not installed; the standard "
            "DeepRobust 0.2.9 package does not export GNNGuard"
        )
    detail = errors[-1] if errors else "no adapter is registered"
    raise RuntimeError(f"{name} defense dependency is unavailable ({detail})")


def _call_supported(function: Any, *args: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    supports_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = kwargs if supports_extra else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return function(*args, **selected)


def _model_kwargs(
    graph: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training = config["probes"]["training"]
    air_args = SimpleNamespace(
        dropout=float(training["dropout"]), lambda_amp=0.5, alpha=0.1,
    )
    return {
        "nnodes": int(graph.num_nodes),
        "nfeat": int(graph.num_features),
        "nhid": int(training["hidden_channels"]),
        "nclass": int(graph.y.max().item()) + 1,
        "dropout": float(training["dropout"]),
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "device": device,
        "nlayers": 2,
        "num_layers": 2,
        "with_bn": False,
        "K": 10,
        "args": air_args,
    }


def _new_model(
    name: str,
    graph: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> Any:
    model_class = _load_defense_class(name)
    kwargs = _model_kwargs(graph, config, device)
    signature = inspect.signature(model_class.__init__)
    selected = {key: value for key, value in kwargs.items() if key in signature.parameters}
    model = model_class(**selected)
    return model.to(device) if callable(getattr(model, "to", None)) else model


def _pyg_graph(
    graph: Any,
    split: Mapping[str, list[int]],
    device: torch.device,
) -> Any:
    data = graph.clone().to(device)
    for partition in ("train", "val", "test"):
        mask = torch.zeros(int(graph.num_nodes), dtype=torch.bool, device=device)
        mask[torch.as_tensor(split[partition], dtype=torch.long, device=device)] = True
        setattr(data, f"{partition}_mask", mask)
    return data


def _probabilities(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    tensor = torch.as_tensor(output).detach().cpu().float()
    if tensor.ndim != 2:
        raise RuntimeError("defense model did not return one score row per node")
    exp_sums = tensor.exp().sum(dim=1)
    if bool(torch.allclose(exp_sums, torch.ones_like(exp_sums), atol=1e-3, rtol=1e-3)):
        return tensor.exp()
    sums = tensor.sum(dim=1)
    if bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3)):
        return tensor
    return torch.softmax(tensor, dim=1)


def _node_accuracy(probabilities: torch.Tensor, labels: torch.Tensor, node_ids: Any) -> float:
    indexes = torch.as_tensor(node_ids, dtype=torch.long)
    truth = labels.detach().cpu().long()
    return float((probabilities.argmax(1)[indexes] == truth[indexes]).float().mean())


def _seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_scipy_defense(
    name: str,
    graph: Any,
    split: Mapping[str, list[int]],
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    model = _new_model(name, graph, config, device)
    adjacency, features, labels = _scipy_graph(graph)
    training = config["probes"]["training"]
    options: dict[str, Any] = {
        "train_iters": int(training["epochs"]),
        "patience": int(training["patience"]),
        "verbose": False,
    }
    if name == "gcn_jaccard":
        options["threshold"] = float(training["jaccard_threshold"])
    elif name == "gcn_svd":
        options["k"] = min(50, int(graph.num_nodes) - 1)
    _call_supported(
        model.fit,
        features,
        adjacency,
        labels,
        np.asarray(split["train"]),
        np.asarray(split["val"]),
        **options,
    )
    if callable(getattr(model, "eval", None)):
        model.eval()
    if not callable(getattr(model, "predict", None)):
        raise RuntimeError(f"{name} does not expose a prediction method")
    return _probabilities(model.predict())


def _fit_pyg_defense(
    name: str,
    graph: Any,
    split: Mapping[str, list[int]],
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    model = _new_model(name, graph, config, device)
    data = _pyg_graph(graph, split, device)
    training_input = _SingleGraphDataset(data) if name in _PYG_DATASET_DEFENSES else data
    training = config["probes"]["training"]
    _call_supported(
        model.fit,
        training_input,
        train_iters=int(training["epochs"]),
        patience=int(training["patience"]),
        verbose=False,
    )
    model.data = data
    if callable(getattr(model, "eval", None)):
        model.eval()
    if callable(getattr(model, "predict", None)):
        try:
            output = model.predict()
        except TypeError:
            output = model.predict(data)
    else:
        output = model(data.x, data.edge_index)
    return _probabilities(output)


def _fit_prognn(
    graph: Any,
    split: Mapping[str, list[int]],
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if int(graph.num_nodes) > _PROGNN_MAX_NODES:
        raise RuntimeError(
            f"ProGNN is disabled for graphs above {_PROGNN_MAX_NODES} nodes because it "
            "optimizes a dense N×N adjacency matrix"
        )
    try:
        GCN = getattr(import_module("deeprobust.graph.defense.gcn"), "GCN")
    except Exception as exc:
        raise RuntimeError(f"ProGNN GCN dependency is unavailable: {exc}") from exc
    ProGNN = _load_defense_class("prognn")
    adjacency, features, labels = _scipy_graph(graph)
    features_tensor = torch.as_tensor(features.toarray(), dtype=torch.float32, device=device)
    adjacency_tensor = torch.as_tensor(adjacency.toarray(), dtype=torch.float32, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    training = config["probes"]["training"]
    model = GCN(
        nfeat=int(graph.num_features), nhid=int(training["hidden_channels"]),
        nclass=int(graph.y.max().item()) + 1, dropout=float(training["dropout"]),
        device=device,
    ).to(device)
    args = SimpleNamespace(
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]), lr_adj=0.01,
        alpha=5e-4, beta=1.5, gamma=1.0, lambda_=0.0, phi=0.0,
        epochs=int(training["epochs"]), outer_steps=1, inner_steps=2,
        symmetric=False, only_gcn=False, debug=False, dataset="uploaded",
    )
    prognn = ProGNN(model, args, device)
    prognn.fit(
        features_tensor,
        adjacency_tensor,
        labels_tensor,
        np.asarray(split["train"]),
        np.asarray(split["val"]),
    )
    best_graph = (
        prognn.best_graph
        if prognn.best_graph is not None
        else prognn.estimator.normalize()
    )
    prognn.model.eval()
    with torch.no_grad():
        return _probabilities(prognn.model(features_tensor, best_graph))


def _fit_defense(
    name: str,
    graph: Any,
    split: Mapping[str, list[int]],
    config: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if name == "prognn":
        return _fit_prognn(graph, split, config, device)
    if name in _SCIPY_DEFENSES:
        return _fit_scipy_defense(name, graph, split, config, device)
    return _fit_pyg_defense(name, graph, split, config, device)


def _load_evaluation_context(
    config_path: str | Path,
    graph_path: str | Path,
    project_root: str | Path,
    profile_id: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(config_path)
    profiles = load_profiles(config["profiles"])
    if profile_id not in profiles:
        raise ValueError(f"unknown dataset profile: {profile_id}")
    profile = profiles[profile_id]
    clean = torch.load(
        _resolve(root, profile.clean_reference), map_location="cpu", weights_only=False,
    ).clone().cpu()
    split = split_from_graph(clean)
    graph = load_graph_file(Path(graph_path), clean_graph=clean).clone().cpu()
    if int(graph.num_nodes) != profile.expected_num_nodes:
        raise ValueError("input graph node count does not match the profile")
    if int(graph.num_features) != int(clean.num_features):
        raise ValueError("input graph feature dimension does not match the profile")
    graph.y = clean.y.detach().clone()
    for partition in ("train", "val", "test"):
        setattr(graph, f"{partition}_ids", getattr(clean, f"{partition}_ids").clone())
    graph.dataset_source = "deeprobust"
    graph.dataset_setting = getattr(
        clean, "dataset_setting", getattr(profile, "setting", "nettack")
    )
    return {
        "root": root,
        "config": config,
        "profile": profile,
        "clean": clean,
        "graph": graph,
        "split": split,
    }


def _per_input_predictions(
    context: Mapping[str, Any],
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    clean = context["clean"]
    graph = context["graph"]
    config = context["config"]
    split = context["split"]
    seed = int(config.get("seed", 0))

    def operation(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, str]:
        _seed_training(seed)
        clean_output = _fit_defense(name, clean, split, config, device)
        _seed_training(seed)
        attacked_output = _fit_defense(name, graph, split, config, device)
        return clean_output, attacked_output, device.type

    return run_with_cpu_fallback(
        operation,
        requested=str(config["runtime"]["device"]),
        label=f"defense evaluation {context['profile'].identifier}/{name}",
    )


def evaluate_robust_probe(
    config_path: str | Path,
    graph_path: str | Path,
    *,
    project_root: str | Path,
    profile_id: str,
    defense_name: str,
) -> dict[str, Any]:
    """Retrain one defense on clean/input graphs using the DeepRobust split."""

    name = str(defense_name).lower()
    spec = defense_spec(name)
    context = _load_evaluation_context(
        config_path, graph_path, project_root, profile_id,
    )
    clean_output, attacked_output, device = _per_input_predictions(context, name)
    clean = context["clean"]
    test_ids = context["split"]["test"]
    clean_accuracy = _node_accuracy(clean_output, clean.y, test_ids)
    attacked_accuracy = _node_accuracy(attacked_output, clean.y, test_ids)
    split_sizes = {
        partition: len(context["split"][partition])
        for partition in ("train", "val", "test")
    }
    return {
        "status": "ok",
        "profile_id": profile_id,
        "defense": name,
        "execution_mode": spec.execution_mode,
        "training_scope": "per_input_graph",
        "split_source": "deeprobust",
        "split_sizes": split_sizes,
        "device": device,
        "evaluated_nodes": len(test_ids),
        "clean_accuracy": clean_accuracy,
        "input_graph_accuracy": attacked_accuracy,
        "accuracy_drop": clean_accuracy - attacked_accuracy,
    }
