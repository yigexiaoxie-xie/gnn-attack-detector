"""Train/load optional per-profile defense benchmarks, separate from detection."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import io
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import scipy.sparse as sp
import torch

from gnn_attack_detector.integrity import graph_fingerprint
from gnn_attack_detector.victim import (
    VictimGCN,
    load_victim_checkpoint,
    train_victim,
    validate_victim_graph_contract,
)


ONLINE_PROBE_NAMES = ("gcn", "gcn_jaccard", "simpgcn")


@dataclass(frozen=True)
class ProbePrediction:
    probabilities: torch.Tensor


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Contain SimPGCN's upstream ``saved_knn`` cache inside the profile."""

    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _scipy_graph(graph: Any) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    edge_index = graph.edge_index.detach().cpu().long().numpy()
    values = np.ones(edge_index.shape[1], dtype=np.float32)
    adjacency = sp.csr_matrix(
        (values, (edge_index[0], edge_index[1])),
        shape=(int(graph.num_nodes), int(graph.num_nodes)),
        dtype=np.float32,
    )
    adjacency.sum_duplicates()
    if adjacency.nnz:
        adjacency.data[:] = 1.0
    features = sp.csr_matrix(graph.x.detach().cpu().float().numpy())
    labels = graph.y.detach().cpu().long().numpy()
    return adjacency, features, labels


def _load_defense_classes():
    try:
        from deeprobust.graph.defense import GCNJaccard, SimPGCN
    except ImportError as exc:
        raise RuntimeError(
            "DeepRobust defenses are unavailable; run scripts\\setup_windows_cpu.bat"
        ) from exc
    return GCNJaccard, SimPGCN


def _robust_payload(profile_id: str, graph: Any, name: str, model: Any) -> dict[str, Any]:
    model = model.cpu().eval()
    model.device = torch.device("cpu")
    for attribute in ("adj_knn", "identity", "features", "adj_norm", "labels"):
        value = getattr(model, attribute, None)
        if torch.is_tensor(value):
            setattr(model, attribute, value.detach().cpu())
    return {
        "format_version": 1,
        "profile_id": str(profile_id),
        "probe_name": str(name),
        "num_nodes": int(graph.num_nodes),
        "graph_hash": graph_fingerprint(graph)["combined"],
        "model": model,
    }


def _node_accuracy(output: torch.Tensor, labels: np.ndarray, node_ids: list[int]) -> float:
    indexes = torch.as_tensor(node_ids, dtype=torch.long, device=output.device)
    truth = torch.as_tensor(labels, dtype=torch.long, device=output.device)
    return float((output.argmax(dim=1)[indexes] == truth[indexes]).float().mean())


def train_profile_probes(
    graph: Any,
    split: Mapping[str, list[int]],
    config: Mapping[str, Any],
    profile_id: str,
    probe_dir: str | Path,
) -> dict[str, dict[str, float | int | str]]:
    """Train one ordinary GCN and two real DeepRobust robust probes."""

    destination = Path(probe_dir)
    destination.mkdir(parents=True, exist_ok=True)
    training = config["probes"]["training"]
    gcn, gcn_metrics = train_victim(graph, dict(split), dict(config), destination / "gcn.pt")
    del gcn

    adjacency, features, labels = _scipy_graph(graph)
    GCNJaccard, SimPGCN = _load_defense_classes()
    requested_device = str(config["runtime"]["device"])
    device = "cuda" if (
        requested_device == "cuda"
        or (requested_device == "auto" and torch.cuda.is_available())
    ) else "cpu"
    common = {
        "nfeat": int(graph.num_features),
        "nhid": int(training["hidden_channels"]),
        "nclass": int(graph.y.max().item()) + 1,
        "dropout": float(training["dropout"]),
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "device": device,
    }
    jaccard = GCNJaccard(**common).to(device)
    jaccard.fit(
        features,
        adjacency,
        labels,
        split["train"],
        split["val"],
        threshold=float(training["jaccard_threshold"]),
        train_iters=int(training["epochs"]),
        patience=int(training["patience"]),
        verbose=False,
    )
    jaccard_accuracy = _node_accuracy(jaccard.predict(), labels, split["test"])
    torch.save(
        _robust_payload(profile_id, graph, "gcn_jaccard", jaccard),
        destination / "gcn_jaccard.pt",
    )

    simpgcn = SimPGCN(nnodes=int(graph.num_nodes), **common).to(device)
    with _working_directory(destination):
        simpgcn.fit(
            features,
            adjacency,
            labels,
            split["train"],
            split["val"],
            train_iters=int(training["epochs"]),
            patience=int(training["patience"]),
            verbose=False,
        )
        simpgcn_accuracy = _node_accuracy(simpgcn.predict(), labels, split["test"])
    torch.save(
        _robust_payload(profile_id, graph, "simpgcn", simpgcn),
        destination / "simpgcn.pt",
    )
    return {
        "gcn": dict(gcn_metrics),
        "gcn_jaccard": {
            "status": "trained",
            "backend": "deeprobust",
            "test_accuracy": jaccard_accuracy,
        },
        "simpgcn": {
            "status": "trained",
            "backend": "deeprobust",
            "test_accuracy": simpgcn_accuracy,
        },
    }


def load_profile_probes(
    probe_dir: str | Path,
    graph: Any,
    profile_id: str,
) -> dict[str, Any]:
    """Load the fixed probe order and reject artifacts from another profile."""

    source = Path(probe_dir)
    gcn, metadata = load_victim_checkpoint(source / "gcn.pt", device="cpu")
    validate_victim_graph_contract(metadata, graph, profile_id)
    probes: dict[str, Any] = {"gcn": gcn.cpu().eval()}
    expected_hash = graph_fingerprint(graph)["combined"]
    for name in ONLINE_PROBE_NAMES[1:]:
        payload = torch.load(source / f"{name}.pt", map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise RuntimeError(f"unsupported robust probe artifact: {name}")
        if payload.get("profile_id") != profile_id:
            raise RuntimeError(f"{name} belongs to another dataset profile")
        if payload.get("graph_hash") != expected_hash:
            raise RuntimeError(f"{name} clean graph fingerprint does not match")
        if int(payload.get("num_nodes", -1)) != int(graph.num_nodes):
            raise RuntimeError(f"{name} node count does not match the profile")
        model = payload["model"].cpu().eval()
        model.device = torch.device("cpu")
        for attribute in ("adj_knn", "identity", "features", "adj_norm", "labels"):
            value = getattr(model, attribute, None)
            if torch.is_tensor(value):
                setattr(model, attribute, value.detach().cpu())
        probes[name] = model
    return probes


def predict_probe(name: str, model: Any, graph: Any, *, cache_dir: str | Path | None = None) -> ProbePrediction:
    """Return a common prediction representation for PyG and DeepRobust probes."""

    if isinstance(model, VictimGCN) or not callable(getattr(model, "predict", None)):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model(graph.x.detach().cpu(), graph.edge_index.detach().cpu())
            if isinstance(output, tuple) and len(output) == 2:
                logits, _ = output
            else:
                logits = output
            probabilities = torch.softmax(logits, dim=-1)
        if was_training:
            model.train()
        return ProbePrediction(probabilities.detach().cpu())

    adjacency, features, _ = _scipy_graph(graph)
    destination = Path(cache_dir) if cache_dir is not None else Path.cwd()
    with _working_directory(destination), redirect_stdout(io.StringIO()), torch.no_grad():
        log_probabilities = model.predict(features, adjacency)
    probabilities = log_probabilities.detach().cpu().float().exp()
    return ProbePrediction(probabilities)
