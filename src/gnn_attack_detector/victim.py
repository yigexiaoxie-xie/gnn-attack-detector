"""Victim GCN training, checkpointing, and reload support."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torch_geometric.nn import GCNConv

from gnn_attack_detector.device import resolve_device, run_with_cpu_fallback
from gnn_attack_detector.integrity import graph_fingerprint


class VictimGCN(nn.Module):
    """Two-layer GCN used as the fixed model attacked by later experiments."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.conv1(x, edge_index).relu()
        hidden = functional.dropout(
            embeddings,
            p=self.dropout,
            training=self.training,
        )
        logits = self.conv2(hidden, edge_index)
        return logits, embeddings


def _accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    node_ids: list[int],
) -> float:
    indexes = torch.tensor(node_ids, dtype=torch.long, device=labels.device)
    return float((predictions[indexes] == labels[indexes]).float().mean().item())


def _train_victim_on_device(
    data: Any,
    split: dict[str, list[int]],
    config: dict[str, Any],
    output_path: str | Path,
    device: torch.device,
) -> tuple[VictimGCN, dict[str, float | int | str]]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    graph = data.to(device)
    model_config = config["probes"]["training"]
    architecture = {
        "in_channels": int(graph.num_features),
        "hidden_channels": int(model_config["hidden_channels"]),
        "out_channels": int(graph.y.max().item()) + 1,
        "dropout": float(model_config["dropout"]),
    }
    model = VictimGCN(**architecture).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    train_ids = torch.tensor(split["train"], dtype=torch.long, device=device)
    epochs = int(model_config["epochs"])
    final_loss = 0.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(graph.x, graph.edge_index)
        loss = functional.cross_entropy(logits[train_ids], graph.y[train_ids])
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        logits, _ = model(graph.x, graph.edge_index)
        predictions = logits.argmax(dim=1)

    metrics: dict[str, float | int | str] = {
        "train_accuracy": _accuracy(predictions, graph.y, split["train"]),
        "val_accuracy": _accuracy(predictions, graph.y, split["val"]),
        "test_accuracy": _accuracy(predictions, graph.y, split["test"]),
        "final_loss": final_loss,
        "epochs": epochs,
        "device": str(device),
    }
    checkpoint = {
        "format_version": 2,
        "dataset_name": str(config["active_profile"]),
        "num_nodes": int(graph.num_nodes),
        "dataset_graph_hash": graph_fingerprint(graph)["combined"],
        "seed": seed,
        "architecture": architecture,
        "training_config": dict(model_config),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    checkpoint_path = Path(output_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return model, metrics


def train_victim(
    data: Any,
    split: dict[str, list[int]],
    config: dict[str, Any],
    output_path: str | Path,
) -> tuple[VictimGCN, dict[str, float | int | str]]:
    """Train the fixed GCN and persist a self-describing checkpoint."""

    return run_with_cpu_fallback(
        lambda device: _train_victim_on_device(
            data, split, config, output_path, device
        ),
        requested=str(config.get("runtime", {}).get("device", "auto")),
        label="victim training",
    )


def load_victim_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[VictimGCN, dict[str, Any]]:
    """Reload a victim GCN without requiring the original training process."""

    resolved_device = resolve_device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=resolved_device)
    model = VictimGCN(**checkpoint["architecture"]).to(resolved_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        key: value for key, value in checkpoint.items() if key != "state_dict"
    }
    return model, metadata


def validate_victim_graph_contract(
    metadata: dict[str, Any],
    graph: Any,
    dataset_name: str,
) -> None:
    """Reject a victim checkpoint trained for another dataset artifact."""

    if metadata.get("format_version") != 2:
        raise RuntimeError(
            "victim checkpoint is obsolete; rerun the prepare command"
        )
    if metadata.get("dataset_name") != str(dataset_name):
        raise RuntimeError("victim checkpoint dataset name does not match")
    if int(metadata.get("num_nodes", -1)) != int(graph.num_nodes):
        raise RuntimeError("victim checkpoint node count does not match the graph")
    expected_hash = graph_fingerprint(graph)["combined"]
    if metadata.get("dataset_graph_hash") != expected_hash:
        raise RuntimeError("victim checkpoint graph fingerprint does not match")
