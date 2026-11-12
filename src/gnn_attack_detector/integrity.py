"""Stable integrity fingerprints and fixed-GCN behavior summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tensor(tensor: torch.Tensor) -> str:
    """Return a digest covering tensor type, shape, and exact values."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def graph_fingerprint(data: Any) -> dict[str, str]:
    """Fingerprint every tensor that defines a supported complete graph."""

    pieces = {
        name: hash_tensor(getattr(data, name))
        for name in ("x", "edge_index", "y")
    }
    combined = hashlib.sha256(
        json.dumps(pieces, sort_keys=True).encode("ascii")
    ).hexdigest()
    return {**pieces, "combined": combined}


def modification_signature(
    *,
    target_nodes: list[int] | tuple[int, ...],
    added_edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    removed_edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> str:
    """Hash canonical attack locations independently of input ordering."""

    def canonical_edges(values: Any) -> list[list[int]]:
        return [
            [left, right]
            for left, right in sorted(
                {
                    tuple(sorted((int(pair[0]), int(pair[1]))))
                    for pair in values
                    if int(pair[0]) != int(pair[1])
                }
            )
        ]

    payload = {
        "target_nodes": sorted({int(node) for node in target_nodes}),
        "added_edges": canonical_edges(added_edges),
        "removed_edges": canonical_edges(removed_edges),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def evaluate_model(model: torch.nn.Module, data: Any) -> dict[str, Any]:
    """Evaluate the fixed GCN and return JSON-serializable node signals."""

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(data.x, data.edge_index)
        probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    confidences = probabilities.max(dim=-1).values
    entropies = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    accuracy = (predictions == data.y).float().mean()
    return {
        "predictions": predictions.detach().cpu().tolist(),
        "probabilities": probabilities.detach().cpu().tolist(),
        "confidences": confidences.detach().cpu().tolist(),
        "entropies": entropies.detach().cpu().tolist(),
        "embeddings": embeddings.detach().cpu().tolist(),
        "accuracy": float(accuracy.item()),
    }
