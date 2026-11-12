"""Prepare known data and the attack surrogate; defenses train per input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from gnn_attack_detector.config import load_config_source
from gnn_attack_detector.data import (
    graph_has_self_loops,
    load_graph_for_config,
    split_from_graph,
)
from gnn_attack_detector.profiles import load_profiles
from gnn_attack_detector.victim import load_victim_checkpoint, train_victim, validate_victim_graph_contract


def _resolve(root: Path, value: str | Path) -> Path:
    path = (root / Path(value)).resolve()
    path.relative_to(root)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_project(
    config_path: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path = Path("."),
    include_defenses: bool = False,
) -> dict[str, Any]:
    """Freeze the known graph and train only the attack-generation surrogate."""

    root = Path(project_root).resolve()
    config = load_config_source(config_path)
    profile = load_profiles(config["profiles"])[str(config["active_profile"])]
    clean_path = _resolve(root, profile.clean_reference)
    probe_dir = _resolve(root, profile.probe_dir)
    report_path = _resolve(
        root,
        Path(config["paths"]["reports"]) / f"{profile.identifier}-profile.json",
    )
    # ``include_defenses`` remains accepted for CLI compatibility. Defense
    # benchmarks no longer have persistent checkpoints.
    names = ("gcn",)
    probe_paths = [probe_dir / f"{name}.pt" for name in names]
    if clean_path.is_file() and all(path.is_file() for path in probe_paths):
        try:
            graph = torch.load(clean_path, map_location="cpu", weights_only=False)
            if int(graph.num_nodes) != profile.expected_num_nodes:
                raise RuntimeError("saved clean graph node count does not match profile")
            if graph_has_self_loops(graph):
                raise RuntimeError("saved clean graph contains self-loops")
            _, metadata = load_victim_checkpoint(probe_dir / "gcn.pt")
            validate_victim_graph_contract(metadata, graph, profile.identifier)
            split = split_from_graph(graph)
            report = {
                "status": "ok",
                "reused": True,
                "profile_id": profile.identifier,
                "dataset": profile.name,
                "source": profile.source,
                "setting": profile.setting,
                "seed": int(config["seed"]),
                "num_nodes": int(graph.num_nodes),
                "num_edges": int(graph.num_edges),
                "num_features": int(graph.num_features),
                "num_classes": int(graph.y.max().item()) + 1,
                "split_sizes": {name: len(values) for name, values in split.items()},
                "clean_reference": clean_path.relative_to(root).as_posix(),
                "probe_dir": probe_dir.relative_to(root).as_posix(),
                "probes": {
                    name: {"status": "reused"}
                    for name in names
                },
            }
            _write_json(report_path, report)
            report["report_path"] = report_path.relative_to(root).as_posix()
            return report
        except Exception:
            # A stale or partial profile is rebuilt from the authoritative source.
            pass

    # Missing/stale models must not silently replace the graph backing saved attacks.
    graph = (torch.load(clean_path, map_location="cpu", weights_only=False) if clean_path.is_file()
             else load_graph_for_config(root, config)).clone().cpu()
    if int(graph.num_nodes) != profile.expected_num_nodes or graph_has_self_loops(graph):
        raise RuntimeError("existing source graph is incompatible; preserve attacks and investigate before replacing it")
    split = split_from_graph(graph)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    if not clean_path.is_file():
        torch.save(graph, clean_path)
    probe_dir.mkdir(parents=True, exist_ok=True)
    _, gcn_metrics = train_victim(graph, split, config, probe_dir / "gcn.pt")
    metrics = {"gcn": gcn_metrics}

    report = {
        "status": "ok",
        "reused": False,
        "profile_id": profile.identifier,
        "dataset": profile.name,
        "source": profile.source,
        "setting": profile.setting,
        "seed": int(config["seed"]),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "num_features": int(graph.num_features),
        "num_classes": int(graph.y.max().item()) + 1,
        "split_sizes": {name: len(values) for name, values in split.items()},
        "clean_reference": clean_path.relative_to(root).as_posix(),
        "probe_dir": probe_dir.relative_to(root).as_posix(),
        "probes": metrics,
    }
    _write_json(report_path, report)
    report["report_path"] = report_path.relative_to(root).as_posix()
    return report
