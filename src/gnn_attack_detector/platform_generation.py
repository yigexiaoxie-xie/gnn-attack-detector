"""Synchronous, isolated attack generation for the web platform."""

from __future__ import annotations

import math
from pathlib import Path
import random
import secrets
from typing import Any, Callable, Iterable
from uuid import uuid4

import torch

from gnn_attack_detector.graph_attacks import (
    AttackCandidateRejected,
    CompleteGraphAttackOutcome,
    edge_difference,
    eligible_attack_targets,
    run_complete_graph_attack,
)
from gnn_attack_detector.graph_io import graph_to_npz_bytes
from gnn_attack_detector.integrity import graph_fingerprint
from gnn_attack_detector.attacks.registry import (
    LIVE_GENERATION_ATTACKS,
    SUPPORTED_ATTACKS,
    TARGETED_ATTACKS,
)


def _random_seed() -> int:
    return secrets.randbelow(2_000_000_000) + 1


def _validate_selection(
    *,
    attack_name: str,
    strength: float | int,
    count: int,
    structural_rates: tuple[float, ...],
    targeted_target_counts: tuple[int, ...],
    max_attempts: int,
) -> tuple[str, float | int, str]:
    name = str(attack_name).lower()
    if name not in SUPPORTED_ATTACKS:
        raise ValueError(f"unsupported attack method: {attack_name}")
    if name not in LIVE_GENERATION_ATTACKS:
        raise ValueError(f"{name} is not available for live generation")
    if int(count) < 1 or int(count) > 10:
        raise ValueError("generation count must be between 1 and 10")
    if int(max_attempts) < 1 or int(max_attempts) > 100:
        raise ValueError("max_attempts must be between 1 and 100")
    if name in TARGETED_ATTACKS:
        value: float | int = int(strength)
        if value not in {int(item) for item in targeted_target_counts}:
            raise ValueError(f"{name} target count is not configured")
        return name, value, "target_nodes"
    value = float(strength)
    if not any(
        math.isclose(value, float(item), rel_tol=0.0, abs_tol=1e-12)
        for item in structural_rates
    ):
        raise ValueError(f"{name} strength is not configured")
    return name, value, "edge_rate"


def _error_row(
    *,
    sample_number: int,
    dataset_name: str,
    attack_name: str,
    strength: float | int,
    strength_unit: str,
    error: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "source_name": f"{attack_name}-generated-{sample_number:02d}",
        "dataset": dataset_name,
        "attack_name": attack_name,
        "strength": strength,
        "strength_unit": strength_unit,
        "actual_modifications": None,
        "seed": None,
        "target_nodes": [],
        "graph_fingerprint": None,
        "pt_path": None,
        "npz_path": None,
        "error": error,
    }


def generate_attack_batch(
    *,
    clean_graph: Any,
    surrogate: torch.nn.Module,
    dataset_name: str,
    attack_name: str,
    strength: float | int,
    count: int,
    output_root: str | Path,
    structural_rates: tuple[float, ...],
    targeted_target_counts: tuple[int, ...],
    targeted_local_budget: int,
    minimum_target_degree: int,
    max_attempts: int = 30,
    seed_factory: Callable[[], int] = _random_seed,
    attack_runner: Callable[..., CompleteGraphAttackOutcome] = run_complete_graph_attack,
    excluded_fingerprints: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Generate independent complete graphs without modifying the formal corpus."""

    name, selected_strength, strength_unit = _validate_selection(
        attack_name=attack_name,
        strength=strength,
        count=count,
        structural_rates=tuple(structural_rates),
        targeted_target_counts=tuple(targeted_target_counts),
        max_attempts=max_attempts,
    )
    dataset = str(dataset_name)
    if not dataset or Path(dataset).name != dataset or dataset in {".", ".."}:
        raise ValueError("dataset name is not safe for generated artifacts")
    local_budget = int(targeted_local_budget)
    if local_budget <= 0:
        raise ValueError("targeted local budget must be positive")
    generated_root = Path(output_root).resolve() / dataset / name
    generated_root.mkdir(parents=True, exist_ok=True)

    eligible: list[int] = []
    if name in TARGETED_ATTACKS:
        eligible = eligible_attack_targets(
            clean_graph,
            surrogate,
            minimum_degree=max(int(minimum_target_degree), local_budget),
            required_count=int(selected_strength),
        )

    seen_seeds: set[int] = set()
    seen_fingerprints: set[str] = set(excluded_fingerprints)
    rows: list[dict[str, Any]] = []
    for sample_number in range(1, int(count) + 1):
        accepted: tuple[CompleteGraphAttackOutcome, int, tuple[int, ...], str] | None = None
        terminal_error: str | None = None
        for _attempt in range(int(max_attempts)):
            seed = int(seed_factory())
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            targets: tuple[int, ...] = ()
            if name in TARGETED_ATTACKS:
                targets = tuple(
                    sorted(
                        random.Random(seed).sample(
                            eligible,
                            int(selected_strength),
                        )
                    )
                )
            try:
                runner_arguments: dict[str, Any] = {
                    "graph": clean_graph.clone(),
                    "surrogate": surrogate,
                    "attack_name": name,
                    "strength": selected_strength,
                    "target_nodes": targets,
                    "local_budget": local_budget,
                    "seed": seed,
                }
                outcome = attack_runner(**runner_arguments)
            except AttackCandidateRejected:
                continue
            except Exception as exc:
                terminal_error = str(exc)
                break
            clean_nodes = int(clean_graph.num_nodes)
            if int(outcome.graph.num_nodes) != clean_nodes:
                terminal_error = "generated attack changed the node count illegally"
                break
            if not torch.equal(outcome.graph.y[:clean_nodes], clean_graph.y):
                terminal_error = "generated attack changed original node labels"
                break
            added, removed = edge_difference(clean_graph, outcome.graph)
            if added != set(outcome.added_edges) or removed != set(outcome.removed_edges):
                terminal_error = "generated attack edge audit does not match graph"
                break
            if not added and not removed and not outcome.changed_features:
                continue
            fingerprint = graph_fingerprint(outcome.graph)["combined"]
            if fingerprint in seen_fingerprints:
                continue
            accepted = outcome, seed, targets, fingerprint
            break
        if accepted is None:
            message = terminal_error or (
                f"could not generate a unique sample after {int(max_attempts)} attempts"
            )
            rows.append(
                _error_row(
                    sample_number=sample_number,
                    dataset_name=dataset,
                    attack_name=name,
                    strength=selected_strength,
                    strength_unit=strength_unit,
                    error=message,
                )
            )
            continue

        outcome, seed, targets, fingerprint = accepted
        seen_fingerprints.add(fingerprint)
        token = uuid4().hex
        pt_path = generated_root / f"{token}.pt"
        npz_path = generated_root / f"{token}.npz"
        trusted_payload = {
            "x": outcome.graph.x.detach().cpu().clone(),
            "edge_index": outcome.graph.edge_index.detach().cpu().clone(),
            "y": outcome.graph.y.detach().cpu().clone(),
        }
        torch.save(trusted_payload, pt_path)
        npz_path.write_bytes(graph_to_npz_bytes(outcome.graph))
        actual_modifications = outcome.actual_modifications
        print(
            f"[GENERATED] platform {sample_number}/{int(count)} {name} "
            f"strength={selected_strength} seed={seed} "
            f"changes={actual_modifications} saved={pt_path}",
            flush=True,
        )
        rows.append(
            {
                "status": "ok",
                "source_name": pt_path.name,
                "dataset": dataset,
                "attack_name": name,
                "strength": selected_strength,
                "strength_unit": strength_unit,
                "actual_modifications": actual_modifications,
                "seed": seed,
                "target_nodes": list(targets),
                "graph_fingerprint": fingerprint,
                "pt_path": str(pt_path),
                "npz_path": str(npz_path),
                "error": None,
            }
        )
    return rows
