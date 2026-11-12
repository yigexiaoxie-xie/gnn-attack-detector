"""Ordered, resumable orchestration for selected profiles and attacks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from gnn_attack_detector.attacks.registry import SUPPORTED_ATTACKS
from gnn_attack_detector.config import load_config_source
from gnn_attack_detector.run_selection import RunSelection, load_run_selection


Operation = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class WorkflowOperations:
    prepare_profile: Operation
    prepare_trainval: Operation
    train_universal: Operation
    prepare_test: Operation
    evaluate_universal: Operation


def _default_operations() -> WorkflowOperations:
    # Heavy torch/DeepRobust imports stay out of selection previews and tests.
    from gnn_attack_detector.pipeline import (
        evaluate_detector,
        generate_detector_test_data,
        prepare_detector_data,
        train_detector,
    )
    from gnn_attack_detector.preparation import prepare_project

    return WorkflowOperations(
        prepare_profile=prepare_project,
        prepare_trainval=prepare_detector_data,
        train_universal=train_detector,
        prepare_test=generate_detector_test_data,
        evaluate_universal=evaluate_detector,
    )


def selected_profile_config(
    base_config: Mapping[str, Any],
    selection: RunSelection,
    profile_id: str,
) -> dict[str, Any]:
    """Return one validated in-memory profile view; never write extra YAML."""

    selected = deepcopy(dict(base_config))
    selected["active_profile"] = str(profile_id)
    selected["attacks"]["protocols"] = list(selection.attacks)
    selected["attacks"]["require_all_protocols"] = False
    return load_config_source(selected)


def _error(errors: list[dict[str, str]], profile: str, phase: str, exc: Exception) -> None:
    errors.append(
        {
            "profile": str(profile),
            "phase": str(phase),
            "error": f"{type(exc).__name__}: {exc}",
        }
    )


def _sum_mapping(payload: Mapping[str, Any], key: str) -> int:
    values = payload.get(key, {})
    return sum(int(value) for value in values.values()) if isinstance(values, Mapping) else 0


def run_selected_workflow(
    config_source: str | Path | Mapping[str, Any],
    run_config: str | Path,
    *,
    project_root: str | Path = Path("."),
    operations: WorkflowOperations | None = None,
) -> dict[str, Any]:
    """Run prepare -> train/val -> one train -> test -> one evaluation."""

    root = Path(project_root).resolve()
    base = load_config_source(config_source)
    selection = load_run_selection(
        run_config,
        known_profiles=tuple(base["profiles"]),
        known_attacks=SUPPORTED_ATTACKS,
    )
    selected_configs = {
        profile_id: selected_profile_config(base, selection, profile_id)
        for profile_id in selection.profiles
    }
    active_operations = operations or _default_operations()
    errors: list[dict[str, str]] = []
    results: dict[str, Any] = {
        "prepare": {},
        "train_validation": {},
        "test": {},
    }
    generated = 0
    reused = 0
    prepared_ok: set[str] = set()

    for profile_id in selection.profiles:
        try:
            results["prepare"][profile_id] = active_operations.prepare_profile(
                selected_configs[profile_id],
                project_root=root,
            )
            prepared_ok.add(profile_id)
        except Exception as exc:
            _error(errors, profile_id, "prepare", exc)

    needs_training = any(
        partition in {"train", "val"} for partition in selection.partitions
    )
    if needs_training:
        for profile_id in selection.profiles:
            if profile_id not in prepared_ok:
                continue
            try:
                payload = active_operations.prepare_trainval(
                    selected_configs[profile_id],
                    project_root=root,
                )
                results["train_validation"][profile_id] = payload
                generated += _sum_mapping(payload, "generated_by_partition")
                reused += _sum_mapping(payload, "reused_by_partition")
            except Exception as exc:
                _error(errors, profile_id, "train_validation", exc)

        if errors:
            return {
                "status": "failed",
                "selected_profiles": list(selection.profiles),
                "selected_attacks": list(selection.attacks),
                "selected_partitions": list(selection.partitions),
                "generated_samples": generated,
                "reused_samples": reused,
                "results": results,
                "errors": errors,
            }
        try:
            results["detector_training"] = active_operations.train_universal(
                selected_configs[selection.profiles[0]],
                project_root=root,
                profile_ids=selection.profiles,
            )
            candidate = results["detector_training"].get("model_path")
            if candidate:
                for selected_config in selected_configs.values():
                    selected_config["detector"]["model_path"] = candidate
        except Exception as exc:
            _error(errors, "universal", "detector_training", exc)
            return {
                "status": "failed",
                "selected_profiles": list(selection.profiles),
                "selected_attacks": list(selection.attacks),
                "selected_partitions": list(selection.partitions),
                "generated_samples": generated,
                "reused_samples": reused,
                "results": results,
                "errors": errors,
            }

    if "test" in selection.partitions:
        for profile_id in selection.profiles:
            if profile_id not in prepared_ok:
                continue
            try:
                payload = active_operations.prepare_test(
                    selected_configs[profile_id],
                    project_root=root,
                )
                results["test"][profile_id] = payload
                generated += int(payload.get("generated_test_graphs", 0))
                reused += int(payload.get("reused_test_graphs", 0))
            except Exception as exc:
                _error(errors, profile_id, "test", exc)
        if errors:
            return {
                "status": "failed",
                "selected_profiles": list(selection.profiles),
                "selected_attacks": list(selection.attacks),
                "selected_partitions": list(selection.partitions),
                "generated_samples": generated,
                "reused_samples": reused,
                "results": results,
                "errors": errors,
            }
        try:
            results["evaluation"] = active_operations.evaluate_universal(
                selected_configs[selection.profiles[0]],
                project_root=root,
                profile_ids=selection.profiles,
            )
        except Exception as exc:
            _error(errors, "universal", "evaluation", exc)

    return {
        "status": "failed" if errors else "ok",
        "selected_profiles": list(selection.profiles),
        "selected_attacks": list(selection.attacks),
        "selected_partitions": list(selection.partitions),
        "generated_samples": generated,
        "reused_samples": reused,
        "results": results,
        "errors": errors,
    }
