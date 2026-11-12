"""Universal complete-graph detector training, evaluation, and inference."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import shutil
from copy import deepcopy
from uuid import uuid4
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from gnn_attack_detector.attacks.registry import SUPPORTED_ATTACKS
from gnn_attack_detector.config import load_config_source, pipeline_config
from gnn_attack_detector.corpus import (
    generate_graph_corpus,
    generate_test_corpus,
    verify_graph_corpus,
)
from gnn_attack_detector.corpus_policy import classify_attack_availability
from gnn_attack_detector.detector_dataset import (
    GraphDetectorDataset,
    dataset_from_records,
    load_graph_detector_dataset,
    merge_graph_detector_datasets,
    save_graph_detector_dataset,
    select_graph_detector_partitions,
    validate_split_integrity,
)
from gnn_attack_detector.detector_features import PREPROCESSING_VERSION, prepare_graph_input, save_prepared_graph, resolve_attribute_mode
from gnn_attack_detector.detector_metrics import (
    attacked_graph_recall,
    binary_metrics,
    official_graph_metrics,
    profile_evaluation_summaries,
)
from gnn_attack_detector.detector_model import (
    load_graph_detector_bundle,
    predict_graph_probabilities,
    save_graph_detector_bundle,
    train_graph_detector_model,
)
from gnn_attack_detector.device import run_with_cpu_fallback
from gnn_attack_detector.graph_io import load_graph_file
from gnn_attack_detector.integrity import hash_file
from gnn_attack_detector.profiles import DatasetProfile, load_profiles


def _resolve(root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project") from exc
    return path


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"corpus manifest not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _context(
    config_path: str | Path | Mapping[str, Any],
    project_root: str | Path,
    profile_id: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], DatasetProfile]:
    root = Path(project_root).resolve()
    config = load_config_source(config_path)
    selected = str(profile_id or config["active_profile"])
    profiles = load_profiles(config["profiles"])
    if selected not in profiles:
        raise ValueError(f"unknown dataset profile: {selected}")
    if selected != config["active_profile"]:
        config = dict(config)
        config["active_profile"] = selected
    return root, config, pipeline_config(config), profiles[selected]


def _profile_feature_path(root, config, profile_id):
    return _resolve(root, Path(config["paths"]["features"]) / "shared-gcn-v4" / "profiles" / f"{profile_id}.pt", "detector inputs")


def _input_fingerprint(graph):
    digest = hashlib.sha256()
    n = int(graph.num_nodes)
    codes = (graph.edge_index[0] * n + graph.edge_index[1]).sort().values
    digest.update(str(n).encode())
    for value in (codes, getattr(graph, "x", None)):
        if value is not None:
            value = value.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _build_profile_dataset(root, config, runtime, profile, records):
    foreign = {str(r.get("dataset_id")) for r in records} - {profile.identifier}
    if foreign:
        raise RuntimeError(f"corpus manifest contains rows from another dataset profile: {sorted(foreign)}")
    clean = torch.load(_resolve(root, profile.clean_reference, "source attributes"), map_location="cpu", weights_only=False)
    cache = _profile_feature_path(root, config, profile.identifier).parent.parent / "graphs" / profile.identifier
    mode = str(config.get("detector", {}).get("attribute_modes", {}).get(profile.identifier, "auto"))
    source_signature = _input_fingerprint(clean)
    metadata, paths = [], []
    seen = {}
    def append(record, path=None, graph=None):
        source_hash = hash_file(path) if path else source_signature
        key = hashlib.sha256(f"v4-pre{PREPROCESSING_VERSION}:{source_hash}:{source_signature}:{mode}".encode()).hexdigest()
        target = cache / f"{key}.pt"
        if not target.is_file():
            if graph is None:
                graph = load_graph_file(path, clean_graph=clean)
            save_prepared_graph(target, prepare_graph_input(graph, attribute_mode=mode))
            print(f"[PREPARED-GRAPH] {record['sample_id']}", flush=True)
        metadata.append(dict(record, attribute_mode=mode))
        paths.append(target)
    for record in records:
        if int(record.get("label", 1)) != 1:
            raise ValueError("attack manifest must contain label 1")
        path = _resolve(root, record["graph_path"], "corpus graph")
        decoded = load_graph_file(path, clean_graph=clean)
        fingerprint = _input_fingerprint(decoded)
        if fingerprint in seen and seen[fingerprint] != record["partition"]:
            raise ValueError("the same attacked graph appears across partitions")
        seen[fingerprint] = record["partition"]
        append(dict(record, label=1, independent_clean=False, input_fingerprint=fingerprint), path=path, graph=decoded)
    negative_path = config.get("detector", {}).get("nonattack_manifest")
    negatives = [] if not negative_path else [r for r in _read_jsonl(_resolve(root, negative_path, "nonattack manifest"))
                                            if r.get("dataset_id") == profile.identifier]
    if negatives:
        fingerprints = set(seen)
        groups = {}
        independent_groups = set()
        for record in negatives:
            if int(record["label"]) != 0:
                raise ValueError("nonattack manifest must have label 0")
            group = record.get("source_group")
            independent = record.get("independent_clean")
            if not isinstance(group, str) or not group.strip() or group.startswith("benchmark:") or not isinstance(independent, bool):
                raise ValueError("nonattack records require source_group and explicit boolean independent_clean")
            if group in groups and groups[group] != record["partition"]:
                raise ValueError("nonattack source group appears across partitions")
            groups[group] = record["partition"]
            if independent and not record.get("independence_evidence"):
                raise ValueError("independent_clean requires independence_evidence; content differences alone do not establish independence")
            if independent and group in independent_groups:
                raise ValueError("only one independent nonattack graph may represent each source_group")
            if independent:
                independent_groups.add(group)
            path = _resolve(root, record["graph_path"], "nonattack graph")
            graph = load_graph_file(path)
            fingerprint = _input_fingerprint(graph)
            if independent and fingerprint == source_signature:
                raise ValueError("the benchmark source graph is not an independent negative")
            if fingerprint in fingerprints:
                raise ValueError("duplicate nonattack graph; independent negatives must be distinct")
            fingerprints.add(fingerprint)
            append(dict(record, attack_name="clean", input_fingerprint=fingerprint), path=path, graph=graph)
        if not {r["partition"] for r in records} <= {r["partition"] for r in negatives}:
            raise ValueError("nonattack manifest must cover every selected partition")
    else:
        for partition in ("train", "val", "test"):
            if any(r["partition"] == partition for r in records):
                append(dict(sample_id=f"{profile.identifier}-benchmark-clean-{partition}", dataset_id=profile.identifier,
                            partition=partition, label=0, attack_name="clean", independent_clean=False,
                            source_group=f"benchmark:{profile.identifier}", input_fingerprint=source_signature), graph=clean)
        print("[BENCHMARK-ONLY] Repeated clean control; NOT an independent clean test set.", flush=True)
    dataset = dataset_from_records(metadata, paths)
    validate_split_integrity(dataset)
    save_graph_detector_dataset(_profile_feature_path(root, config, profile.identifier), dataset)
    return dataset


def prepare_detector_data(
    config_path: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path = Path("."),
    attack_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Generate selected train/validation attacks and prepare graph inputs."""

    root, config, runtime, profile = _context(config_path, project_root)
    generation: dict[str, Any] = {
        "partitions": ("train", "val"),
        "project_root": root,
    }
    if attack_runner is not None:
        generation["attack_runner"] = attack_runner
    corpus = generate_graph_corpus(config_path, **generation)
    verification = verify_graph_corpus(
        config_path,
        expected_partitions=("train", "val"),
        project_root=root,
    )
    manifest = _read_jsonl(_resolve(root, runtime["sample_index_path"], "corpus manifest"))
    selected = [item for item in manifest if item["partition"] in {"train", "val"}
                and item["attack_name"] in config["attacks"]["protocols"]]
    dataset = _build_profile_dataset(root, config, runtime, profile, selected)
    feature_path = _profile_feature_path(root, config, profile.identifier)
    return {
        "status": "ok",
        "profile_id": profile.identifier,
        "generated_by_partition": corpus["generated_by_partition"],
        "reused_by_partition": corpus["reused_by_partition"],
        "verified_attack_graphs": verification["verified_samples"],
        "feature_rows": len(dataset),
        "feature_count": 12,
        "feature_path": _relative(root, feature_path),
    }


def _available_profile_datasets(
    root: Path,
    config: dict[str, Any],
    *,
    profile_ids: Sequence[str] | None = None,
) -> list[GraphDetectorDataset]:
    directory = _resolve(root, Path(config["paths"]["features"]) / "shared-gcn-v4" / "profiles", "profile features")
    configured = tuple(str(value) for value in config["profiles"])
    if profile_ids is None:
        selected = tuple(
            profile_id
            for profile_id in sorted(configured)
            if (directory / f"{profile_id}.pt").is_file()
        )
    else:
        selected = tuple(str(value) for value in profile_ids)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("profile_ids must contain unique configured profiles")
        unknown = set(selected) - set(configured)
        if unknown:
            raise ValueError(f"unknown dataset profiles: {sorted(unknown)}")
        missing = [
            profile_id
            for profile_id in selected
            if not (directory / f"{profile_id}.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"selected profile feature datasets are missing: {missing}"
            )
    paths = [directory / f"{profile_id}.pt" for profile_id in selected]
    if not paths:
        raise FileNotFoundError("no profile feature datasets exist; run prepare-detector-data")
    datasets = []
    for profile_id, path in zip(selected, paths):
        dataset = load_graph_detector_dataset(path)
        unexpected = set(dataset.dataset_ids) - {profile_id}
        if unexpected:
            raise RuntimeError(
                f"profile feature file {path.name} belongs to another dataset profile: "
                f"{sorted(unexpected)}"
            )
        datasets.append(dataset)
    return datasets


def train_detector(
    config_path: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path = Path("."),
    profile_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train selected profiles and atomically update the deployed detector."""

    root, config, runtime, _ = _context(config_path, project_root)
    profile_ids = tuple(profile_ids) if profile_ids is not None else (str(config["active_profile"]),)
    torch.set_num_threads(int(config["runtime"].get("cpu_threads", 8)))
    dataset = merge_graph_detector_datasets(
        [
            select_graph_detector_partitions(item, ("train", "val"))
            for item in _available_profile_datasets(
                root, config, profile_ids=profile_ids
            )
        ]
    )
    model_config = runtime["model"]
    result = run_with_cpu_fallback(
        lambda device: train_graph_detector_model(
            dataset,
            hidden_channels=int(model_config["hidden_channels"]),
            dropout=float(model_config["dropout"]),
            epochs=int(model_config["epochs"]),
            patience=int(model_config["patience"]),
            learning_rate=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
            seed=int(config["seed"]),
            device=device,
            target_group_recall=float(model_config["target_group_recall"]),
            minimum_clean_specificity=float(model_config["minimum_clean_specificity"]),
            target_consecutive_epochs=int(model_config["target_consecutive_epochs"]),
            minimum_validation_group_size=int(model_config["minimum_validation_group_size"]),
        ),
        requested=str(config.get("runtime", {}).get("device", "auto")),
        label="detector training",
    )
    active_path = _resolve(root, runtime["model_path"], "detector model")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    run_path = active_path.parent / "shared-gcn-runs" / run_id / "detector.pt"
    save_graph_detector_bundle(run_path, result)
    if active_path.is_file():
        previous_path = active_path.parent / "shared-gcn-runs" / f"previous-{run_id}" / "detector.pt"
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(active_path, previous_path)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_active = active_path.with_name(f".{active_path.name}.{run_id}.tmp")
    shutil.copy2(run_path, temporary_active)
    temporary_active.replace(active_path)
    return {
        "status": "ok",
        "detector_scope": "shared_gcn_selected_profiles",
        "profiles": sorted(set(dataset.dataset_ids)),
        "training_rows": len(dataset),
        "feature_count": 12,
        "threshold": result.bundle.threshold,
        "best_epoch": result.best_epoch,
        "epochs_ran": result.epochs_ran,
        "model_path": _relative(root, active_path),
        "run_model_path": _relative(root, run_path),
        "activation_required": False,
        "active_model_unchanged": False,
        "active_model_updated": True,
    }


def generate_detector_test_data(
    config_path: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path = Path("."),
    attack_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Generate held-out graphs only after the universal detector is frozen."""

    root, config, runtime, profile = _context(config_path, project_root)
    model_path = _resolve(root, runtime["model_path"], "detector model")
    if not model_path.is_file():
        raise FileNotFoundError("train the detector before generating held-out test graphs")
    generation: dict[str, Any] = {"project_root": root}
    if attack_runner is not None:
        generation["attack_runner"] = attack_runner
    corpus = generate_test_corpus(config_path, **generation)
    manifest = _read_jsonl(_resolve(root, runtime["sample_index_path"], "corpus manifest"))
    selected = [r for r in manifest if r["attack_name"] in config["attacks"]["protocols"]]
    dataset = _build_profile_dataset(root, config, runtime, profile, selected)
    return {
        "status": "ok",
        "profile_id": profile.identifier,
        "generated_test_graphs": corpus["generated_by_partition"].get("test", 0),
        "reused_test_graphs": corpus["reused_by_partition"].get("test", 0),
        "feature_rows": len(dataset),
        "feature_path": _relative(root, _profile_feature_path(root, config, profile.identifier)),
    }


def _prediction_rows(
    dataset: GraphDetectorDataset,
    probabilities: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, partition in enumerate(dataset.partitions):
        if partition != "test":
            continue
        probability = float(probabilities[index])
        rows.append(
            {
                "sample_id": dataset.graph_ids[index],
                "partition": "test",
                "profile_id": dataset.dataset_ids[index],
                "label": int(dataset.y[index]),
                "independent_clean": bool(dataset.independent_clean[index]),
                "source_group": dataset.source_groups[index],
                "attack_name": dataset.attack_names[index],
                "strength_value": dataset.strength_values[index],
                "strength_unit": dataset.strength_units[index],
                "probability": probability,
                "prediction": int(probability >= threshold),
            }
        )
    return rows


def _official_for_rows(rows, minimum):
    # One independent unit per declared source, never benchmark controls.
    clean_groups = {}
    for row in rows:
        if row["label"] == 0 and row.get("independent_clean") and row.get("source_group"):
            clean_groups.setdefault(row["source_group"], int(row["prediction"]))
    return official_graph_metrics(
        clean_predictions=list(clean_groups.values()),
        attacked_predictions=[r["prediction"] for r in rows if r["label"] == 1],
        independent_clean_graphs=len(clean_groups), minimum_clean_graphs=minimum,
    )


def _profile_report(report, profile_id):
    scoped = deepcopy(report)
    detail = scoped.get("profile_evaluations", {}).get(profile_id)
    scoped["profiles"] = [profile_id]
    scoped["report_scope"] = "single_profile"
    scoped["metrics_by_profile"] = {profile_id: scoped.get("metrics_by_profile", {}).get(profile_id)}
    scoped["profile_evaluations"] = {profile_id: detail} if detail else {}
    scoped["test_metrics"] = scoped["metrics_by_profile"][profile_id]
    scoped.pop("acceptance", None)  # Aggregate acceptance cannot describe one dataset.
    if detail:
        for section in ("attacked_complete_graph_recall", "clean_complete_graph_check"):
            scoped[section] = detail[section]
        scoped["official_complete_graph_metrics"] = detail.get("official_complete_graph_metrics", {
            "official_graph_metric_eligible": False, "official_graph_accuracy": None,
            "reason": "legacy report lacks per-profile independence evidence",
        })
    return scoped


def evaluate_detector(
    config_path: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path = Path("."),
    profile_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate held-out rows and report the actual measured values."""

    root, config, runtime, _ = _context(config_path, project_root)
    model_path = _resolve(root, runtime["model_path"], "detector model")
    bundle = load_graph_detector_bundle(model_path)
    profile_ids = tuple(profile_ids) if profile_ids is not None else bundle.trained_profiles
    dataset = merge_graph_detector_datasets(
        _available_profile_datasets(root, config, profile_ids=profile_ids)
    )
    validate_split_integrity(dataset)
    if tuple(bundle.feature_names) != tuple(dataset.feature_names):
        raise RuntimeError("detector feature schema does not match prepared profiles")
    dataset = select_graph_detector_partitions(dataset, ("test",))
    for i, mode in enumerate(dataset.attribute_modes):
        if mode != bundle.attribute_modes.get(dataset.dataset_ids[i], "auto"):
            raise ValueError("test preprocessing differs from checkpoint; rebuild inputs with the saved attribute policy")
        benchmark = dataset.source_groups[i].startswith("benchmark:") and int(dataset.y[i]) == 0 and not bool(dataset.independent_clean[i])
        if not benchmark:
            if (dataset.input_fingerprints[i] in bundle.fitted_fingerprints
                    or dataset.source_groups[i] and dataset.source_groups[i] in bundle.fitted_source_groups):
                raise ValueError("test graph or source group was used to fit the checkpoint")
    probabilities = predict_graph_probabilities(bundle, dataset, device=config["runtime"]["device"])
    rows = _prediction_rows(dataset, probabilities, bundle.threshold)
    if not rows:
        raise RuntimeError("no held-out test rows exist")
    labels = [int(item["label"]) for item in rows]
    guesses = [int(item["prediction"]) for item in rows]
    metrics = binary_metrics(labels, guesses)
    metrics_by_profile: dict[str, dict[str, Any]] = {}
    for profile_id in sorted({str(item["profile_id"]) for item in rows}):
        profile_rows = [
            item for item in rows if str(item["profile_id"]) == profile_id
        ]
        profile_labels = [int(item["label"]) for item in profile_rows]
        profile_guesses = [int(item["prediction"]) for item in profile_rows]
        metrics_by_profile[profile_id] = binary_metrics(profile_labels, profile_guesses)
    attacked_rows = [item for item in rows if item["label"] == 1]
    attack_records = [
        {
            "sample_id": item["sample_id"],
            "attack_name": item["attack_name"],
            "strength_value": item["strength_value"],
            "strength_unit": item["strength_unit"],
        }
        for item in attacked_rows
    ]
    recall = attacked_graph_recall(
        attack_records,
        {item["sample_id"]: int(item["prediction"]) for item in attacked_rows},
    )
    selected_attacks = [str(value) for value in config["attacks"]["protocols"]]
    catalog_attacks = list(SUPPORTED_ATTACKS)
    minimum = float(config["evaluation"]["minimum_accuracy"])
    minimum_samples = int(config["evaluation"]["minimum_positive_per_attack"])
    minimum_attack_count = int(config["evaluation"]["minimum_attacks"])
    counts = Counter(item["attack_name"] for item in attacked_rows)
    per_attack_pass = {
        name: bool(
            name in recall["by_attack"]
            and counts[name] >= minimum_samples
            and recall["by_attack"][name]["recall"] >= minimum
        )
        for name in catalog_attacks
    }
    eligible_attacks = [
        name for name in catalog_attacks if counts[name] >= minimum_samples
    ]
    minimum_attack_coverage_met = len(eligible_attacks) >= minimum_attack_count
    eligible_attacks_pass = bool(
        eligible_attacks
        and all(per_attack_pass[name] for name in eligible_attacks)
    )
    selected_progress = {
        name: {
            "test_graphs": counts[name],
            "eligible": counts[name] >= minimum_samples,
            "recall": (
                recall["by_attack"][name]["recall"]
                if name in recall["by_attack"]
                else None
            ),
            "passed": per_attack_pass[name],
        }
        for name in selected_attacks
    }
    minimum_clean = int(config["evaluation"]["minimum_clean_graphs"])
    official = _official_for_rows(rows, minimum_clean)
    profile_summaries = profile_evaluation_summaries(rows)
    for profile_id, summary in profile_summaries.items():
        summary["official_complete_graph_metrics"] = _official_for_rows(
            [r for r in rows if r["profile_id"] == profile_id], minimum_clean)
    official_eligible = bool(official["official_graph_metric_eligible"] and all(
        summary["official_complete_graph_metrics"]["official_graph_metric_eligible"]
        for summary in profile_summaries.values()))
    formal_metrics = official.get("metrics") or {}
    clean_specificity_pass = bool(formal_metrics.get("specificity", 0) >= minimum)
    report = {
        "format_version": 3,
        "status": "ok",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_model_hash": hash_file(model_path),
        "architecture": "shared_gcn_reference_free",
        "trained_profiles": list(bundle.trained_profiles),
        "clean_control_warning": "Repeated benchmark clean controls are not independent negatives; these metrics do not establish real-world generalization.",
        "detector_scope": "shared_gcn_selected_profiles",
        "profiles": sorted(set(item["profile_id"] for item in rows)),
        "selected_attacks": selected_attacks,
        "catalog_attacks": catalog_attacks,
        "observed_attacks": sorted(counts),
        "threshold": bundle.threshold,
        "test_metrics": metrics,
        "metrics_by_profile": metrics_by_profile,
        "profile_evaluations": profile_summaries,
        "attacked_complete_graph_recall": recall,
        "clean_complete_graph_check": {
            "total": sum(item["label"] == 0 for item in rows),
            "correct": sum(item["label"] == 0 and item["prediction"] == 0 for item in rows),
        },
        "official_complete_graph_metrics": official,
        "acceptance": {
            "minimum": minimum,
            "minimum_test_graphs_per_attack": minimum_samples,
            "minimum_attack_count": minimum_attack_count,
            "eligible_attacks": eligible_attacks,
            "eligible_attack_count": len(eligible_attacks),
            "minimum_attack_coverage_met": minimum_attack_coverage_met,
            "twenty_protocols_present": minimum_attack_coverage_met,
            "selected_progress": selected_progress,
            "per_attack_pass": per_attack_pass,
            "all_attacks_at_least_minimum": eligible_attacks_pass,
            "overall_accuracy_at_least_minimum": formal_metrics.get("accuracy", 0) >= minimum,
            "clean_specificity_at_least_minimum": clean_specificity_pass,
            "independent_clean_graphs_eligible": official_eligible,
            "passed": bool(
                minimum_attack_coverage_met
                and eligible_attacks_pass
                and formal_metrics.get("accuracy", 0) >= minimum
                and clean_specificity_pass
                and official_eligible
            ),
        },
    }
    report_path = _resolve(root, runtime["report_path"], "evaluation report")
    # Preserve pre-migration reports too, before the first new-schema run.
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_history = report_path.parent / "shared-gcn-history" / ("previous-" + uuid4().hex[:8])
        _write_json(previous_history / "evaluation.json", previous)
        previous_predictions = _resolve(root, runtime["prediction_path"], "prediction report")
        if previous_predictions.is_file():
            shutil.copy2(previous_predictions, previous_history / "predictions.jsonl")
        for profile_id in previous.get("profiles", previous.get("metrics_by_profile", {})):
            if profile_id not in config["profiles"]:
                continue
            saved = report_path.parent / "shared-gcn-profiles" / f"{profile_id}.json"
            if not saved.is_file():
                _write_json(saved, _profile_report(previous, profile_id))
    _write_jsonl(_resolve(root, runtime["prediction_path"], "prediction report"), rows)
    history = report_path.parent / "shared-gcn-history" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    _write_json(history / "evaluation.json", report)
    _write_jsonl(history / "predictions.jsonl", rows)
    for profile_id in report["profiles"]:
        _write_json(report_path.parent / "shared-gcn-profiles" / f"{profile_id}.json", _profile_report(report, profile_id))
    _write_json(report_path, report)
    report["report_path"] = _relative(root, report_path)
    return report


def _load_detection_input(root, config, path, profile_id=None):
    """Resolve legacy compression only; never infer provenance from node count."""
    from gnn_attack_detector.graph_io import load_pt_payload

    profiles = load_profiles(config["profiles"])

    def trusted_graph(identifier, label):
        profile = profiles[identifier]
        graph = torch.load(
            _resolve(root, profile.clean_reference, label),
            map_location="cpu",
            weights_only=False,
        )
        if int(graph.num_nodes) != int(profile.expected_num_nodes):
            raise RuntimeError(
                f"trusted graph has {int(graph.num_nodes)} nodes; profile expects "
                f"{int(profile.expected_num_nodes)}"
            )
        return graph

    provenance = profile_id
    clean = None
    if provenance is not None:
        if provenance not in profiles:
            raise ValueError(f"unknown dataset profile: {provenance}")
        clean = trusted_graph(provenance, "upload source")
    if path.suffix.lower() == ".pt":
        payload = load_pt_payload(path)
        if isinstance(payload, dict) and payload.get("format_version") in (2, 3):
            provenance = payload.get("record", {}).get("dataset_id") or profile_id
            if provenance not in profiles:
                raise ValueError("legacy compact graph omits attributes; export complete NPZ using its original project")
            clean = trusted_graph(provenance, "legacy source")
    graph = load_graph_file(path, clean_graph=clean)
    if clean is not None and int(graph.num_nodes) != int(clean.num_nodes):
        raise ValueError(
            f"input graph has {int(graph.num_nodes)} nodes; selected dataset has "
            f"{int(clean.num_nodes)} nodes"
        )
    return graph, provenance


def detect_graph(config_path, graph_path, *, project_root=Path("."), profile_id=None, attribute_mode="auto"):
    root, config, runtime, _ = _context(config_path, project_root, profile_id)
    bundle = load_graph_detector_bundle(_resolve(root, runtime["model_path"], "detector model"))
    input_path = Path(graph_path).resolve()
    graph, provenance = _load_detection_input(root, config, input_path, profile_id)
    # Checkpoint policy, not mutable configuration, defines the automatic route.
    # Explicit attribute semantics supplied by the caller take precedence.
    mode = bundle.attribute_modes.get(provenance, "auto") if attribute_mode == "auto" else attribute_mode
    mode = resolve_attribute_mode(graph, mode)
    prepared = prepare_graph_input(graph, attribute_mode=mode)
    probability = float(predict_graph_probabilities(bundle, [prepared], device=config["runtime"]["device"])[0])
    result = {
        "status": "ok", "profile_id": provenance, "dataset": provenance or "unknown",
        "input_graph": str(input_path),
        "prediction": "attacked" if probability >= bundle.threshold else "clean",
        "prediction_value": int(probability >= bundle.threshold),
        "graph_attack_probability": probability, "threshold": bundle.threshold,
        "scanned_nodes": int(graph.num_nodes), "scanned_undirected_edges": int(graph.edge_index.shape[1]) // 2,
        "architecture": "shared_gcn_reference_free", "trained_profiles": list(bundle.trained_profiles),
        "attribute_mode": mode,
        "attribute_policy_note": "Unknown sources use auto; supply none for synthetic attributes. Explicit overrides may be outside training coverage.",
        "coverage_note": "Provenance is metadata, not a predicted dataset. Unknown/new graphs are not guaranteed to generalize.",
        "explanation_note": "GCN pooling and attribute consistency; no clean-reference comparison. Score is not a calibrated probability guarantee.",
    }
    demo_path = _resolve(root, runtime["demo_path"], "demo report")
    _write_json(demo_path, result)
    result["result_path"] = _relative(root, demo_path)
    return result


def activate_detector(config_path, model_path, *, project_root=Path(".")):
    root, _, runtime, _ = _context(config_path, project_root)
    candidate = _resolve(root, model_path, "candidate model")
    bundle = load_graph_detector_bundle(candidate)
    active = _resolve(root, runtime["model_path"], "active model")
    active.parent.mkdir(parents=True, exist_ok=True)
    if active.is_file() and active != candidate:
        backup = active.parent / "shared-gcn-runs" / ("previous-" + uuid4().hex) / active.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(active, backup)
    if active != candidate:
        temporary = active.with_suffix(".tmp")
        shutil.copy2(candidate, temporary)
        temporary.replace(active)
    return {"status": "ok", "active_model": _relative(root, active), "trained_profiles": list(bundle.trained_profiles)}


def rebuild_detector(config_path, run_config, *, project_root=Path(".")):
    """Reuse existing attacks only; never invoke an attack algorithm."""
    from gnn_attack_detector.run_selection import load_run_selection
    from gnn_attack_detector.workflow import selected_profile_config
    root, config, _, _ = _context(config_path, project_root)
    selection = load_run_selection(run_config, known_profiles=tuple(config["profiles"]), known_attacks=SUPPORTED_ATTACKS)
    if not {"train", "val"} <= set(selection.partitions):
        raise ValueError("rebuild-detector requires train and val; use evaluate-detector for test-only runs")
    print(
        f"[DETECTOR-REBUILD] profiles={','.join(selection.profiles)} "
        f"attacks={len(selection.attacks)} partitions={','.join(selection.partitions)}",
        flush=True,
    )
    selected_config = None
    availability: dict[str, dict[str, Any]] = {}
    for profile_id in selection.profiles:
        print(f"[DETECTOR-REBUILD] preparing features for {profile_id}", flush=True)
        selected_config = selected_profile_config(config, selection, profile_id)
        _, _, runtime, profile = _context(selected_config, root)
        records = [
            record
            for record in _read_jsonl(
                _resolve(root, runtime["sample_index_path"], "corpus manifest")
            )
            if record["attack_name"] in selection.attacks
            and record["partition"] in selection.partitions
        ]
        status = classify_attack_availability(
            runtime,
            records,
            attacks=selection.attacks,
            partitions=selection.partitions,
        )
        available = set(status["available_attacks"])
        records = [record for record in records if record["attack_name"] in available]
        availability[profile_id] = {
            "available_attacks": list(status["available_attacks"]),
            "unavailable_attacks": list(status["unavailable_attacks"]),
            "unavailable_reasons": dict(status["unavailable_reasons"]),
        }
        print(
            f"[DETECTOR-REBUILD] {profile_id} available_attacks="
            f"{len(status['available_attacks'])} unavailable_attacks="
            f"{len(status['unavailable_attacks'])}",
            flush=True,
        )
        for attack_name in status["unavailable_attacks"]:
            print(
                f"[SKIPPED-ATTACK] {profile_id} {attack_name}: "
                f"{status['unavailable_reasons'][attack_name]}",
                flush=True,
            )
        _build_profile_dataset(root, selected_config, runtime, profile, records)
        print(f"[DETECTOR-REBUILD] features ready for {profile_id}", flush=True)
    print("[DETECTOR-REBUILD] starting shared GCN training", flush=True)
    training = train_detector(selected_config, project_root=root, profile_ids=selection.profiles)
    print("[DETECTOR-REBUILD] shared GCN training finished", flush=True)
    evaluation_config = deepcopy(selected_config)
    evaluation_config["detector"]["model_path"] = training["model_path"]
    evaluation = evaluate_detector(evaluation_config, project_root=root, profile_ids=selection.profiles) if "test" in selection.partitions else None
    return {"status": "ok", "training": training, "evaluation": evaluation,
            "attack_availability": availability,
            "note": "Active detector updated automatically. Existing attacks were not regenerated; per-profile reports are retained."}
