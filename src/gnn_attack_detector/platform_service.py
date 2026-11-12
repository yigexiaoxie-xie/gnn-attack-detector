"""Service functions shared by the command line and Streamlit platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import torch

from gnn_attack_detector.config import load_config, pipeline_config
from gnn_attack_detector.corpus_policy import expected_corpus_count
from gnn_attack_detector.defenses.registry import defense_spec
from gnn_attack_detector.defense_service import defense_profile_status
from gnn_attack_detector.detector_metrics import binary_metrics, profile_evaluation_summaries
from gnn_attack_detector.integrity import hash_file
from gnn_attack_detector.platform_generation import generate_attack_batch
from gnn_attack_detector.victim import load_victim_checkpoint, validate_victim_graph_contract
from gnn_attack_detector.profiles import load_profiles


_REPORT_SECTIONS = (
    "attacked_complete_graph_recall",
    "clean_complete_graph_check",
    "official_complete_graph_metrics",
)


def _project_path(root: Path, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"platform artifact path must be relative: {relative_path}")
    resolved = (root / path).resolve()
    resolved.relative_to(root)
    return resolved


def collect_platform_status(project_root: str | Path) -> dict[str, Any]:
    """Return profile, detector, protocol, and defense readiness."""

    root = Path(project_root).resolve()
    config = load_config(root / "configs" / "base.yaml")
    runtime = pipeline_config(config)
    profiles = load_profiles(config["profiles"])
    profile_rows = []
    for profile in profiles.values():
        clean = _project_path(root, profile.clean_reference)
        probe_dir = _project_path(root, profile.probe_dir)
        probes = {
            name: (probe_dir / f"{name}.pt").is_file()
            for name in config["probes"]["online"]
        }
        defense_states = defense_profile_status(
            root, profile, clean_ready=clean.is_file()
        )
        profile_rows.append(
            {
                "id": profile.identifier,
                "name": profile.name,
                "expected_num_nodes": profile.expected_num_nodes,
                "clean_ready": clean.is_file(),
                "probes": probes,
                "defenses": defense_states,
                "ready": clean.is_file() and probes.get("gcn", False),
                "defense_ready": clean.is_file(),
            }
        )
    detector_path = _project_path(root, runtime["model_path"])
    detector_ready = False
    if detector_path.is_file():
        try:
            from gnn_attack_detector.detector_model import load_graph_detector_bundle
            load_graph_detector_bundle(detector_path)
            detector_ready = True
        except (RuntimeError, ValueError, KeyError, OSError):
            pass
    report_path = _project_path(root, runtime["report_path"])
    attack_root = _project_path(root, runtime["output_dir"])
    graph_count = len(list(attack_root.rglob("*.pt"))) if attack_root.is_dir() else 0
    defenses = [
        {
            "id": name,
            "display_name": defense_spec(name).display_name,
            "backend": defense_spec(name).backend,
            "role": defense_spec(name).role,
            "execution_mode": defense_spec(name).execution_mode,
            "requires_preparation": False,
        }
        for name in config["defenses"]["models"]
    ]
    active = str(config["active_profile"])
    return {
        "active_profile": active,
        "profiles": profile_rows,
        "ready_for_detection": detector_ready,
        "detector_model": str(detector_path.relative_to(root)),
        "evaluation_report": str(report_path.relative_to(root)),
        "prediction_report": str(runtime["prediction_path"]),
        "evaluation_ready": report_path.is_file(),
        "attack_methods": list(config["attacks"]["protocols"]),
        "supported_attack_count": len(config["attacks"]["protocols"]),
        "required_attack_count": int(config["evaluation"]["minimum_attacks"]),
        "defenses": defenses,
        "supported_defense_count": len(defenses),
        "attack_graph_count": graph_count,
        "expected_attack_graph_count": expected_corpus_count(runtime),
    }


def load_platform_evaluation(
    report_path: str | Path,
    *,
    prediction_path: str | Path | None = None,
    model_path: str | Path | None = None,
    profile_id: str | None = None,
) -> dict[str, Any]:
    path = Path(report_path)
    if profile_id is not None:
        if Path(profile_id).name != profile_id or "/" in profile_id or "\\" in profile_id:
            raise ValueError("invalid profile identifier")
        profile_path = path.parent / "shared-gcn-profiles" / f"{profile_id}.json"
        if profile_path.is_file():
            path = profile_path
    if not path.is_file():
        return {"available": False, "report": None, "reason": "尚无测试结果，请先完成测试评测。"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"evaluation report is not valid JSON: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError("evaluation report must contain a JSON object")
    if report.get("format_version") not in {2, 3}:
        return {
            "available": False,
            "report": None,
            "reason": "评测报告版本较旧，请重新运行 evaluate-detector 更新结果。",
        }
    for section in _REPORT_SECTIONS:
        if section not in report:
            raise ValueError(f"evaluation report missing section: {section}")
    # Older reports aggregate attack counts across datasets. Rebuild the view
    # from the matching prediction file, without retraining or modifying it.
    if not report.get("profile_evaluations") and prediction_path is not None:
        predictions = Path(prediction_path)
        if predictions.is_file():
            try:
                rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
                rows = [item for item in rows if item.get("partition", "test") == "test"]
                measured = binary_metrics([item["label"] for item in rows], [item["prediction"] for item in rows])
                summaries = profile_evaluation_summaries(rows)
                matching = measured == report.get("test_metrics") and all(
                    summary["test_metrics"] == report.get("metrics_by_profile", {}).get(profile_id)
                    for profile_id, summary in summaries.items()
                )
                if matching:
                    report["profile_evaluations"] = summaries
            except (OSError, ValueError, KeyError, TypeError):
                pass
    historical = False
    if model_path is not None:
        current_model = Path(model_path)
        recorded_hash = report.get("detector_model_hash")
        historical = bool(
            not current_model.is_file()
            or not isinstance(recorded_hash, str)
            or hash_file(current_model) != recorded_hash
        )
    return {
        "available": True,
        "report": report,
        "reason": None,
        "historical": historical,
    }


def load_platform_accuracy_summary(
    report_path: str | Path,
    *,
    profile_ids: list[str] | tuple[str, ...],
    prediction_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate the saved accuracy reports without treating missing data as zero."""

    measurements: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        loaded = load_platform_evaluation(
            report_path,
            prediction_path=prediction_path,
            model_path=model_path,
            profile_id=profile_id,
        )
        if not loaded["available"]:
            continue
        report = loaded["report"] or {}
        detail = report.get("profile_evaluations", {}).get(profile_id, {})
        metrics = detail.get("test_metrics") or report.get("metrics_by_profile", {}).get(profile_id)
        if not isinstance(metrics, dict) or metrics.get("count", 0) <= 0:
            continue
        matrix = metrics.get("confusion_matrix", {})
        correct = int(matrix.get("tp", 0)) + int(matrix.get("tn", 0))
        count = int(metrics["count"])
        measurements.append({
            "profile_id": profile_id,
            "accuracy": float(metrics["accuracy"]),
            "count": count,
            "correct": correct,
            "historical": bool(loaded.get("historical")),
        })
    if not measurements:
        return {
            "available": False,
            "evaluated_profiles": [],
            "missing_profiles": list(profile_ids),
            "macro_accuracy": None,
            "weighted_accuracy": None,
        }
    total_count = sum(item["count"] for item in measurements)
    return {
        "available": True,
        "evaluated_profiles": [item["profile_id"] for item in measurements],
        "missing_profiles": [item for item in profile_ids if item not in {m["profile_id"] for m in measurements}],
        "macro_accuracy": sum(item["accuracy"] for item in measurements) / len(measurements),
        "weighted_accuracy": sum(item["correct"] for item in measurements) / total_count,
        "total_count": total_count,
        "historical": any(item["historical"] for item in measurements),
    }


def save_uploaded_graph(
    project_root: str | Path,
    filename: str,
    content: bytes,
    *,
    max_bytes: int = 100 * 1024 * 1024,
) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pt", ".npz"}:
        raise ValueError("Only PT or NPZ graph files are accepted")
    if not content or len(content) > int(max_bytes):
        raise ValueError("The graph file must be non-empty and no larger than 100 MB")
    output = Path(project_root).resolve() / "artifacts" / "uploads" / f"{uuid4().hex}{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return output


def detect_platform_graph(
    config_path: str | Path,
    graph_path: str | Path,
    *,
    project_root: str | Path,
    profile_id: str | None = None,
    detector: Callable[..., dict[str, Any]] | None = None,
    attribute_mode: str = "auto",
) -> dict[str, Any]:
    if detector is None:
        from gnn_attack_detector.pipeline import detect_graph

        detector = detect_graph
    options = {"attribute_mode": attribute_mode} if attribute_mode != "auto" else {}
    result = dict(
        detector(
            Path(config_path),
            Path(graph_path),
            project_root=Path(project_root).resolve(),
            profile_id=profile_id,
            **options,
        )
    )
    if result.get("prediction") not in {"clean", "attacked"}:
        raise ValueError("detector returned an invalid prediction")
    result["prediction_label"] = "检测到攻击" if result["prediction"] == "attacked" else "未检测到攻击"
    return result


def detect_platform_graphs(
    config_path: str | Path,
    graph_paths: list[str | Path],
    *,
    project_root: str | Path,
    profile_id: str | None = None,
    detector: Callable[..., dict[str, Any]] | None = None,
    attribute_mode: str = "auto",
) -> list[dict[str, Any]]:
    if not graph_paths:
        raise ValueError("at least one graph input is required")
    rows = []
    for graph_path in graph_paths:
        path = Path(graph_path)
        try:
            row = detect_platform_graph(
                config_path,
                path,
                project_root=project_root,
                profile_id=profile_id,
                detector=detector,
                attribute_mode=attribute_mode,
            )
            row["source_name"] = path.name
        except Exception as exc:
            row = {
                "status": "error",
                "profile_id": profile_id,
                "source_name": path.name,
                "prediction": "error",
                "prediction_label": "检测失败",
                "graph_attack_probability": None,
                "threshold": None,
                "scanned_nodes": 0,
                "scanned_undirected_edges": 0,
                "error": str(exc),
            }
        rows.append(row)
    return rows


def load_platform_generation_context(
    config_path: str | Path,
    project_root: str | Path,
    *,
    profile_id: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(config_path)
    selected = str(profile_id or config["active_profile"])
    profile = load_profiles(config["profiles"])[selected]
    clean_path = _project_path(root, profile.clean_reference)
    if not clean_path.is_file():
        raise FileNotFoundError(f"clean reference not found: {clean_path}")
    clean_graph = torch.load(clean_path, map_location="cpu", weights_only=False).cpu()
    surrogate, metadata = load_victim_checkpoint(_project_path(root, profile.probe_dir) / "gcn.pt")
    validate_victim_graph_contract(metadata, clean_graph, profile.identifier)
    return {"config": config, "profile": profile, "clean_graph": clean_graph, "surrogate": surrogate}


def generate_and_detect_platform_graphs(
    config_path: str | Path,
    *,
    dataset_name: str,
    attack_name: str,
    strength: float | int,
    count: int,
    project_root: str | Path,
    context_loader: Callable[..., dict[str, Any]] = load_platform_generation_context,
    generator: Callable[..., list[dict[str, Any]]] = generate_attack_batch,
    detector: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    if float(strength) == 0:
        if int(count) != 1:
            raise ValueError("0 扰动直接检测一张干净图，请将生成数量设为 1。")
        config = load_config(config_path)
        profile = load_profiles(config["profiles"])[dataset_name]
        clean_path = _project_path(root, profile.clean_reference)
        rows = detect_platform_graphs(
            config_path, [clean_path], project_root=root,
            profile_id=profile.identifier, detector=detector,
        )
        for row in rows:
            row.update({
                "source_name": f"{profile.name} · 干净图",
                "dataset": profile.name,
                "attack_name": "clean", "strength": 0, "strength_unit": "none",
                "actual_modifications": 0, "input_kind": "clean",
                "correctly_detected": row.get("prediction") == "clean",
            })
        return rows
    context = context_loader(config_path, root, profile_id=dataset_name)
    config = context["config"]
    runtime = pipeline_config(config)
    profile = context["profile"]
    # Exclude every formal partition, even when the selected dataset is not
    # the active training profile in base.yaml.
    manifest = _project_path(root, Path(config["corpus"]["sample_index_dir"]) / f"{profile.identifier}.jsonl")
    excluded = set()
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("graph_hash"):
                    excluded.add(str(record["graph_hash"]))
    generated = generator(
        clean_graph=context["clean_graph"],
        surrogate=context["surrogate"],
        dataset_name=profile.identifier,
        attack_name=str(attack_name).lower(),
        strength=strength,
        count=int(count),
        output_root=root / "artifacts" / "platform" / "generated",
        structural_rates=tuple(float(value) for value in runtime["structural_rates"]),
        targeted_target_counts=tuple(int(value) for value in runtime["targeted_target_counts"]),
        targeted_local_budget=int(runtime["targeted_local_budget"]),
        minimum_target_degree=int(runtime["minimum_target_degree"]),
        max_attempts=min(int(runtime["max_generation_attempts"]), 30),
        excluded_fingerprints=excluded,
    )
    rows: list[dict[str, Any]] = []
    for generated_row in generated:
        row = dict(generated_row)
        row.update({"profile_id": profile.identifier, "dataset": profile.name, "input_kind": "generated"})
        if row.get("status") == "ok":
            try:
                detected = detect_platform_graph(
                    config_path,
                    row["pt_path"],
                    project_root=root,
                    profile_id=profile.identifier,
                    detector=detector,
                )
                row.update(detected)
                row["correctly_detected"] = row.get("prediction") == "attacked"
            except Exception as exc:
                row.update({"status": "error", "prediction": "error", "prediction_label": "检测失败", "error": str(exc)})
        else:
            row.update({"prediction": "error", "prediction_label": "生成失败", "correctly_detected": False})
        rows.append(row)
    return rows
