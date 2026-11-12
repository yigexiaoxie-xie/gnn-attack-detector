"""Metrics for one-verdict-per-complete-graph detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    labels: Sequence[int] | torch.Tensor,
    predictions: Sequence[int] | torch.Tensor,
) -> dict[str, Any]:
    truth = [int(value) for value in torch.as_tensor(labels).view(-1).tolist()]
    guessed = [int(value) for value in torch.as_tensor(predictions).view(-1).tolist()]
    if len(truth) != len(guessed) or not truth:
        raise ValueError("binary labels and predictions must have equal nonzero length")
    if any(value not in (0, 1) for value in truth + guessed):
        raise ValueError("binary labels and predictions must contain only 0 or 1")
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(truth, guessed))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(truth, guessed))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(truth, guessed))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(truth, guessed))
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    return {
        "count": len(truth),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": _ratio(tp + tn, len(truth)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def calibrate_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Choose a deterministic threshold from validation graph rows only."""

    values = probabilities.detach().cpu().float().view(-1)
    truth = labels.detach().cpu().long().view(-1)
    if int(values.numel()) != int(truth.numel()) or int(values.numel()) == 0:
        raise ValueError("validation graph probabilities and labels must align")
    if set(int(value) for value in truth.tolist()) != {0, 1}:
        raise ValueError("validation graph rows must contain clean and attacked labels")
    if not torch.isfinite(values).all():
        raise ValueError("validation graph probabilities contain NaN or infinity")
    candidates = sorted({0.0, 1.0, *(float(value) for value in values.tolist())})
    best_rank: tuple[float, float, float] | None = None
    best: dict[str, Any] | None = None
    for threshold in candidates:
        metrics = binary_metrics(truth, (values >= threshold).long())
        rank = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            float(threshold),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = {"threshold": float(threshold), "metrics": metrics}
    if best is None:
        raise RuntimeError("graph threshold calibration produced no candidate")
    return best


def calibrate_group_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    profiles: Sequence[str],
    attacks: Sequence[str],
    *,
    minimum_group_recall: float,
    minimum_clean_specificity: float,
    minimum_group_size: int,
) -> dict[str, Any]:
    """Choose one validation threshold by clean specificity and worst-group recall."""

    values = probabilities.detach().cpu().float().view(-1)
    truth = labels.detach().cpu().long().view(-1)
    profile_values = tuple(str(value) for value in profiles)
    attack_values = tuple(str(value) for value in attacks)
    count = int(values.numel())
    if count == 0 or len(truth) != count or len(profile_values) != count or len(attack_values) != count:
        raise ValueError("validation probabilities, labels and groups must align")
    if set(int(value) for value in truth.tolist()) != {0, 1}:
        raise ValueError("validation rows must contain clean and attacked graphs")
    if not 0 <= float(minimum_group_recall) <= 1 or not 0 <= float(minimum_clean_specificity) <= 1:
        raise ValueError("validation targets must be between zero and one")
    if not isinstance(minimum_group_size, int) or minimum_group_size < 1:
        raise ValueError("minimum validation group size must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(truth.tolist()):
        if int(label) == 1:
            groups[f"{profile_values[index]}::{attack_values[index]}"].append(index)
    if not groups:
        raise ValueError("validation rows contain no attacked groups")
    undersized = {name: len(indices) for name, indices in groups.items() if len(indices) < minimum_group_size}
    if undersized:
        raise ValueError(f"attack groups require at least {minimum_group_size} validation graphs: {undersized}")
    candidates = sorted({0.0, 1.0, *(float(value) for value in values.tolist())})
    best_rank: tuple[float, ...] | None = None
    best: dict[str, Any] | None = None
    for threshold in candidates:
        predictions = (values >= threshold).long()
        metrics = binary_metrics(truth, predictions)
        recalls = {
            name: _ratio(sum(int(predictions[index]) for index in indices), len(indices))
            for name, indices in groups.items()
        }
        worst = min(recalls.values())
        macro = sum(recalls.values()) / len(recalls)
        passed = sum(value >= minimum_group_recall for value in recalls.values())
        specificity = float(metrics["specificity"])
        clean_passed = specificity >= minimum_clean_specificity
        rank = (
            float(clean_passed),
            float(worst),
            float(passed),
            float(macro),
            specificity,
            float(threshold),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = {
                "threshold": float(threshold),
                "metrics": metrics,
                "group_recalls": recalls,
                "worst_group_recall": float(worst),
                "macro_group_recall": float(macro),
                "groups_at_target": int(passed),
                "total_groups": len(recalls),
                "clean_specificity": specificity,
                "clean_target_met": bool(clean_passed),
                "target_met": bool(clean_passed and passed == len(recalls)),
            }
    if best is None:
        raise RuntimeError("group threshold calibration produced no candidate")
    return best


def _strength_label(record: dict[str, Any]) -> str:
    value = record["strength_value"]
    unit = str(record["strength_unit"])
    if unit == "edge_rate":
        return f"{float(value) * 100:g}%"
    if unit == "target_nodes":
        return f"{int(float(value))} nodes"
    return f"{value} {unit}"


def attacked_graph_recall(
    records: list[dict[str, Any]],
    graph_predictions: dict[str, int],
) -> dict[str, Any]:
    """Report attacked graph recall by method/strength, method, and overall."""

    group_totals: dict[tuple[str, str], int] = defaultdict(int)
    group_detected: dict[tuple[str, str], int] = defaultdict(int)
    attack_totals: dict[str, int] = defaultdict(int)
    attack_detected: dict[str, int] = defaultdict(int)
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in graph_predictions:
            raise ValueError(f"missing graph prediction for {sample_id}")
        prediction = int(graph_predictions[sample_id])
        if prediction not in (0, 1):
            raise ValueError(f"invalid graph prediction for {sample_id}: {prediction}")
        attack = str(record["attack_name"])
        strength = _strength_label(record)
        group_totals[(attack, strength)] += 1
        group_detected[(attack, strength)] += prediction
        attack_totals[attack] += 1
        attack_detected[attack] += prediction

    by_attack_strength: dict[str, dict[str, dict[str, Any]]] = {}
    for attack in sorted(attack_totals):
        by_attack_strength[attack] = {}
        for group_attack, strength in sorted(group_totals):
            if group_attack != attack:
                continue
            total = group_totals[(attack, strength)]
            detected = group_detected[(attack, strength)]
            by_attack_strength[attack][strength] = {
                "total": total,
                "detected": detected,
                "recall": _ratio(detected, total),
            }
    by_attack = {
        attack: {
            "total": attack_totals[attack],
            "detected": attack_detected[attack],
            "recall": _ratio(attack_detected[attack], attack_totals[attack]),
        }
        for attack in sorted(attack_totals)
    }
    total = sum(attack_totals.values())
    detected = sum(attack_detected.values())
    return {
        "by_attack_strength": by_attack_strength,
        "by_attack": by_attack,
        "overall": {
            "total": total,
            "detected": detected,
            "recall": _ratio(detected, total),
        },
    }


def official_graph_metrics(
    *,
    clean_predictions: Sequence[int],
    attacked_predictions: Sequence[int],
    independent_clean_graphs: int,
    minimum_clean_graphs: int,
) -> dict[str, Any]:
    """Compute accuracy only when enough genuinely independent clean graphs exist."""

    clean = [int(value) for value in clean_predictions]
    attacked = [int(value) for value in attacked_predictions]
    independent = int(independent_clean_graphs)
    minimum = int(minimum_clean_graphs)
    payload: dict[str, Any] = {
        "independent_clean_graphs": independent,
        "clean_checks": len(clean),
        "attacked_graphs": len(attacked),
        "minimum_clean_graphs": minimum,
    }
    if independent < minimum:
        payload.update(
            {
                "official_graph_metric_eligible": False,
                "official_graph_accuracy": None,
                "metrics": None,
                "reason": (
                    "insufficient independent clean complete graphs: "
                    f"found {independent}, require {minimum}"
                ),
            }
        )
        return payload
    if not clean or not attacked:
        raise ValueError("official graph accuracy requires clean and attacked graphs")
    metrics = binary_metrics(
        [0] * len(clean) + [1] * len(attacked),
        clean + attacked,
    )
    payload.update(
        {
            "official_graph_metric_eligible": True,
            "official_graph_accuracy": metrics["accuracy"],
            "metrics": metrics,
            "reason": None,
        }
    )
    return payload


def profile_evaluation_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize saved test predictions separately for each dataset."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("partition", "test") != "test":
            continue
        grouped[str(row["profile_id"])].append(row)
    summaries = {}
    for profile_id, samples in sorted(grouped.items()):
        metrics = binary_metrics(
            [item["label"] for item in samples],
            [item["prediction"] for item in samples],
        )
        attacked = [item for item in samples if int(item["label"]) == 1]
        # Scope keys locally; legacy files may reuse sample IDs across datasets.
        records = [dict(item, sample_id=str(index)) for index, item in enumerate(attacked)]
        recall = attacked_graph_recall(
            records, {str(index): int(item["prediction"]) for index, item in enumerate(attacked)}
        )
        matrix = metrics["confusion_matrix"]
        summaries[profile_id] = {
            "test_metrics": metrics,
            "attacked_complete_graph_recall": recall,
            "clean_complete_graph_check": {
                "total": matrix["tn"] + matrix["fp"], "correct": matrix["tn"],
            },
        }
    return summaries
