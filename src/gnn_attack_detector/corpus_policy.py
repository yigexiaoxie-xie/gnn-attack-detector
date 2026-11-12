"""Shared per-attack corpus quota calculations.

The same functions are used by generation, verification, the web status page,
and the Windows training entry point so they cannot disagree about totals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


PARTITIONS = ("train", "val", "test")


def _counts(corpus: Mapping[str, Any], attack_name: str) -> Mapping[str, Any]:
    policy = corpus.get("sample_policy")
    if isinstance(policy, Mapping):
        default = policy.get("default")
        overrides = policy.get("overrides", {})
        if not isinstance(default, Mapping) or not isinstance(overrides, Mapping):
            raise ValueError("corpus.sample_policy is invalid")
        selected = overrides.get(str(attack_name), default)
        if not isinstance(selected, Mapping):
            raise ValueError(
                f"corpus.sample_policy.overrides.{attack_name} must be a mapping"
            )
        return selected
    legacy = corpus.get("samples_per_combination")
    if isinstance(legacy, Mapping):
        return legacy
    raise ValueError("corpus.sample_policy is required")


def sample_count(corpus: Mapping[str, Any], attack_name: str, partition: str) -> int:
    """Return the active sample quota for one attack/partition/strength."""

    name = str(partition)
    if name not in PARTITIONS:
        raise ValueError(f"unsupported corpus partition: {partition}")
    value = _counts(corpus, str(attack_name)).get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"sample quota for {attack_name}/{partition} must be a non-negative integer"
        )
    return int(value)


def _strength_count(corpus: Mapping[str, Any], attack_name: str) -> int:
    from gnn_attack_detector.attacks.registry import attack_spec

    family = attack_spec(attack_name).family
    key = "structural_rates"
    if family == "targeted":
        key = "targeted_target_counts"
    values = corpus.get(key)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{key} is required")
    return len(values)


def _strengths(corpus: Mapping[str, Any], attack_name: str) -> tuple[float, ...]:
    from gnn_attack_detector.attacks.registry import attack_spec

    key = (
        "targeted_target_counts"
        if attack_spec(attack_name).family == "targeted"
        else "structural_rates"
    )
    values = corpus.get(key)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{key} is required")
    return tuple(float(value) for value in values)


def classify_attack_availability(
    corpus: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    attacks: Iterable[str],
    partitions: Iterable[str],
) -> dict[str, Any]:
    """Separate complete attacks from absent or partial attack corpora.

    Different profiles may own different attack sets. Only attacks with every
    selected partition/strength quota complete are made available to training.
    """

    selected_attacks = tuple(str(value) for value in attacks)
    selected_partitions = tuple(str(value) for value in partitions)
    if not selected_attacks or not selected_partitions:
        raise ValueError("attacks and partitions must be nonempty")
    if any(value not in PARTITIONS for value in selected_partitions):
        raise ValueError("unsupported corpus partition")
    counts = Counter(
        (
            str(record.get("attack_name")),
            str(record.get("partition")),
            float(record.get("strength_value")),
        )
        for record in records
        if str(record.get("attack_name")) in selected_attacks
        and str(record.get("partition")) in selected_partitions
    )
    available: list[str] = []
    unavailable: list[str] = []
    unavailable_reasons: dict[str, str] = {}
    for attack_name in selected_attacks:
        present = sum(
            count
            for (name, _partition, _strength), count in counts.items()
            if name == attack_name
        )
        if present == 0:
            unavailable.append(attack_name)
            unavailable_reasons[attack_name] = "no rows in selected partitions"
            continue
        expected_strengths = _strengths(corpus, attack_name)
        expected_keys = {
            (attack_name, partition, strength)
            for partition in selected_partitions
            for strength in expected_strengths
        }
        unexpected = sorted(
            key for key in counts if key[0] == attack_name and key not in expected_keys
        )
        if unexpected:
            unavailable.append(attack_name)
            unavailable_reasons[attack_name] = f"unexpected rows {unexpected}"
            continue
        incomplete_reason: str | None = None
        for partition in selected_partitions:
            for strength in expected_strengths:
                wanted = sample_count(corpus, attack_name, partition)
                found = counts[(attack_name, partition, strength)]
                if found != wanted:
                    incomplete_reason = (
                        f"{partition}/{strength:g} expected {wanted}, found {found}"
                    )
                    break
            if incomplete_reason is not None:
                break
        if incomplete_reason is not None:
            unavailable.append(attack_name)
            unavailable_reasons[attack_name] = incomplete_reason
            continue
        available.append(attack_name)
    if not available:
        raise RuntimeError("no complete selected attacks are available for this dataset")
    return {
        "available_attacks": tuple(available),
        "unavailable_attacks": tuple(unavailable),
        "unavailable_reasons": unavailable_reasons,
    }


def expected_corpus_count(
    corpus: Mapping[str, Any],
    partitions: Iterable[str] = PARTITIONS,
    *,
    attacks: Iterable[str] | None = None,
) -> int:
    """Return the maximum active attacked-graph count for the selected policy."""

    selected_partitions = tuple(str(value) for value in partitions)
    selected_attacks = tuple(
        str(value) for value in (corpus.get("attacks", ()) if attacks is None else attacks)
    )
    return sum(
        sample_count(corpus, attack_name, partition)
        * _strength_count(corpus, attack_name)
        for partition in selected_partitions
        for attack_name in selected_attacks
    )
