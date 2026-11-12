"""Load the small, human-edited selection for one resumable run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class RunSelection:
    profiles: tuple[str, ...]
    attacks: tuple[str, ...]
    partitions: tuple[str, ...]

    @property
    def combination_count(self) -> int:
        return len(self.profiles) * len(self.attacks)


def _selected(values, known: Iterable[str], label: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"run.{label} must contain at least one item")
    normalized = tuple(str(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"run.{label} must contain unique items")
    unknown = set(normalized) - set(str(value) for value in known)
    if unknown:
        singular = {"profiles": "profile", "attacks": "attack", "partitions": "partition"}[label]
        raise ValueError(f"unknown {singular}: {sorted(unknown)}")
    return normalized


def load_run_selection(
    path: str | Path,
    *,
    known_profiles: Iterable[str],
    known_attacks: Iterable[str],
) -> RunSelection:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"run configuration not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    run = loaded.get("run") if isinstance(loaded, dict) else None
    if not isinstance(run, dict):
        raise ValueError("run is required")
    return RunSelection(
        profiles=_selected(run.get("profiles"), known_profiles, "profiles"),
        attacks=_selected(run.get("attacks"), known_attacks, "attacks"),
        partitions=_selected(run.get("partitions"), PARTITIONS, "partitions"),
    )
