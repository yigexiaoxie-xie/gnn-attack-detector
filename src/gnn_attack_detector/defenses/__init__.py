"""Robust-model catalog used by detection probes and defense benchmarks."""

from gnn_attack_detector.defenses.registry import (
    DEFENSE_SPECS,
    SUPPORTED_DEFENSES,
    DefenseSpec,
    defense_spec,
)

__all__ = ["DEFENSE_SPECS", "SUPPORTED_DEFENSES", "DefenseSpec", "defense_spec"]
