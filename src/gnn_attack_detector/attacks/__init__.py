"""Attack protocol metadata and lazy execution adapters."""

from gnn_attack_detector.attacks.registry import (
    ATTACK_SPECS,
    SUPPORTED_ATTACKS,
    AttackSpec,
    attack_spec,
)

__all__ = ["ATTACK_SPECS", "SUPPORTED_ATTACKS", "AttackSpec", "attack_spec"]
