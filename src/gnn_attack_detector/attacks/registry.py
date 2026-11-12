"""Canonical DeepRobust attack protocols exposed by the platform."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackSpec:
    identifier: str
    display_name: str
    algorithm: str
    operation: str
    family: str
    strength_unit: str
    backend: str = "deeprobust"
    generation_mode: str = "live"
    parallel_safe: bool = True
    cuda_capable: bool = False

    @property
    def live_generation(self) -> bool:
        return self.generation_mode in {"live", "precomputed_or_live"}


def _global(identifier: str, display_name: str, algorithm: str, operation: str, **kwargs) -> AttackSpec:
    return AttackSpec(identifier, display_name, algorithm, operation, "global", "edge_rate", **kwargs)


def _targeted(identifier: str, display_name: str, algorithm: str, operation: str = "structure", **kwargs) -> AttackSpec:
    return AttackSpec(identifier, display_name, algorithm, operation, "targeted", "target_nodes", **kwargs)


ATTACK_SPECS = (
    _global("random_add", "Random Add", "random", "add"),
    _global("random_delete", "Random Delete", "random", "remove"),
    _global("random_flip", "Random Flip", "random", "flip"),
    _global("dice_add", "DICE Add", "dice", "add"),
    _global("dice_delete", "DICE Delete", "dice", "remove"),
    _global("dice_mix", "DICE Mixed", "dice", "mix"),
    _global("degree_add", "Degree Add", "degree", "add"),
    _global("degree_delete", "Degree Delete", "degree", "remove"),
    _global("prbcd", "PRBCD", "prbcd", "flip", parallel_safe=False, cuda_capable=True),
    _global(
        "structack",
        "Structack",
        "structack",
        "add",
        backend="structack",
    ),
    _targeted(
        "gfattack",
        "GFAttack",
        "gfattack",
        backend="greatx",
        parallel_safe=False,
    ),
    _global(
        "fgattack_untargeted",
        "FGAttack Untargeted",
        "fgattack",
        "flip",
        backend="greatx",
        parallel_safe=False,
        cuda_capable=True,
    ),
    _targeted("rnd", "RND", "rnd"),
    _targeted("nettack_structure", "Nettack Structure", "nettack", "structure"),
    _targeted("nettack_feature", "Nettack Feature", "nettack", "feature"),
    _targeted("nettack_hybrid", "Nettack Hybrid", "nettack", "hybrid"),
    _targeted("fga", "FGA", "fga", cuda_capable=True),
    _targeted("sgattack", "SGAttack", "sgattack", cuda_capable=True),
    _global("pgdattack", "PGDAttack", "pgdattack", "flip", parallel_safe=False, cuda_capable=True),
    _global(
        "metattack",
        "Metattack",
        "metattack",
        "flip",
        generation_mode="precomputed_or_live",
        parallel_safe=False,
        cuda_capable=True,
    ),
    _targeted("igattack", "IGAttack", "igattack", cuda_capable=True),
    _global(
        "bridge_add",
        "Bridge Add",
        "bridge_add",
        "add",
        backend="fast_structural",
    ),
    _global(
        "hub_rewire",
        "Hub Rewire",
        "hub_rewire",
        "flip",
        backend="fast_structural",
    ),
)

_BY_ID = {item.identifier: item for item in ATTACK_SPECS}
if len(_BY_ID) != len(ATTACK_SPECS):
    raise RuntimeError("attack protocol identifiers must be unique")
if len({item.display_name for item in ATTACK_SPECS}) != len(ATTACK_SPECS):
    raise RuntimeError("attack protocol display names must be unique")

SUPPORTED_ATTACKS = tuple(item.identifier for item in ATTACK_SPECS)
GLOBAL_ATTACKS = tuple(item.identifier for item in ATTACK_SPECS if item.family == "global")
TARGETED_ATTACKS = tuple(item.identifier for item in ATTACK_SPECS if item.family == "targeted")
LIVE_GENERATION_ATTACKS = tuple(item.identifier for item in ATTACK_SPECS if item.live_generation)


def attack_spec(identifier: str) -> AttackSpec:
    try:
        return _BY_ID[str(identifier).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported attack protocol: {identifier}") from exc
