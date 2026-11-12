"""Portable configuration validation for the open-source toolkit."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import yaml

from gnn_attack_detector.attacks.registry import SUPPORTED_ATTACKS
from gnn_attack_detector.defenses.registry import (
    OFFLINE_REFERENCE_DEFENSES,
    ONLINE_PROBE_DEFENSES,
    SUPPORTED_DEFENSES,
)
from gnn_attack_detector.profiles import load_profiles


ALLOWED_DEVICES = {"auto", "cpu", "cuda"}


def _is_absolute(value: Any) -> bool:
    return Path(str(value)).is_absolute() or PureWindowsPath(str(value)).is_absolute()


def _mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} is required")
    return value


def _relative(mapping: Mapping[str, Any], keys: tuple[str, ...], prefix: str) -> None:
    for key in keys:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix}.{key} is required")
        if _is_absolute(value):
            raise ValueError(f"{prefix}.{key} must be relative")


def _positive_list(values: Any, label: str, *, fractions: bool = False) -> None:
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must contain unique values")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} contains a non-number")
        if fractions and not 0.0 < float(value) < 1.0:
            raise ValueError(f"{label} must contain values between 0 and 1")
        if not fractions and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{label} must contain positive integers")


def _validate_sample_policy(corpus: Mapping[str, Any]) -> None:
    policy = corpus.get("sample_policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("default"), dict):
        raise ValueError("corpus.sample_policy.default is required")
    overrides = policy.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("corpus.sample_policy.overrides must be a mapping")
    groups = [("default", policy["default"], False)] + [
        (str(name), values, True) for name, values in overrides.items()
    ]
    for name, counts, allows_zero in groups:
        if not isinstance(counts, dict):
            raise ValueError(f"corpus sample policy for {name} must be a mapping")
        for partition in ("train", "val", "test"):
            value = counts.get(partition)
            minimum = 0 if allows_zero else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"invalid corpus quota for {name}/{partition}")


def _validated_config(loaded: Any) -> dict[str, Any]:
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    loaded = deepcopy(loaded)

    runtime = _mapping(loaded, "runtime")
    if runtime.get("device", "auto") not in ALLOWED_DEVICES:
        raise ValueError("runtime.device must be auto, cpu, or cuda")
    paths = _mapping(loaded, "paths")
    for key, value in paths.items():
        if not isinstance(value, str) or _is_absolute(value):
            raise ValueError(f"paths.{key} must be relative")

    profiles = load_profiles(_mapping(loaded, "profiles"))
    active_profile = loaded.get("active_profile")
    if active_profile not in profiles:
        raise ValueError("active_profile must name a configured profile")

    attacks = _mapping(loaded, "attacks")
    protocols = attacks.get("protocols")
    if not isinstance(protocols, list) or len(protocols) != len(set(protocols)):
        raise ValueError("attacks.protocols must contain unique values")
    if any(str(value) not in SUPPORTED_ATTACKS for value in protocols):
        raise ValueError("attacks.protocols contains an unsupported protocol")
    if attacks.get("require_all_protocols") and tuple(protocols) != SUPPORTED_ATTACKS:
        raise ValueError(
            f"attacks.protocols must match the fixed {len(SUPPORTED_ATTACKS)}-protocol registry"
        )
    _positive_list(attacks.get("structural_rates"), "attacks.structural_rates", fractions=True)
    _positive_list(attacks.get("targeted_target_counts"), "attacks.targeted_target_counts")
    if not isinstance(attacks.get("targeted_local_budget"), int) or attacks["targeted_local_budget"] <= 0:
        raise ValueError("attacks.targeted_local_budget must be positive")
    _relative(attacks, ("precomputed_metattack_dir",), "attacks")

    corpus = _mapping(loaded, "corpus")
    _validate_sample_policy(corpus)
    _relative(corpus, ("output_dir", "sample_index_dir"), "corpus")

    detector = _mapping(loaded, "detector")
    detector_model = _mapping(detector, "model")
    if detector_model.get("type") != "shared_gcn_reference_free":
        raise ValueError("detector.model.type must be shared_gcn_reference_free; migrate configs/base.yaml")
    for key in (
        "hidden_channels",
        "epochs",
        "patience",
        "target_consecutive_epochs",
        "minimum_validation_group_size",
    ):
        value = detector_model.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"detector.model.{key} must be a positive integer")
    for key in ("target_group_recall", "minimum_clean_specificity"):
        value = detector_model.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 1
        ):
            raise ValueError(f"detector.model.{key} must be between zero and one")
    modes = detector.get("attribute_modes", {})
    if not isinstance(modes, dict) or any(p not in profiles or mode not in {"auto", "none", "content"} for p, mode in modes.items()):
        raise ValueError("detector.attribute_modes must map configured profiles to auto/none/content")
    if detector.get("nonattack_manifest") is not None:
        _relative(detector, ("nonattack_manifest",), "detector")
    _relative(detector, ("model_path", "report_path", "prediction_path", "demo_path"), "detector")

    defenses = _mapping(loaded, "defenses").get("models")
    if tuple(defenses or ()) != SUPPORTED_DEFENSES:
        raise ValueError("defenses.models must match the fixed defense catalog")
    probes = _mapping(loaded, "probes")
    if tuple(probes.get("online", ())) != ONLINE_PROBE_DEFENSES:
        raise ValueError("probes.online must match the online probe catalog")
    if tuple(probes.get("offline_reference", ())) != OFFLINE_REFERENCE_DEFENSES:
        raise ValueError("probes.offline_reference must match the offline catalog")
    training = _mapping(probes, "training")
    for key in ("hidden_channels", "epochs", "patience"):
        value = training.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"probes.training.{key} must be a positive integer")
    for key in ("dropout", "learning_rate", "weight_decay", "jaccard_threshold"):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            raise ValueError(f"probes.training.{key} must be a non-negative number")
    return loaded


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return _validated_config(loaded)


def load_config_source(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a file-backed or already-loaded configuration uniformly."""

    if isinstance(source, Mapping):
        return _validated_config(dict(source))
    return load_config(source)


def pipeline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return one flat runtime view for the existing corpus/pipeline functions."""

    attacks = dict(_mapping(config, "attacks"))
    corpus = dict(_mapping(config, "corpus"))
    detector = dict(_mapping(config, "detector"))
    profile_id = str(config["active_profile"])
    sample_index_path = (
        Path(str(corpus["sample_index_dir"])) / f"{profile_id}.jsonl"
    ).as_posix()
    return {
        **attacks,
        **corpus,
        "sample_index_path": sample_index_path,
        "attacks": list(attacks["protocols"]),
        "model": dict(detector["model"]),
        "model_path": detector["model_path"],
        "report_path": detector["report_path"],
        "prediction_path": detector["prediction_path"],
        "demo_path": detector["demo_path"],
        "evaluation": dict(_mapping(config, "evaluation")),
        "clean_reference_path": _mapping(config, "profiles")[
            str(config["active_profile"])
        ]["clean_reference"],
    }
