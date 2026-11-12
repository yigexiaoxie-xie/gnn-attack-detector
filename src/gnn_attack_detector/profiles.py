"""Portable metadata for every known graph dataset supported by the detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Mapping


_PROFILE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{label} must be relative")
    return Path(value).as_posix()


@dataclass(frozen=True)
class DatasetProfile:
    identifier: str
    name: str
    source: str
    setting: str
    expected_num_nodes: int
    clean_reference: str
    probe_dir: str

    @classmethod
    def from_mapping(cls, identifier: str, values: Mapping[str, Any]) -> "DatasetProfile":
        profile_id = str(identifier)
        if not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError(f"invalid profile id: {identifier}")
        required_strings = {}
        for key in ("name", "source", "setting"):
            value = values.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"profiles.{profile_id}.{key} is required")
            required_strings[key] = value.strip()
        node_count = values.get("expected_num_nodes")
        if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count <= 0:
            raise ValueError(
                f"profiles.{profile_id}.expected_num_nodes must be a positive integer"
            )
        return cls(
            identifier=profile_id,
            name=required_strings["name"],
            source=required_strings["source"].lower(),
            setting=required_strings["setting"].lower(),
            expected_num_nodes=node_count,
            clean_reference=_relative_path(
                values.get("clean_reference"),
                f"profiles.{profile_id}.clean_reference",
            ),
            probe_dir=_relative_path(
                values.get("probe_dir"), f"profiles.{profile_id}.probe_dir"
            ),
        )


def load_profiles(values: Mapping[str, Any]) -> dict[str, DatasetProfile]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("profiles must be a non-empty mapping")
    profiles: dict[str, DatasetProfile] = {}
    for identifier, profile_values in values.items():
        if not isinstance(profile_values, Mapping):
            raise ValueError(f"profiles.{identifier} must be a mapping")
        profile = DatasetProfile.from_mapping(str(identifier), profile_values)
        profiles[profile.identifier] = profile
    return profiles
