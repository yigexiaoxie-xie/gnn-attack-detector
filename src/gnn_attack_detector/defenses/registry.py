"""Small, explicit catalog of DeepRobust defense models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefenseSpec:
    identifier: str
    display_name: str
    class_name: str
    role: str
    execution_mode: str
    backend: str = "deeprobust"

    @property
    def online(self) -> bool:
        return self.role == "online_probe"


DEFENSE_SPECS = (
    DefenseSpec("gcn", "GCN", "GCN", "online_probe", "input_fit"),
    DefenseSpec("gcn_jaccard", "GCN-Jaccard", "GCNJaccard", "online_probe", "input_fit"),
    DefenseSpec("gcn_svd", "GCN-SVD", "GCNSVD", "benchmark", "input_fit"),
    DefenseSpec("rgcn", "RGCN", "RGCN", "benchmark", "input_fit"),
    DefenseSpec("median_gcn", "MedianGCN", "MedianGCN", "benchmark", "input_fit"),
    DefenseSpec("simpgcn", "SimPGCN", "SimPGCN", "online_probe", "input_fit"),
    DefenseSpec("prognn", "ProGNN", "ProGNN", "offline_reference", "offline"),
    DefenseSpec("gnnguard", "GNNGuard", "GNNGuard", "benchmark", "input_fit"),
    DefenseSpec("airgnn", "AirGNN", "AirGNN", "benchmark", "input_fit"),
    DefenseSpec("sgc", "SGC", "SGC", "benchmark", "input_fit"),
)

_BY_ID = {item.identifier: item for item in DEFENSE_SPECS}
if len(_BY_ID) != len(DEFENSE_SPECS):
    raise RuntimeError("defense identifiers must be unique")

SUPPORTED_DEFENSES = tuple(item.identifier for item in DEFENSE_SPECS)
ONLINE_PROBE_DEFENSES = tuple(item.identifier for item in DEFENSE_SPECS if item.online)
CHECKPOINT_DEFENSES = tuple(item.identifier for item in DEFENSE_SPECS if item.execution_mode == "checkpoint")
INPUT_FIT_DEFENSES = tuple(item.identifier for item in DEFENSE_SPECS if item.execution_mode == "input_fit")
OFFLINE_REFERENCE_DEFENSES = tuple(item.identifier for item in DEFENSE_SPECS if item.role == "offline_reference")


def defense_spec(identifier: str) -> DefenseSpec:
    try:
        return _BY_ID[str(identifier).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported defense model: {identifier}") from exc
