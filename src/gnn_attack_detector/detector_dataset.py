"""Portable disk-backed graphs; provenance and labels stay outside model inputs."""
from dataclasses import dataclass, fields
from pathlib import Path
import os

import torch

from gnn_attack_detector.detector_features import NODE_FEATURE_NAMES, load_prepared_graph


@dataclass(frozen=True)
class GraphDetectorDataset:
    graph_paths: tuple[Path, ...]
    y: torch.Tensor
    partitions: tuple[str, ...]
    graph_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    attack_names: tuple[str, ...]
    strength_values: tuple[float | None, ...]
    strength_units: tuple[str, ...]
    independent_clean: torch.Tensor
    source_groups: tuple[str, ...]
    input_fingerprints: tuple[str, ...]
    attribute_modes: tuple[str, ...]

    @property
    def feature_names(self):
        return NODE_FEATURE_NAMES

    def __len__(self):
        return len(self.graph_paths)

    def graph(self, index):
        return load_prepared_graph(self.graph_paths[index])


def dataset_from_records(records, paths):
    if not records or len(records) != len(paths):
        raise ValueError("graph records and inputs must be nonempty and aligned")
    if any(int(r['label']) not in (0, 1) or r['partition'] not in ('train', 'val', 'test') for r in records):
        raise ValueError("invalid graph labels or partitions")
    return GraphDetectorDataset(
        tuple(Path(p).resolve() for p in paths), torch.tensor([r['label'] for r in records]).long(),
        tuple(r['partition'] for r in records), tuple(r['sample_id'] for r in records),
        tuple(r['dataset_id'] for r in records), tuple(r.get('attack_name', 'clean') for r in records),
        tuple(r.get('strength_value') for r in records), tuple(r.get('strength_unit', 'none') for r in records),
        torch.tensor([bool(r.get('independent_clean', False)) and r['label'] == 0 for r in records]),
        tuple(str(r.get('source_group', '')) for r in records),
        tuple(str(r.get('input_fingerprint', '')) for r in records),
        tuple(str(r.get('attribute_mode', 'auto')) for r in records),
    )


def save_graph_detector_dataset(path, dataset):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {field.name: getattr(dataset, field.name) for field in fields(dataset)}
    payload['graph_paths'] = tuple(os.path.relpath(p, destination.parent).replace('\\', '/') for p in dataset.graph_paths)
    payload['format_version'] = 4
    temporary = destination.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_graph_detector_dataset(path):
    source = Path(path).resolve()
    payload = torch.load(source, map_location='cpu', weights_only=True)
    if payload.pop('format_version', None) != 4:
        raise RuntimeError('obsolete detector dataset; run rebuild-detector (attacks remain reusable)')
    payload['graph_paths'] = tuple((source.parent / p).resolve() for p in payload['graph_paths'])
    dataset = GraphDetectorDataset(**payload)
    if not len(dataset) or any(len(getattr(dataset, f.name)) != len(dataset) for f in fields(dataset)):
        raise ValueError('inconsistent detector dataset rows')
    if any(not p.is_file() for p in dataset.graph_paths):
        raise FileNotFoundError('prepared graph cache missing; run rebuild-detector')
    return dataset


def merge_graph_detector_datasets(datasets):
    if not datasets:
        raise ValueError('at least one detector dataset is required')
    payload = {}
    for field in fields(GraphDetectorDataset):
        values = [getattr(d, field.name) for d in datasets]
        payload[field.name] = torch.cat(values) if isinstance(values[0], torch.Tensor) else sum(values, ())
    return GraphDetectorDataset(**payload)


def validate_split_integrity(dataset):
    """Reject content/source leakage, except explicitly marked benchmark controls."""
    seen, groups = {}, {}
    for i, partition in enumerate(dataset.partitions):
        group = dataset.source_groups[i]
        benchmark = group.startswith('benchmark:') and int(dataset.y[i]) == 0 and not bool(dataset.independent_clean[i])
        if benchmark:
            continue
        fingerprint = dataset.input_fingerprints[i]
        if fingerprint:
            if fingerprint in seen and seen[fingerprint] != partition:
                raise ValueError('identical detector input appears across partitions')
            seen[fingerprint] = partition
        if group:
            if group in groups and groups[group] != partition:
                raise ValueError('source group appears across partitions')
            groups[group] = partition


def select_graph_detector_partitions(dataset, partitions):
    indices = [i for i, p in enumerate(dataset.partitions) if p in partitions]
    if not indices:
        raise ValueError('no detector rows match requested partitions')
    payload = {}
    for field in fields(dataset):
        value = getattr(dataset, field.name)
        payload[field.name] = value[indices] if isinstance(value, torch.Tensor) else tuple(value[i] for i in indices)
    return GraphDetectorDataset(**payload)
