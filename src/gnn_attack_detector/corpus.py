"""Resumable complete-graph corpus generation for selected attack protocols."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import torch

from gnn_attack_detector.device import resolve_device, run_with_cpu_fallback

from gnn_attack_detector.attacks.registry import (
    GLOBAL_ATTACKS,
    SUPPORTED_ATTACKS,
    TARGETED_ATTACKS,
    attack_spec,
)
from gnn_attack_detector.config import load_config_source, pipeline_config
from gnn_attack_detector.corpus_policy import PARTITIONS, expected_corpus_count, sample_count
from gnn_attack_detector.data import load_graph_for_config
from gnn_attack_detector.graph_attacks import (
    AttackCandidateRejected,
    CompleteGraphAttackOutcome,
    edge_difference,
    eligible_attack_targets,
    feature_difference,
    run_complete_graph_attack,
    structural_budget,
)
from gnn_attack_detector.integrity import graph_fingerprint
from gnn_attack_detector.profiles import load_profiles
from gnn_attack_detector.victim import load_victim_checkpoint, validate_victim_graph_contract


def _resolve(root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the project") from exc
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid corpus manifest line {line_number}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"corpus manifest line {line_number} must be an object")
        records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_compact_graph(
    path: str | Path,
    attacked_graph: Any,
    clean_fingerprint: dict[str, str],
    record: dict[str, Any],
) -> None:
    """Store topology and store features only when an attack changed them."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean_x_hash = str(clean_fingerprint["x"])
    changed_features = graph_fingerprint(attacked_graph)["x"] != clean_x_hash
    payload: dict[str, Any] = {
        "format_version": 3 if changed_features else 2,
        "edge_index": attacked_graph.edge_index.detach().cpu().long(),
        "clean_fingerprint": dict(clean_fingerprint),
        "record": dict(record),
    }
    if changed_features:
        payload["x"] = attacked_graph.x.detach().cpu().float()
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_compact_graph(path: str | Path, clean_graph: Any) -> Any:
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") not in {2, 3}:
        raise RuntimeError(f"unsupported graph artifact format: {source}")
    if payload.get("clean_fingerprint") != graph_fingerprint(clean_graph):
        raise RuntimeError("clean reference fingerprint does not match graph artifact")
    edge_index = payload.get("edge_index")
    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RuntimeError("graph artifact edge_index must have shape [2, num_edges]")
    graph = clean_graph.clone().cpu()
    graph.edge_index = edge_index.detach().cpu().long().contiguous()
    if int(payload["format_version"]) == 3:
        features = payload.get("x")
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != tuple(graph.x.shape):
            raise RuntimeError("feature attack artifact has invalid x")
        graph.x = features.detach().cpu().float().contiguous()
    if int(graph.edge_index.numel()) and (
        int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= int(graph.num_nodes)
    ):
        raise RuntimeError("graph artifact edge index is outside the profile")
    return graph


def _strengths(runtime: dict[str, Any], attack_name: str) -> list[float | int]:
    return list(
        runtime["structural_rates"]
        if attack_name in GLOBAL_ATTACKS
        else runtime["targeted_target_counts"]
    )


def _sample_seed(
    runtime: dict[str, Any],
    partition: str,
    attack_index: int,
    strength_index: int,
    sample_index: int,
    attempt: int,
) -> int:
    base = int(runtime["seed_namespaces"][partition])
    return base + attack_index * 1_000_000 + strength_index * 10_000 + sample_index * 100 + attempt


def _context(
    config_path: str | Path | Mapping[str, Any],
    project_root: str | Path,
    graph_loader: Callable[..., Any] | None,
    checkpoint_loader: Callable[..., Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], Any, Any]:
    root = Path(project_root).resolve()
    config = load_config_source(config_path)
    runtime = pipeline_config(config)
    graph = (
        load_graph_for_config(root, config)
        if graph_loader is None
        else graph_loader(root, config)
    ).clone().cpu()
    profiles = load_profiles(config["profiles"])
    profile = profiles[str(config["active_profile"])]
    checkpoint = _resolve(root, Path(profile.probe_dir) / "gcn.pt", "profile GCN")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"profile GCN not found: {checkpoint}")
    model, metadata = checkpoint_loader(checkpoint, device="cpu")
    validate_victim_graph_contract(metadata, graph, profile.identifier)
    model.eval()
    return root, config, runtime, graph, model.cpu()


def _precomputed_meta_candidates(
    root: Path,
    runtime: dict[str, Any],
    dataset_name: str,
) -> tuple[Path, ...]:
    directory = _resolve(
        root,
        runtime["precomputed_metattack_dir"],
        "Metattack artifact directory",
    )
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise RuntimeError(f"Metattack artifact directory is not a folder: {directory}")
    dataset = str(dataset_name).lower().replace("-", "_")
    return tuple(
        path.resolve()
        for path in sorted(directory.glob("*.npz"))
        if path.name.lower().replace("-", "_").startswith(dataset)
        and "meta" in path.name.lower()
    )


def _expected_modifications(
    clean_graph: Any,
    runtime: dict[str, Any],
    attack_name: str,
    strength: float | int,
) -> int:
    spec = attack_spec(attack_name)
    if spec.family == "global":
        return structural_budget(clean_graph, float(strength))
    return int(strength) * int(runtime["targeted_local_budget"])


def _compatible_existing_graph(
    clean_graph: Any,
    graph: Any,
    runtime: dict[str, Any],
    record: dict[str, Any],
    attack_name: str,
    strength: float | int,
) -> bool:
    if (
        int(graph.num_nodes) != int(clean_graph.num_nodes)
        or tuple(graph.x.shape) != tuple(clean_graph.x.shape)
        or not torch.equal(graph.y.detach().cpu(), clean_graph.y.detach().cpu())
    ):
        return False
    added, removed = edge_difference(clean_graph, graph)
    changed = feature_difference(clean_graph, graph)
    actual = len(added) + len(removed) + len(changed)
    if actual <= 0:
        return False
    spec = attack_spec(attack_name)
    if spec.family == "global":
        return actual == _expected_modifications(
            clean_graph, runtime, attack_name, strength
        )
    targets = [int(value) for value in record.get("target_nodes", [])]
    expected = _expected_modifications(clean_graph, runtime, attack_name, strength)
    return bool(
        len(targets) == int(strength)
        and len(set(targets)) == len(targets)
        and len(targets) <= actual <= expected
    )


def generate_graph_corpus(
    config_path: str | Path | Mapping[str, Any],
    *,
    partitions: Sequence[str],
    project_root: str | Path = Path("."),
    graph_loader: Callable[..., Any] | None = None,
    checkpoint_loader: Callable[..., Any] = load_victim_checkpoint,
    attack_runner: Callable[..., CompleteGraphAttackOutcome] = run_complete_graph_attack,
) -> dict[str, Any]:
    requested = tuple(str(value) for value in partitions)
    if not requested or len(set(requested)) != len(requested) or any(value not in PARTITIONS for value in requested):
        raise ValueError("partitions must contain unique train, val, or test values")
    root, config, runtime, clean_graph, model = _context(
        config_path, project_root, graph_loader, checkpoint_loader
    )
    output_dir = _resolve(root, runtime["output_dir"], "corpus.output_dir")
    manifest_path = _resolve(root, runtime["sample_index_path"], "corpus.sample_index_path")
    clean_path = _resolve(root, runtime["clean_reference_path"], "profile.clean_reference")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_fingerprint = graph_fingerprint(clean_graph)
    if clean_path.is_file():
        saved_clean = torch.load(clean_path, map_location="cpu", weights_only=False)
        if graph_fingerprint(saved_clean) != clean_fingerprint:
            raise RuntimeError("saved clean reference does not match active profile")
    else:
        torch.save(clean_graph, clean_path)

    existing_rows = _read_jsonl(manifest_path)
    legacy_manifest = _resolve(
        root,
        Path(config["paths"]["features"]) / "graph-samples.jsonl",
        "legacy corpus manifest",
    )
    migration_rows = list(existing_rows)
    if legacy_manifest != manifest_path:
        migration_rows.extend(_read_jsonl(legacy_manifest))
    existing = {str(item.get("sample_id")): item for item in existing_rows}
    migrated_paths: set[str] = set()
    used_reuse_paths: set[str] = set()
    used_precomputed_paths: set[str] = set()
    rejected_precomputed: set[tuple[str, float]] = set()
    generated = Counter()
    reused = Counter()
    profile_id = str(config["active_profile"])
    attacks = [str(value) for value in runtime["attacks"]]
    requested_partitions = set(requested)
    selected_attacks = set(attacks)
    precomputed_meta_candidates = (
        _precomputed_meta_candidates(
            root,
            runtime,
            str(config["profiles"][profile_id]["name"]),
        )
        if "metattack" in selected_attacks
        else ()
    )
    records = [
        item
        for item in existing.values()
        if str(item.get("partition")) not in requested_partitions
        or str(item.get("attack_name")) not in selected_attacks
    ]
    max_targets = max(int(value) for value in runtime["targeted_target_counts"])
    target_pool = (
        eligible_attack_targets(
            clean_graph,
            model,
            minimum_degree=int(runtime["minimum_target_degree"]),
            required_count=max_targets,
        )
        if any(name in TARGETED_ATTACKS for name in attacks)
        else []
    )

    for partition in requested:
        for attack_name in attacks:
            attack_index = SUPPORTED_ATTACKS.index(attack_name)
            spec = attack_spec(attack_name)
            requested_device = (
                str(config.get("runtime", {}).get("device", "auto"))
                if spec.cuda_capable
                else "cpu"
            )
            attack_device = resolve_device(requested_device)
            print(f"[DEVICE] {partition} {attack_name}: {attack_device}")
            expected_backend = spec.backend
            allow_precomputed_reuse = spec.generation_mode == "precomputed_or_live"
            ready = 0
            total = sample_count(runtime, attack_name, partition) * len(_strengths(runtime, attack_name))
            for strength_index, strength in enumerate(_strengths(runtime, attack_name)):
                for ordinal in range(sample_count(runtime, attack_name, partition)):
                    sample_id = f"{profile_id}-{partition}-{attack_name}-{strength_index:02d}-{ordinal:03d}"
                    relative = Path(runtime["output_dir"]) / profile_id / partition / attack_name / f"{sample_id}.pt"
                    graph_path = _resolve(root, relative, "corpus graph")
                    previous = existing.get(sample_id)
                    previous_relative = (
                        str(previous.get("graph_path"))
                        if previous and previous.get("graph_path")
                        else relative.as_posix()
                    )
                    previous_path = _resolve(
                        root,
                        previous_relative,
                        "existing corpus graph",
                    )
                    seen_hashes = {
                        str(item.get("graph_hash"))
                        for item in records
                        if item.get("attack_name") == attack_name
                        and item.get("strength_value") == strength
                    }
                    if (
                        previous
                        and str(previous.get("backend", "deeprobust")).lower()
                        == expected_backend
                        and previous_path.is_file()
                        and previous_relative not in used_reuse_paths
                    ):
                        try:
                            restored = load_compact_graph(previous_path, clean_graph)
                            restored_hash = graph_fingerprint(restored)["combined"]
                            if (
                                restored_hash == previous.get("graph_hash")
                                and restored_hash not in seen_hashes
                                and _compatible_existing_graph(
                                    clean_graph,
                                    restored,
                                    runtime,
                                    previous,
                                    attack_name,
                                    strength,
                                )
                            ):
                                normalized_previous = dict(previous)
                                normalized_previous["backend"] = expected_backend
                                normalized_previous.setdefault(
                                    "generation_mode", "reused"
                                )
                                records.append(normalized_previous)
                                used_reuse_paths.add(previous_relative)
                                precomputed_source = normalized_previous.get(
                                    "precomputed_source_path"
                                )
                                if precomputed_source:
                                    used_precomputed_paths.add(
                                        str(
                                            _resolve(
                                                root,
                                                str(precomputed_source),
                                                "Metattack source",
                                            )
                                        )
                                    )
                                seen_hashes.add(restored_hash)
                                _write_jsonl(manifest_path, records)
                                reused[partition] += 1
                                ready += 1
                                print(f"[REUSED] {partition} {attack_name} {ready}/{total} saved={previous_relative}")
                                continue
                        except Exception:
                            pass

                    # Import a compatible graph from a legacy manifest instead
                    # of spending CPU time generating the same protocol slot again.
                    legacy = next(
                        (
                            item
                            for item in migration_rows
                            if (
                                str(item.get("graph_path", "")) not in migrated_paths
                            )
                            and str(item.get("graph_path", "")) not in used_reuse_paths
                            and item.get("dataset_id") == profile_id
                            and str(item.get("backend", "deeprobust")).lower()
                            == expected_backend
                            and str(item.get("partition")) == partition
                            and str(item.get("attack_name")) == attack_name
                            and item.get("strength_value") is not None
                            and abs(float(item["strength_value"]) - float(strength)) < 1e-12
                        ),
                        None,
                    )
                    if legacy is not None:
                        source_relative = str(legacy["graph_path"])
                        source_path = _resolve(root, source_relative, "legacy corpus graph")
                        if source_path.is_file():
                            try:
                                try:
                                    restored = load_compact_graph(source_path, clean_graph)
                                except Exception:
                                    from gnn_attack_detector.graph_io import load_graph_file

                                    restored = load_graph_file(source_path, clean_graph=clean_graph)
                                if _compatible_existing_graph(
                                    clean_graph,
                                    restored,
                                    runtime,
                                    legacy,
                                    attack_name,
                                    strength,
                                ):
                                    restored_hash = graph_fingerprint(restored)["combined"]
                                    if restored_hash not in seen_hashes:
                                        added, removed = edge_difference(clean_graph, restored)
                                        changed = feature_difference(clean_graph, restored)
                                        actual = len(added) + len(removed) + len(changed)
                                        record = {
                                            "sample_id": sample_id,
                                            "dataset_id": profile_id,
                                            "dataset": str(config["profiles"][profile_id]["name"]),
                                            "partition": partition,
                                            "label": 1,
                                            "attack_name": attack_name,
                                            "backend": expected_backend,
                                            "generation_mode": (
                                                "reused_precomputed"
                                                if allow_precomputed_reuse
                                                else "migrated"
                                            ),
                                            "independent": True,
                                            "strength_value": strength,
                                            "strength_unit": spec.strength_unit,
                                            "seed": int(legacy.get("seed", -1)),
                                            "target_nodes": list(legacy.get("target_nodes", [])),
                                            "requested_modifications": _expected_modifications(
                                                clean_graph,
                                                runtime,
                                                attack_name,
                                                strength,
                                            ),
                                            "actual_modifications": actual,
                                            "added_edges": [list(edge) for edge in sorted(added)],
                                            "removed_edges": [list(edge) for edge in sorted(removed)],
                                            "changed_features": [list(value) for value in sorted(changed)],
                                            "graph_path": relative.as_posix(),
                                            "graph_hash": restored_hash,
                                            "migrated_from": source_relative,
                                        }
                                        save_compact_graph(graph_path, restored, clean_fingerprint, record)
                                        records.append(record)
                                        migrated_paths.add(source_relative)
                                        used_reuse_paths.add(source_relative)
                                        seen_hashes.add(restored_hash)
                                        _write_jsonl(manifest_path, records)
                                        reused[partition] += 1
                                        ready += 1
                                        print(f"[MIGRATED] {partition} {attack_name} {ready}/{total} from={source_relative}")
                                        continue
                            except Exception:
                                pass

                    outcome = None
                    used_seed = None
                    used_precomputed_path: Path | None = None
                    rejection_counts: Counter[str] = Counter()
                    last_rejection: str | None = None
                    for attempt in range(int(runtime["max_generation_attempts"])):
                        seed = _sample_seed(runtime, partition, attack_index, strength_index, ordinal, attempt)
                        targets: tuple[int, ...] = ()
                        if spec.family == "targeted":
                            targets = tuple(random.Random(seed).sample(target_pool, int(strength)))
                        if attempt == 0 or attempt % 10 == 0:
                            print(
                                f"[ATTACKING] {partition} {attack_name} "
                                f"{ready + 1}/{total} attempt={attempt + 1} "
                                f"seed={seed}",
                                flush=True,
                            )
                        try:
                            precomputed_path = None
                            if attack_name == "metattack":
                                precomputed_path = next(
                                    (
                                        path
                                        for path in precomputed_meta_candidates
                                        if str(path) not in used_precomputed_paths
                                        and (str(path), float(strength))
                                        not in rejected_precomputed
                                    ),
                                    None,
                                )
                            candidate = run_with_cpu_fallback(
                                lambda device: attack_runner(
                                    graph=clean_graph,
                                    surrogate=model,
                                    attack_name=attack_name,
                                    strength=strength,
                                    target_nodes=targets,
                                    local_budget=int(runtime["targeted_local_budget"]),
                                    seed=seed,
                                    precomputed_graph_path=precomputed_path,
                                    device=str(device),
                                ),
                                requested=requested_device,
                                label=(
                                    f"{partition} {attack_name} "
                                    f"graph {ready + 1}/{total}"
                                ),
                                announce=False,
                            )
                        except AttackCandidateRejected as exc:
                            rejection_counts["attack_contract"] += 1
                            last_rejection = str(exc)
                            if precomputed_path is not None:
                                rejected_precomputed.add(
                                    (str(precomputed_path), float(strength))
                                )
                            continue
                        except Exception as exc:
                            if precomputed_path is None:
                                raise
                            rejection_counts["precomputed_error"] += 1
                            last_rejection = (
                                f"{precomputed_path.name}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            rejected_precomputed.add(
                                (str(precomputed_path), float(strength))
                            )
                            continue
                        if not _compatible_existing_graph(
                            clean_graph,
                            candidate.graph,
                            runtime,
                            {"target_nodes": list(candidate.target_nodes)},
                            attack_name,
                            strength,
                        ):
                            rejection_counts["incompatible_contract"] += 1
                            if precomputed_path is not None:
                                rejected_precomputed.add(
                                    (str(precomputed_path), float(strength))
                                )
                            continue
                        digest = graph_fingerprint(candidate.graph)["combined"]
                        if digest == clean_fingerprint["combined"]:
                            rejection_counts["clean_graph"] += 1
                            if precomputed_path is not None:
                                rejected_precomputed.add(
                                    (str(precomputed_path), float(strength))
                                )
                            continue
                        if digest in seen_hashes:
                            rejection_counts["duplicate_graph"] += 1
                            if precomputed_path is not None:
                                used_precomputed_paths.add(str(precomputed_path))
                            continue
                        outcome, used_seed = candidate, seed
                        used_precomputed_path = precomputed_path
                        if precomputed_path is not None:
                            used_precomputed_paths.add(str(precomputed_path))
                        break
                    if outcome is None or used_seed is None:
                        details = ", ".join(
                            f"{name}={count}" for name, count in sorted(rejection_counts.items())
                        ) or "no_valid_candidate=0"
                        if last_rejection:
                            details += f"; last_rejection={last_rejection}"
                        raise RuntimeError(
                            f"could not generate unique {partition}/{attack_name}/{strength}/{ordinal} "
                            f"after {int(runtime['max_generation_attempts'])} attempts ({details})"
                        )
                    digest = graph_fingerprint(outcome.graph)["combined"]
                    seen_hashes.add(digest)
                    record = {
                        "sample_id": sample_id,
                        "dataset_id": profile_id,
                        "dataset": str(config["profiles"][profile_id]["name"]),
                        "partition": partition,
                        "label": 1,
                        "attack_name": attack_name,
                        "backend": expected_backend,
                        "generation_mode": (
                            "precomputed"
                            if used_precomputed_path is not None
                            else "generated"
                        ),
                        "precomputed_source_path": (
                            used_precomputed_path.relative_to(root).as_posix()
                            if used_precomputed_path is not None
                            else None
                        ),
                        "independent": True,
                        "strength_value": strength,
                        "strength_unit": spec.strength_unit,
                        "seed": used_seed,
                        "target_nodes": list(outcome.target_nodes),
                        "requested_modifications": outcome.requested_modifications,
                        "actual_modifications": outcome.actual_modifications,
                        "added_edges": [list(edge) for edge in outcome.added_edges],
                        "removed_edges": [list(edge) for edge in outcome.removed_edges],
                        "changed_features": [list(value) for value in outcome.changed_features],
                        "graph_path": relative.as_posix(),
                        "graph_hash": graph_fingerprint(outcome.graph)["combined"],
                    }
                    save_compact_graph(graph_path, outcome.graph, clean_fingerprint, record)
                    records.append(record)
                    _write_jsonl(manifest_path, records)
                    generated[partition] += 1
                    ready += 1
                    status_label = (
                        "REUSED-PRECOMPUTED"
                        if used_precomputed_path is not None
                        else "GENERATED"
                    )
                    print(
                        f"[{status_label}] {partition} {attack_name} "
                        f"{ready}/{total} saved={relative.as_posix()}"
                    )
            print(f"[ATTACK-COMPLETE] {partition} {attack_name}: {ready}/{total} graphs ready")

    records.sort(key=lambda item: (PARTITIONS.index(str(item["partition"])), str(item["sample_id"])))
    _write_jsonl(manifest_path, records)
    return {
        "status": "ok",
        "generated_samples": sum(generated.values()),
        "generated_by_partition": dict(generated),
        "reused_by_partition": dict(reused),
        "manifest_path": manifest_path.relative_to(root).as_posix(),
    }


def generate_test_corpus(
    config_path: str | Path | Mapping[str, Any],
    *,
    project_root: str | Path = Path("."),
    graph_loader: Callable[..., Any] | None = None,
    checkpoint_loader: Callable[..., Any] = load_victim_checkpoint,
    attack_runner: Callable[..., CompleteGraphAttackOutcome] = run_complete_graph_attack,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config_source(config_path)
    model_path = _resolve(root, config["detector"]["model_path"], "detector.model_path")
    if not model_path.is_file():
        raise FileNotFoundError("freeze the universal detector before generating test data")
    return generate_graph_corpus(
        config_path,
        partitions=("test",),
        project_root=root,
        graph_loader=graph_loader,
        checkpoint_loader=checkpoint_loader,
        attack_runner=attack_runner,
    )


def verify_graph_corpus(
    config_path: str | Path | Mapping[str, Any],
    *,
    expected_partitions: Sequence[str],
    project_root: str | Path = Path("."),
    graph_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config_source(config_path)
    runtime = pipeline_config(config)
    clean_graph = (
        load_graph_for_config(root, config)
        if graph_loader is None
        else graph_loader(root, config)
    ).clone().cpu()
    records = _read_jsonl(_resolve(root, runtime["sample_index_path"], "corpus.sample_index_path"))
    expected = set(str(value) for value in expected_partitions)
    unknown = {str(item.get("partition")) for item in records} - set(PARTITIONS)
    if unknown:
        raise RuntimeError(f"manifest contains unsupported partitions: {sorted(unknown)}")
    selected_attacks = set(str(value) for value in runtime["attacks"])
    profile_id = str(config["active_profile"])
    selected_records = [
        item
        for item in records
        if str(item.get("partition")) in expected
        and str(item.get("attack_name")) in selected_attacks
        and str(item.get("dataset_id")) == profile_id
    ]
    total = expected_corpus_count(runtime, expected, attacks=runtime["attacks"])
    if len(selected_records) != total:
        raise RuntimeError(
            f"manifest has {len(selected_records)} selected rows; expected {total}"
        )
    groups = Counter()
    for record in selected_records:
        expected_backend = attack_spec(str(record.get("attack_name"))).backend
        if str(record.get("backend")) != expected_backend:
            raise RuntimeError(
                "corpus record backend does not match the attack registry"
            )
        if str(record.get("dataset_id")) != profile_id:
            raise RuntimeError("corpus record belongs to another dataset profile")
        path = _resolve(root, str(record["graph_path"]), "corpus graph")
        graph = load_compact_graph(path, clean_graph)
        if graph_fingerprint(graph)["combined"] != record.get("graph_hash"):
            raise RuntimeError(f"graph hash mismatch: {path}")
        added, removed = edge_difference(clean_graph, graph)
        features = feature_difference(clean_graph, graph)
        actual = len(added) + len(removed) + len(features)
        if actual != int(record["actual_modifications"]):
            raise RuntimeError(f"attack audit mismatch: {path}")
        groups[(str(record["partition"]), str(record["attack_name"]), str(record["strength_value"]))] += 1
    for partition in expected:
        for attack_name in runtime["attacks"]:
            for strength in _strengths(runtime, attack_name):
                wanted = sample_count(runtime, attack_name, partition)
                if groups[(partition, attack_name, str(strength))] != wanted:
                    raise RuntimeError(f"corpus quota mismatch for {partition}/{attack_name}/{strength}")
    return {"status": "ok", "verified_samples": len(selected_records)}
