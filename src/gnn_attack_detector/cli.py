"""Command-line interface for profile preparation, training, and detection."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Sequence

from gnn_attack_detector.attacks.registry import attack_spec
from gnn_attack_detector.config import load_config, pipeline_config
from gnn_attack_detector.corpus_policy import expected_corpus_count
from gnn_attack_detector.run_selection import load_run_selection


SYSTEM_REQUIREMENTS = {
    "operating_system": "Windows 10/11 x64 or Linux x64",
    "python": "3.10 or 3.11",
    "memory": "16 GB minimum; 32 GB recommended for attack generation",
    "disk": "10 GB plus generated graph artifacts",
    "gpu": "optional; CPU mode is supported",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnn-attack-detector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check-config", "validate the toolkit configuration"),
        ("prepare", "prepare the active profile and attack GCN surrogate"),
        ("prepare-defenses", "prepare known datasets used by defense evaluation"),
        ("prepare-detector-data", "generate train/validation attack graphs and features"),
        ("train-detector", "train the one universal detector"),
        ("generate-detector-test-data", "generate frozen held-out test graphs"),
        ("evaluate-detector", "evaluate the checkpoint's profiles or an explicit selection"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/base.yaml")
        if name in {"train-detector", "evaluate-detector"}:
            command.add_argument("--profiles", nargs="+", default=None)
        if name == "prepare-defenses":
            command.add_argument("--profiles", nargs="+", default=None)
        if name == "evaluate-detector":
            command.add_argument("--model", default=None)

    for name, help_text in (
        ("show-selection", "preview selected profiles, attacks, and graph count"),
        ("run-selected", "run the selected resumable workflow"),
        ("rebuild-detector", "train shared GCN using existing selected attacks only"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/base.yaml")
        command.add_argument("--run-config", default="configs/run.yaml")
    subparsers.add_parser("show-system-requirements")

    detect = subparsers.add_parser("detect-graph", help="return one verdict for one graph")
    detect.add_argument("--config", default="configs/base.yaml")
    detect.add_argument("--profile", default=None)
    detect.add_argument("--graph", required=True)
    detect.add_argument("--attribute-mode", choices=("auto", "content", "none"), default="auto")

    activate = subparsers.add_parser("activate-detector", help="explicitly activate a saved candidate model")
    activate.add_argument("--config", default="configs/base.yaml")
    activate.add_argument("--model", required=True)
    export = subparsers.add_parser("export-graph", help="export a self-contained reference-free NPZ")
    export.add_argument("--config", default="configs/base.yaml")
    export.add_argument("--graph", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--profile", default=None)

    generate = subparsers.add_parser("generate-attack", help="generate attacked graphs")
    generate.add_argument("--config", default="configs/base.yaml")
    generate.add_argument("--profile", default=None)
    generate.add_argument("--attack", required=True)
    generate.add_argument("--strength", required=True, type=float)
    generate.add_argument("--count", default=1, type=int)
    return parser


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def selection_summary(
    config_path: str | Path,
    run_config: str | Path,
) -> dict[str, object]:
    config = load_config(config_path)
    selection = load_run_selection(
        run_config,
        known_profiles=tuple(config["profiles"]),
        known_attacks=tuple(config["attacks"]["protocols"]),
    )
    runtime = pipeline_config(config)
    expected_per_profile = expected_corpus_count(
        runtime,
        selection.partitions,
        attacks=selection.attacks,
    )
    required_modules: set[str] = set()
    for attack_name in selection.attacks:
        spec = attack_spec(attack_name)
        if spec.backend == "deeprobust":
            required_modules.add("deeprobust")
        if spec.backend == "greatx":
            required_modules.update(("greatx", "torch_cluster", "torch_sparse"))
        if attack_name == "prbcd":
            required_modules.add("torch_sparse")
    missing = sorted(
        name for name in required_modules if not _module_available(name)
    )
    return {
        "status": "ok" if not missing else "missing_dependencies",
        "profiles": list(selection.profiles),
        "attacks": list(selection.attacks),
        "partitions": list(selection.partitions),
        "combination_count": selection.combination_count,
        "expected_attacked_graphs_per_profile": expected_per_profile,
        "expected_attacked_graphs": expected_per_profile * len(selection.profiles),
        "missing_dependencies": missing,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "show-system-requirements":
        _print(SYSTEM_REQUIREMENTS)
        return 0

    config_path = Path(args.config)
    if args.command in {"show-selection", "run-selected"}:
        summary = selection_summary(config_path, Path(args.run_config))
        if args.command == "show-selection":
            _print(summary)
            return 0
        if summary["missing_dependencies"]:
            _print(summary)
            return 2
        from gnn_attack_detector.workflow import run_selected_workflow

        result = run_selected_workflow(
            config_path,
            Path(args.run_config),
            project_root=Path.cwd(),
        )
        _print(result)
        return 0 if result.get("status") == "ok" else 1

    if args.command == "check-config":
        config = load_config(config_path)
        _print(
            {
                "status": "ok",
                "active_profile": config["active_profile"],
                "profiles": list(config["profiles"]),
                "online_defense_models": config["probes"]["online"],
                "detector_architecture": config["detector"]["model"]["type"],
                "detector_requires_reference": False,
                "attack_protocols": len(config["attacks"]["protocols"]),
                "defense_models": len(config["defenses"]["models"]),
                "device": config["runtime"]["device"],
            }
        )
        return 0

    if args.command == "rebuild-detector":
        from gnn_attack_detector.pipeline import rebuild_detector
        result = rebuild_detector(config_path, args.run_config, project_root=Path.cwd())
    elif args.command == "activate-detector":
        from gnn_attack_detector.pipeline import activate_detector
        result = activate_detector(config_path, args.model, project_root=Path.cwd())
    elif args.command == "export-graph":
        from gnn_attack_detector.pipeline import _load_detection_input
        from gnn_attack_detector.graph_io import graph_to_npz_bytes
        config = load_config(config_path)
        graph, provenance = _load_detection_input(Path.cwd(), config, Path(args.graph).resolve(), args.profile)
        graph.attribute_mode = config["detector"].get("attribute_modes", {}).get(provenance, getattr(graph, "attribute_mode", "auto"))
        output = Path(args.output)
        if output.suffix.lower() != ".npz" or output.exists():
            raise ValueError("output must be a new .npz file; existing files are never overwritten")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(graph_to_npz_bytes(graph))
        result = {"status": "ok", "output": str(output), "source_profile": provenance}
    elif args.command in {"prepare", "prepare-defenses"}:
        from gnn_attack_detector.preparation import prepare_project
        if args.command == "prepare-defenses" and args.profiles:
            base_config = load_config(config_path)
            unknown = sorted(set(args.profiles) - set(base_config["profiles"]))
            if unknown:
                raise ValueError(f"unknown profile: {unknown}")
            prepared = {}
            for profile_id in args.profiles:
                selected = deepcopy(base_config)
                selected["active_profile"] = profile_id
                prepared[profile_id] = prepare_project(
                    selected, project_root=Path.cwd(), include_defenses=True
                )
            result = {"status": "ok", "profiles": prepared}
        else:
            result = prepare_project(
                config_path,
                project_root=Path.cwd(),
                include_defenses=args.command == "prepare-defenses",
            )
    elif args.command == "prepare-detector-data":
        from gnn_attack_detector.pipeline import prepare_detector_data

        result = prepare_detector_data(config_path, project_root=Path.cwd())
    elif args.command == "train-detector":
        from gnn_attack_detector.pipeline import train_detector

        result = train_detector(config_path, project_root=Path.cwd(), profile_ids=args.profiles)
    elif args.command == "generate-detector-test-data":
        from gnn_attack_detector.pipeline import generate_detector_test_data

        result = generate_detector_test_data(config_path, project_root=Path.cwd())
    elif args.command == "evaluate-detector":
        from gnn_attack_detector.pipeline import evaluate_detector

        evaluation_config = load_config(config_path)
        if args.model:
            evaluation_config["detector"]["model_path"] = args.model
        result = evaluate_detector(evaluation_config, project_root=Path.cwd(), profile_ids=args.profiles)
    elif args.command == "detect-graph":
        from gnn_attack_detector.pipeline import detect_graph

        result = detect_graph(
            config_path,
            Path(args.graph),
            project_root=Path.cwd(),
            profile_id=args.profile,
            attribute_mode=args.attribute_mode,
        )
    elif args.command == "generate-attack":
        from gnn_attack_detector.platform_generation import generate_attack_batch
        from gnn_attack_detector.platform_service import load_platform_generation_context

        root = Path.cwd().resolve()
        context = load_platform_generation_context(config_path, root, profile_id=args.profile)
        runtime = pipeline_config(context["config"])
        result = generate_attack_batch(
            clean_graph=context["clean_graph"],
            surrogate=context["surrogate"],
            dataset_name=context["profile"].identifier,
            attack_name=args.attack,
            strength=args.strength,
            count=args.count,
            output_root=root / "artifacts" / "platform" / "generated",
            structural_rates=tuple(runtime["structural_rates"]),
            targeted_target_counts=tuple(runtime["targeted_target_counts"]),
            targeted_local_budget=int(runtime["targeted_local_budget"]),
            minimum_target_degree=int(runtime["minimum_target_degree"]),
            max_attempts=min(int(runtime["max_generation_attempts"]), 30),
        )
    else:
        raise RuntimeError(f"unsupported command: {args.command}")
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
