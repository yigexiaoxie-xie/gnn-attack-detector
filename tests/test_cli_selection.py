from pathlib import Path
import unittest

from gnn_attack_detector.cli import build_parser, selection_summary


ROOT = Path(__file__).resolve().parents[1]


class CliSelectionTests(unittest.TestCase):
    def test_parser_accepts_show_and_run_selected_commands(self) -> None:
        parser = build_parser()
        for command in ("show-selection", "run-selected"):
            args = parser.parse_args(
                [
                    command,
                    "--config",
                    "configs/base.yaml",
                    "--run-config",
                    "configs/run.yaml",
                ]
            )
            self.assertEqual(args.command, command)
            self.assertEqual(args.run_config, "configs/run.yaml")

    def test_default_selection_previews_the_exact_graph_count(self) -> None:
        summary = selection_summary(
            ROOT / "configs" / "base.yaml",
            ROOT / "configs" / "run.yaml",
        )
        self.assertEqual(summary["profiles"], ["cora-lcc"])
        self.assertEqual(summary["attacks"], ["random_add", "random_delete", "random_flip"])
        self.assertEqual(summary["combination_count"], 3)
        self.assertEqual(summary["expected_attacked_graphs"], 144)

    def test_prepare_defenses_accepts_explicit_profiles(self) -> None:
        args = build_parser().parse_args(
            ["prepare-defenses", "--profiles", "cora-lcc", "pubmed-lcc"]
        )
        self.assertEqual(args.profiles, ["cora-lcc", "pubmed-lcc"])


if __name__ == "__main__":
    unittest.main()
