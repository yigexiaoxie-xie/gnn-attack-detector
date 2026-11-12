from pathlib import Path
import tempfile
import unittest

import yaml

from gnn_attack_detector.attacks.registry import SUPPORTED_ATTACKS
from gnn_attack_detector.config import load_config, load_config_source, pipeline_config
from gnn_attack_detector.defenses.registry import SUPPORTED_DEFENSES


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_base_configuration_matches_current_contract(self) -> None:
        config = load_config(ROOT / "configs" / "base.yaml")
        runtime = pipeline_config(config)

        self.assertEqual(tuple(config["attacks"]["protocols"]), SUPPORTED_ATTACKS)
        self.assertEqual(tuple(config["defenses"]["models"]), SUPPORTED_DEFENSES)
        self.assertEqual(config["probes"]["online"], ["gcn", "gcn_jaccard", "simpgcn"])
        self.assertEqual(runtime["model_path"], "artifacts/models/universal-attack-detector.pt")
        self.assertEqual(
            runtime["sample_index_path"],
            "artifacts/features/manifests/cora-lcc.jsonl",
        )
        self.assertNotIn("feature_path", config["corpus"])
        self.assertNotIn("day4", config)
        self.assertNotIn("victim", config)

    def test_base_configuration_registers_four_lcc_profiles(self) -> None:
        config = load_config(ROOT / "configs" / "base.yaml")

        self.assertEqual(
            tuple(config["profiles"]),
            ("cora-lcc", "citeseer-lcc", "polblogs-lcc", "pubmed-lcc"),
        )
        self.assertEqual(config["profiles"]["citeseer-lcc"]["expected_num_nodes"], 2110)
        self.assertEqual(config["profiles"]["polblogs-lcc"]["expected_num_nodes"], 1222)
        self.assertEqual(config["profiles"]["pubmed-lcc"]["expected_num_nodes"], 19717)

    def test_rejects_absolute_profile_path(self) -> None:
        raw = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
        raw["profiles"]["cora-lcc"]["clean_reference"] = "E:/machine/clean.pt"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean_reference must be relative"):
                load_config(path)

    def test_validated_mapping_source_is_copied_for_in_memory_profiles(self) -> None:
        raw = yaml.safe_load(
            (ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
        )
        loaded = load_config_source(raw)
        loaded["active_profile"] = "citeseer-lcc"
        self.assertEqual(raw["active_profile"], "cora-lcc")


if __name__ == "__main__":
    unittest.main()
