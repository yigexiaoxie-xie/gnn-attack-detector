import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CurrentConfigSourceTests(unittest.TestCase):
    def test_base_config_uses_profiles_attacks_corpus_and_universal_detector(self) -> None:
        source = (ROOT / "configs/base.yaml").read_text(encoding="utf-8")
        for heading in ("profiles:", "active_profile:", "attacks:", "corpus:", "detector:", "defenses:"):
            self.assertIn(heading, source)
        for obsolete in ("day4:", "victim:", "injection_node_counts", "backdoor_feature_budget", "twenty-two"):
            self.assertNotIn(obsolete, source)

    def test_config_validator_has_no_day_number_or_fixed_twenty_two_message(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/config.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("DAY4", source)
        self.assertNotIn('loaded.get("day4")', source)
        self.assertNotIn("twenty-two", source)
        self.assertIn("load_profiles", source)

    def test_detector_group_targets_are_explicit(self) -> None:
        source = (ROOT / "configs/base.yaml").read_text(encoding="utf-8")
        for setting in (
            "target_group_recall: 0.80",
            "minimum_clean_specificity: 0.80",
            "target_consecutive_epochs: 3",
            "minimum_validation_group_size: 8",
        ):
            self.assertIn(setting, source)


if __name__ == "__main__":
    unittest.main()
