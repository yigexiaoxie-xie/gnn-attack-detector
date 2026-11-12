from pathlib import Path
import tempfile
import unittest

from gnn_attack_detector.run_selection import load_run_selection


class RunSelectionTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.yaml"
            path.write_text(text, encoding="utf-8")
            return load_run_selection(
                path,
                known_profiles=("cora-lcc", "citeseer-lcc", "pubmed-lcc"),
                known_attacks=("random_add", "dice_mix", "igattack"),
            )

    def test_loads_multiple_profiles_attacks_and_partitions_in_order(self) -> None:
        selected = self._load(
            """run:
  profiles: [cora-lcc, citeseer-lcc]
  attacks: [random_add, dice_mix]
  partitions: [train, val, test]
"""
        )

        self.assertEqual(selected.profiles, ("cora-lcc", "citeseer-lcc"))
        self.assertEqual(selected.attacks, ("random_add", "dice_mix"))
        self.assertEqual(selected.partitions, ("train", "val", "test"))

    def test_rejects_empty_duplicate_and_unknown_profiles(self) -> None:
        cases = (
            ("[]", "run.profiles must contain at least one item"),
            ("[cora-lcc, cora-lcc]", "run.profiles must contain unique items"),
            ("[unknown]", "unknown profile"),
        )
        for profiles, message in cases:
            with self.subTest(profiles=profiles):
                with self.assertRaisesRegex(ValueError, message):
                    self._load(
                        "run:\n"
                        f"  profiles: {profiles}\n"
                        "  attacks: [random_add]\n"
                        "  partitions: [train]\n"
                    )

    def test_rejects_empty_duplicate_and_unknown_attacks(self) -> None:
        cases = (
            ("[]", "run.attacks must contain at least one item"),
            ("[random_add, random_add]", "run.attacks must contain unique items"),
            ("[unknown]", "unknown attack"),
        )
        for attacks, message in cases:
            with self.subTest(attacks=attacks):
                with self.assertRaisesRegex(ValueError, message):
                    self._load(
                        "run:\n"
                        "  profiles: [cora-lcc]\n"
                        f"  attacks: {attacks}\n"
                        "  partitions: [train]\n"
                    )

    def test_rejects_duplicate_and_unknown_partitions(self) -> None:
        cases = (
            ("[train, train]", "run.partitions must contain unique items"),
            ("[other]", "unknown partition"),
        )
        for partitions, message in cases:
            with self.subTest(partitions=partitions):
                with self.assertRaisesRegex(ValueError, message):
                    self._load(
                        "run:\n"
                        "  profiles: [cora-lcc]\n"
                        "  attacks: [random_add]\n"
                        f"  partitions: {partitions}\n"
                    )


if __name__ == "__main__":
    unittest.main()
