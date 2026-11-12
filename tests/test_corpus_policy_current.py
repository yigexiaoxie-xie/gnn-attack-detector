import unittest

from gnn_attack_detector.attacks.registry import SUPPORTED_ATTACKS
from gnn_attack_detector.corpus_policy import (
    classify_attack_availability,
    expected_corpus_count,
    sample_count,
)


class CorpusPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = {
            "attacks": list(SUPPORTED_ATTACKS),
            "structural_rates": [0.05, 0.10, 0.15, 0.20],
            "targeted_target_counts": [5, 10, 15, 20],
            "sample_policy": {
                "default": {"train": 5, "val": 2, "test": 5},
                "overrides": {},
            },
        }

    def test_every_attack_uses_the_same_48_graph_quota(self) -> None:
        self.assertEqual(len(SUPPORTED_ATTACKS), 23)
        for attack_name in SUPPORTED_ATTACKS:
            self.assertEqual(sample_count(self.runtime, attack_name, "train"), 5)
            self.assertEqual(sample_count(self.runtime, attack_name, "val"), 2)
            self.assertEqual(sample_count(self.runtime, attack_name, "test"), 5)
        self.assertEqual(expected_corpus_count(self.runtime), 1104)
        self.assertEqual(expected_corpus_count(self.runtime, ("train", "val")), 644)
        self.assertEqual(expected_corpus_count(self.runtime, ("test",)), 460)

    def test_metattack_is_not_given_a_smaller_quota(self) -> None:
        self.assertEqual(sample_count(self.runtime, "metattack", "train"), 5)
        self.assertEqual(sample_count(self.runtime, "metattack", "val"), 2)
        self.assertEqual(sample_count(self.runtime, "metattack", "test"), 5)

    def test_absent_attacks_are_unavailable_not_incomplete(self) -> None:
        runtime = {
            **self.runtime,
            "structural_rates": [0.05],
            "targeted_target_counts": [5],
            "sample_policy": {
                "default": {"train": 1, "val": 1, "test": 1},
                "overrides": {},
            },
        }
        records = [
            {
                "partition": partition,
                "attack_name": "random_add",
                "strength_value": 0.05,
            }
            for partition in ("train", "val", "test")
        ]

        result = classify_attack_availability(
            runtime,
            records,
            attacks=("random_add", "fga"),
            partitions=("train", "val", "test"),
        )

        self.assertEqual(result["available_attacks"], ("random_add",))
        self.assertEqual(result["unavailable_attacks"], ("fga",))

    def test_partially_present_attack_is_skipped_as_a_whole(self) -> None:
        runtime = {
            **self.runtime,
            "structural_rates": [0.05],
            "targeted_target_counts": [5],
            "sample_policy": {
                "default": {"train": 1, "val": 1, "test": 1},
                "overrides": {},
            },
        }
        records = [
            {
                "partition": "train",
                "attack_name": "random_add",
                "strength_value": 0.05,
            },
            *[
                {
                    "partition": partition,
                    "attack_name": "random_delete",
                    "strength_value": 0.05,
                }
                for partition in ("train", "val", "test")
            ],
        ]

        result = classify_attack_availability(
            runtime,
            records,
            attacks=("random_add", "random_delete"),
            partitions=("train", "val", "test"),
        )

        self.assertEqual(result["available_attacks"], ("random_delete",))
        self.assertEqual(result["unavailable_attacks"], ("random_add",))
        self.assertRegex(
            result["unavailable_reasons"]["random_add"],
            r"val/0.05 expected 1, found 0",
        )

    def test_rebuild_is_rejected_when_every_attack_is_incomplete(self) -> None:
        runtime = {
            **self.runtime,
            "structural_rates": [0.05],
            "targeted_target_counts": [5],
            "sample_policy": {
                "default": {"train": 1, "val": 1, "test": 1},
                "overrides": {},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "no complete selected attacks"):
            classify_attack_availability(
                runtime,
                [{
                    "partition": "train",
                    "attack_name": "random_add",
                    "strength_value": 0.05,
                }],
                attacks=("random_add",),
                partitions=("train", "val", "test"),
            )


if __name__ == "__main__":
    unittest.main()
