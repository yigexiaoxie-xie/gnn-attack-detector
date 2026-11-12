import unittest

import torch

import gnn_attack_detector.detector_metrics as detector_metrics
from gnn_attack_detector.detector_metrics import (
    attacked_graph_recall,
    binary_metrics,
    calibrate_threshold,
    calibrate_group_threshold,
    official_graph_metrics,
)


class BinaryMetricTests(unittest.TestCase):
    def test_confusion_counts_and_formulas_are_exact(self) -> None:
        metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])

        self.assertEqual(metrics["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)

    def test_calibration_uses_only_supplied_validation_graph_probabilities(self) -> None:
        result = calibrate_threshold(
            probabilities=torch.tensor([0.1, 0.2, 0.8, 0.9]),
            labels=torch.tensor([0, 0, 1, 1]),
        )

        self.assertAlmostEqual(result["threshold"], 0.8, places=6)
        self.assertEqual(result["metrics"]["balanced_accuracy"], 1.0)

    def test_group_calibration_protects_clean_graphs_and_the_worst_attack(self) -> None:
        result = calibrate_group_threshold(
            probabilities=torch.tensor([0.40, 0.35, 0.45, 0.46, 0.90, 0.95, 0.50, 0.51]),
            labels=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
            profiles=("a", "b", "a", "a", "a", "a", "b", "b"),
            attacks=("clean", "clean", "hard", "hard", "easy", "easy", "hard", "hard"),
            minimum_group_recall=0.80,
            minimum_clean_specificity=0.80,
            minimum_group_size=2,
        )

        self.assertAlmostEqual(result["threshold"], 0.45, places=6)
        self.assertEqual(result["clean_specificity"], 1.0)
        self.assertEqual(result["worst_group_recall"], 1.0)
        self.assertEqual(result["groups_at_target"], 3)
        self.assertEqual(result["total_groups"], 3)
        self.assertTrue(result["target_met"])

    def test_group_calibration_rejects_small_validation_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation graphs"):
            calibrate_group_threshold(
                probabilities=torch.tensor([0.1, 0.9]),
                labels=torch.tensor([0, 1]),
                profiles=("a", "a"),
                attacks=("clean", "hard"),
                minimum_group_recall=0.80,
                minimum_clean_specificity=0.80,
                minimum_group_size=2,
            )

    def test_group_calibration_does_not_pass_by_predicting_everything_attacked(self) -> None:
        result = calibrate_group_threshold(
            probabilities=torch.tensor([0.90, 0.80, 0.75]),
            labels=torch.tensor([0, 1, 1]),
            profiles=("a", "a", "a"),
            attacks=("clean", "hard", "hard"),
            minimum_group_recall=0.80,
            minimum_clean_specificity=0.80,
            minimum_group_size=2,
        )

        self.assertEqual(result["clean_specificity"], 1.0)
        self.assertEqual(result["worst_group_recall"], 0.0)
        self.assertFalse(result["target_met"])


class CompleteGraphMetricTests(unittest.TestCase):
    def test_maximum_node_graph_heuristic_no_longer_exists(self) -> None:
        self.assertFalse(hasattr(detector_metrics, "graph_attack_score"))

    def test_recall_is_reported_by_attack_strength_attack_and_overall(self) -> None:
        records = [
            {"sample_id": "r5a", "attack_name": "random", "strength_value": 0.05, "strength_unit": "edge_rate"},
            {"sample_id": "r5b", "attack_name": "random", "strength_value": 0.05, "strength_unit": "edge_rate"},
            {"sample_id": "r10", "attack_name": "random", "strength_value": 0.10, "strength_unit": "edge_rate"},
            {"sample_id": "n5", "attack_name": "nettack", "strength_value": 5, "strength_unit": "target_nodes"},
        ]
        report = attacked_graph_recall(
            records,
            graph_predictions={"r5a": 1, "r5b": 0, "r10": 1, "n5": 1},
        )

        self.assertEqual(report["by_attack_strength"]["random"]["5%"]["recall"], 0.5)
        self.assertEqual(report["by_attack_strength"]["nettack"]["5 nodes"]["recall"], 1.0)
        self.assertAlmostEqual(report["by_attack"]["random"]["recall"], 2 / 3)
        self.assertEqual(report["overall"]["recall"], 0.75)

    def test_one_independent_clean_graph_cannot_claim_official_accuracy(self) -> None:
        report = official_graph_metrics(
            clean_predictions=[0],
            attacked_predictions=[1, 1, 0],
            independent_clean_graphs=1,
            minimum_clean_graphs=2,
        )

        self.assertFalse(report["official_graph_metric_eligible"])
        self.assertIsNone(report["official_graph_accuracy"])
        self.assertIn("insufficient independent clean complete graphs", report["reason"])


if __name__ == "__main__":
    unittest.main()
