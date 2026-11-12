import unittest


class ResultViewerTests(unittest.TestCase):
    def _report(self):
        return {
            "format_version": 3,
            "profiles": ["cora-lcc", "pubmed-lcc"],
            "test_metrics": {
                "count": 8,
                "accuracy": 0.625,
                "recall": 0.75,
                "specificity": 0.5,
                "precision": 0.6,
                "f1": 2 / 3,
                "confusion_matrix": {"tp": 3, "tn": 2, "fp": 2, "fn": 1},
            },
            "metrics_by_profile": {
                "cora-lcc": {
                    "count": 4,
                    "accuracy": 0.75,
                    "recall": 1.0,
                    "specificity": 0.5,
                    "precision": 2 / 3,
                    "f1": 0.8,
                    "confusion_matrix": {"tp": 2, "tn": 1, "fp": 1, "fn": 0},
                },
                "pubmed-lcc": {
                    "count": 4,
                    "accuracy": 0.5,
                    "recall": 0.5,
                    "specificity": 0.5,
                    "precision": 0.5,
                    "f1": 0.5,
                    "confusion_matrix": {"tp": 1, "tn": 1, "fp": 1, "fn": 1},
                },
            },
            "attacked_complete_graph_recall": {
                "by_attack": {
                    "dice_add": {"total": 4, "detected": 2, "recall": 0.5},
                    "random_add": {"total": 4, "detected": 3, "recall": 0.75},
                },
                "overall": {"total": 8, "detected": 5, "recall": 0.625},
            },
            "acceptance": {
                "minimum": 0.8,
                "minimum_attack_count": 20,
                "minimum_test_graphs_per_attack": 20,
                "passed": False,
            },
        }

    def test_terminal_report_prints_actual_percentages_and_counts(self) -> None:
        from gnn_attack_detector.result_viewer import format_evaluation_report

        output = format_evaluation_report(self._report())

        self.assertIn("综合准确率: 62.50% (5/8)", output)
        self.assertIn("cora-lcc: 准确率 75.00%", output)
        self.assertIn("pubmed-lcc: 准确率 50.00%", output)
        self.assertIn("random_add: 75.00% (3/4)", output)
        self.assertIn("dice_add: 50.00% (2/4)", output)
        self.assertIn("配置的验收阈值: 80.00%", output)
        self.assertIn("最终验收状态: 未通过", output)

    def test_terminal_report_requires_real_metrics(self) -> None:
        from gnn_attack_detector.result_viewer import format_evaluation_report

        with self.assertRaisesRegex(ValueError, "test_metrics"):
            format_evaluation_report({"acceptance": {"minimum": 0.8}})


if __name__ == "__main__":
    unittest.main()
