from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ActualMetricReportingSourceTests(unittest.TestCase):
    def test_evaluation_contains_per_profile_actual_metrics(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/pipeline.py").read_text(encoding="utf-8")

        self.assertIn('"metrics_by_profile"', source)
        self.assertIn("binary_metrics(profile_labels, profile_guesses)", source)

    def test_web_page_does_not_hardcode_eighty_percent_as_a_result(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/web_app.py").read_text(encoding="utf-8")
        evaluation = source.split("def _evaluation", 1)[1].split("def _defenses", 1)[0]

        self.assertNotIn("80%", evaluation)
        self.assertNotIn("≥80", evaluation)
        self.assertIn('metrics["accuracy"]', evaluation)
        self.assertNotIn('acceptance["minimum"]', evaluation)
        self.assertNotIn("是否达标", evaluation)
        self.assertNotIn("配置要求", evaluation)

    def test_pycharm_result_entry_point_exists(self) -> None:
        source = (ROOT / "show_results.py").read_text(encoding="utf-8")

        self.assertIn("gnn_attack_detector.result_viewer", source)
        self.assertIn("artifacts", source)


if __name__ == "__main__":
    unittest.main()
