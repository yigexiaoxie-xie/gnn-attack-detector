import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UniversalDetectorSourceContractTests(unittest.TestCase):
    def test_disk_dataset_persists_dataset_ids_in_version_four(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/detector_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("dataset_ids: tuple[str, ...]", source)
        self.assertIn("payload['format_version'] = 4", source)

    def test_detector_bundle_is_not_bound_to_one_reference_graph_or_victim(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/detector_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("trained_profiles: tuple[str, ...]", source)
        self.assertNotIn("probe_names", source)
        self.assertNotIn("reference_graph_hash", source)
        self.assertNotIn("victim_model_hash", source)
        self.assertIn("'format_version': 4", source)

    def test_acceptance_requires_independent_clean_graphs(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"independent_clean_graphs_eligible"', source)
        self.assertIn("official_eligible", source)
        self.assertIn("and official_eligible", source)


if __name__ == "__main__":
    unittest.main()
