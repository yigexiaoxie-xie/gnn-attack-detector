from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProbeContractSourceTests(unittest.TestCase):
    def test_real_deeprobust_robust_models_are_trained(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/probes.py").read_text(encoding="utf-8")
        self.assertIn("GCNJaccard", source)
        self.assertIn("SimPGCN", source)
        self.assertIn("jaccard.fit", source)
        self.assertIn("simpgcn.fit", source)

    def test_pipeline_merges_profile_datasets_into_one_model(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/pipeline.py").read_text(encoding="utf-8")
        self.assertIn("merge_graph_detector_datasets", source)
        self.assertIn('select_graph_detector_partitions(item, ("train", "val"))', source)
        self.assertIn("universal-attack-detector", (ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("reference_graph_hash", source)
        self.assertNotIn("victim_model_hash", source)

    def test_online_defense_benchmark_is_executable(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/defense_service.py").read_text(encoding="utf-8")
        self.assertIn("evaluate_robust_probe", source)
        self.assertIn("clean_accuracy", source)
        self.assertIn("input_graph_accuracy", source)

    def test_prepare_does_not_build_defense_checkpoints(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/preparation.py").read_text(encoding="utf-8")
        self.assertNotIn("prepare_profile_defenses", source)
        self.assertNotIn('report["defenses"]', source)


if __name__ == "__main__":
    unittest.main()
