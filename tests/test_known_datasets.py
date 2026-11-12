from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class KnownDatasetSourceTests(unittest.TestCase):
    def test_missing_features_use_deterministic_structural_columns(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/data.py").read_text(encoding="utf-8")

        self.assertIn("def _structural_features", source)
        self.assertIn("maximum_degree", source)
        self.assertIn("np.ones", source)
        self.assertIn("features is None", source)

    def test_deeprobust_loader_removes_upstream_self_loops(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/data.py").read_text(
            encoding="utf-8"
        )
        loader = source.split("def load_deeprobust_graph", 1)[1]

        self.assertIn("adjacency.setdiag(0)", loader)
        self.assertIn("adjacency.eliminate_zeros()", loader)
        self.assertLess(
            loader.index("adjacency.setdiag(0)"),
            loader.index("_structural_features(adjacency)"),
        )

    def test_cached_profiles_with_self_loops_are_rebuilt(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/preparation.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("graph_has_self_loops", source)
        self.assertIn("saved clean graph contains self-loops", source)


if __name__ == "__main__":
    unittest.main()
