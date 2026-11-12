import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AttackBackendSourceTests(unittest.TestCase):
    def test_graph_dispatch_keeps_external_backends_in_adapters(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/graph_attacks.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("greatx_adapter", source)
        self.assertIn("structack_adapter", source)
        self.assertNotIn("from greatx.", source)
        self.assertNotIn("pyg_adapter", source)
        self.assertNotIn("project_adapter", source)
        self.assertNotIn("INJECTION_ATTACKS", source)
        self.assertNotIn("BACKDOOR_ATTACKS", source)
        self.assertIn("int(edge_count * value)", source)
        self.assertNotIn("round(edge_count * value)", source)

    def test_deeprobust_adapter_supports_protocol_parameters_and_igattack(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/deeprobust_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def run_deeprobust_igattack", source)
        self.assertIn("def run_deeprobust_prbcd", source)
        self.assertIn("attack_type: str = \"flip\"", source)
        self.assertIn("attack_mode: str = \"structure\"", source)
        self.assertNotIn("def run_deeprobust_node_embedding", source)
        self.assertNotIn("def run_deeprobust_eigenvector_attack", source)
        self.assertNotIn("run_deeprobust_ugba", source)


if __name__ == "__main__":
    unittest.main()
