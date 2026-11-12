import unittest

import torch
from torch_geometric.data import Data

from gnn_attack_detector.graph_attacks import (
    feature_difference,
    run_global_protocol,
    run_multi_target_protocol,
    structural_budget,
)


def make_graph() -> Data:
    return Data(
        x=torch.eye(4),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        y=torch.tensor([0, 0, 1, 1]),
    )


class AttackDispatchTests(unittest.TestCase):
    def test_random_add_passes_explicit_add_operation(self) -> None:
        clean = make_graph()
        calls = []

        def adapter(graph, **kwargs):
            calls.append(kwargs)
            attacked = graph.clone()
            attacked.edge_index = torch.cat(
                [attacked.edge_index, torch.tensor([[0, 3], [3, 0]])], dim=1
            )
            return attacked

        result = run_global_protocol(
            clean, torch.nn.Linear(1, 1), "random_add", 0.34, 7,
            adapter=adapter, device="cuda",
        )
        self.assertEqual(structural_budget(clean, 0.34), 1)
        self.assertEqual(calls[0]["operation"], "add")
        self.assertEqual(calls[0]["device"], "cuda")
        self.assertEqual(result.actual_modifications, 1)

    def test_new_global_protocols_keep_their_declared_operation(self) -> None:
        expected = {
            "prbcd": "flip",
            "structack": "add",
            "fgattack_untargeted": "flip",
            "bridge_add": "add",
            "hub_rewire": "flip",
        }
        for attack_name, operation in expected.items():
            with self.subTest(attack_name=attack_name):
                clean = make_graph()
                calls = []

                def adapter(graph, **kwargs):
                    calls.append(kwargs)
                    attacked = graph.clone()
                    attacked.edge_index = torch.cat(
                        [
                            attacked.edge_index,
                            torch.tensor([[0, 3], [3, 0]]),
                        ],
                        dim=1,
                    )
                    return attacked

                result = run_global_protocol(
                    clean,
                    torch.nn.Linear(1, 1),
                    attack_name,
                    0.34,
                    7,
                    adapter=adapter,
                )
                self.assertEqual(calls[0]["operation"], operation)
                self.assertEqual(result.actual_modifications, 1)

    def test_fast_structural_protocols_use_the_live_global_dispatcher(self) -> None:
        for attack_name in ("bridge_add", "hub_rewire"):
            with self.subTest(attack_name=attack_name):
                clean = make_graph()
                result = run_global_protocol(
                    clean,
                    torch.nn.Linear(1, 1),
                    attack_name,
                    0.34,
                    7,
                )
                self.assertEqual(result.attack_name, attack_name)
                self.assertEqual(result.actual_modifications, 1)
                self.assertTrue(torch.equal(result.graph.x, clean.x))
                self.assertTrue(torch.equal(result.graph.y, clean.y))

    def test_gfattack_uses_the_targeted_protocol_contract(self) -> None:
        clean = make_graph()
        calls = []

        def adapter(graph, **kwargs):
            calls.append(kwargs)
            attacked = graph.clone()
            attacked.edge_index = torch.cat(
                [attacked.edge_index, torch.tensor([[0, 3], [3, 0]])],
                dim=1,
            )
            return attacked

        result = run_multi_target_protocol(
            clean,
            torch.nn.Linear(1, 1),
            "gfattack",
            [0],
            1,
            7,
            adapter=adapter,
            device="cpu",
        )
        self.assertEqual(calls[0]["operation"], "structure")
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertEqual(result.actual_modifications, 1)

    def test_nettack_feature_records_feature_changes(self) -> None:
        clean = make_graph()

        def adapter(graph, **kwargs):
            attacked = graph.clone()
            attacked.x[int(kwargs["target_node"]), 0] = 1.0
            return attacked

        result = run_multi_target_protocol(
            clean, torch.nn.Linear(1, 1), "nettack_feature", [1], 1, 9,
            adapter=adapter,
        )
        self.assertEqual(result.changed_features, ((1, 0),))
        self.assertEqual(feature_difference(clean, result.graph), {(1, 0)})


if __name__ == "__main__":
    unittest.main()
