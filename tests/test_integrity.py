from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import Data

from gnn_attack_detector.integrity import (
    evaluate_model,
    graph_fingerprint,
    hash_file,
    hash_tensor,
    modification_signature,
)


class FakeModel(torch.nn.Module):
    def forward(self, x, edge_index):
        logits = torch.tensor([[3.0, 1.0], [0.5, 2.5]])
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        return logits, embeddings


class IntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = Data(
            x=torch.tensor([[1.0], [0.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.tensor([0, 1], dtype=torch.long),
        )

    def test_file_hash_changes_with_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            path.write_bytes(b"first")
            first = hash_file(path)
            path.write_bytes(b"second")
            second = hash_file(path)

        self.assertNotEqual(first, second)

    def test_tensor_hash_is_stable_for_equal_values(self) -> None:
        first = torch.tensor([[1.0, 2.0]])
        second = torch.tensor([[1.0, 2.0]])

        self.assertEqual(hash_tensor(first), hash_tensor(second))

    def test_tensor_hash_changes_when_one_value_changes(self) -> None:
        first = torch.tensor([[1.0, 2.0]])
        second = torch.tensor([[1.0, 3.0]])

        self.assertNotEqual(hash_tensor(first), hash_tensor(second))

    def test_graph_fingerprint_covers_all_attack_inputs(self) -> None:
        result = graph_fingerprint(self.graph)

        self.assertEqual(
            set(result),
            {"x", "edge_index", "y", "combined"},
        )

    def test_modification_signature_is_order_independent_and_auditable(self) -> None:
        first = modification_signature(
            target_nodes=[4, 1],
            added_edges=[(3, 2), (0, 5)],
            removed_edges=[(8, 7)],
        )
        second = modification_signature(
            target_nodes=[1, 4],
            added_edges=[(5, 0), (2, 3)],
            removed_edges=[(7, 8)],
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_evaluate_model_records_all_node_signals(self) -> None:
        signals = evaluate_model(FakeModel(), self.graph)

        self.assertEqual(signals["predictions"], [0, 1])
        self.assertEqual(len(signals["probabilities"]), 2)
        self.assertEqual(len(signals["embeddings"]), 2)
        self.assertEqual(signals["accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()
