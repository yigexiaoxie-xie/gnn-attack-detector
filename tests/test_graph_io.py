from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from gnn_attack_detector.graph_io import graph_to_npz_bytes, load_graph_file


class GraphIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.graph = Data(
            x=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.tensor([0, 1], dtype=torch.long),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_npz(
        self,
        name: str,
        *,
        x=None,
        edge_index=None,
        y=None,
        **extra,
    ) -> Path:
        path = self.root / name
        arrays = {
            "x": self.graph.x.numpy() if x is None else x,
            "edge_index": (
                self.graph.edge_index.numpy() if edge_index is None else edge_index
            ),
            "y": self.graph.y.numpy() if y is None else y,
            **extra,
        }
        np.savez(path, **arrays)
        return path

    def test_npz_round_trip_preserves_three_graph_tensors(self) -> None:
        first = graph_to_npz_bytes(self.graph)
        second = graph_to_npz_bytes(self.graph)
        path = self.root / "graph.npz"
        path.write_bytes(first)

        loaded = load_graph_file(path, clean_graph=self.graph)

        self.assertEqual(first, second)
        self.assertTrue(torch.equal(loaded.x, self.graph.x))
        self.assertTrue(torch.equal(loaded.edge_index, self.graph.edge_index))
        self.assertTrue(torch.equal(loaded.y, self.graph.y))

    def test_scipy_csr_npz_attaches_the_trusted_reference_features(self) -> None:
        path = self.root / "deeprobust-adjacency.npz"
        adjacency = sp.csr_matrix(
            np.array([[0, 1], [1, 0]], dtype=np.float32)
        )
        sp.save_npz(path, adjacency)

        loaded = load_graph_file(path, clean_graph=self.graph)

        self.assertTrue(torch.equal(loaded.x, self.graph.x))
        self.assertTrue(torch.equal(loaded.y, self.graph.y))
        self.assertTrue(torch.equal(loaded.edge_index, self.graph.edge_index))

    def test_scipy_csr_npz_must_match_the_trusted_node_count(self) -> None:
        path = self.root / "wrong-size.npz"
        sp.save_npz(path, sp.eye(3, dtype=np.float32, format="csr"))

        with self.assertRaisesRegex(ValueError, "trusted graph has 2"):
            load_graph_file(path, clean_graph=self.graph)

    def test_pt_dictionary_remains_supported(self) -> None:
        path = self.root / "graph.pt"
        torch.save(
            {
                "x": self.graph.x,
                "edge_index": self.graph.edge_index,
                "y": self.graph.y,
            },
            path,
        )

        loaded = load_graph_file(path, clean_graph=self.graph)

        self.assertTrue(torch.equal(loaded.edge_index, self.graph.edge_index))

    def test_npz_accepts_appended_nodes_with_unknown_labels(self) -> None:
        x = np.vstack([self.graph.x.numpy(), np.array([[1.0, 1.0]], dtype=np.float32)])
        y = np.concatenate([self.graph.y.numpy(), np.array([-1], dtype=np.int64)])
        edge_index = np.array(
            [[0, 1, 2, 0], [1, 0, 0, 2]],
            dtype=np.int64,
        )
        path = self._write_npz("injected.npz", x=x, edge_index=edge_index, y=y)

        loaded = load_graph_file(path, clean_graph=self.graph)

        self.assertEqual(int(loaded.num_nodes), 3)
        self.assertEqual(int(loaded.y[-1]), -1)

    def test_npz_requires_exact_keys(self) -> None:
        missing = self.root / "missing.npz"
        np.savez(missing, x=self.graph.x.numpy(), y=self.graph.y.numpy())
        with self.assertRaisesRegex(ValueError, "edge_index"):
            load_graph_file(missing, clean_graph=self.graph)

        extra = self._write_npz("extra.npz", answer=np.array([1]))
        with self.assertRaisesRegex(ValueError, "answer"):
            load_graph_file(extra, clean_graph=self.graph)

    def test_npz_rejects_object_array_without_pickle(self) -> None:
        path = self._write_npz(
            "object.npz",
            x=np.array([object()], dtype=object),
        )

        with self.assertRaisesRegex(ValueError, "pickle|object"):
            load_graph_file(path, clean_graph=self.graph)

    def test_npz_requires_exact_dtypes_and_shapes(self) -> None:
        cases = {
            "x-dtype": {"x": self.graph.x.numpy().astype(np.float64)},
            "edge-dtype": {
                "edge_index": self.graph.edge_index.numpy().astype(np.int32)
            },
            "y-dtype": {"y": self.graph.y.numpy().astype(np.int32)},
            "x-shape": {"x": np.zeros((2,), dtype=np.float32)},
            "edge-shape": {"edge_index": np.zeros((3, 2), dtype=np.int64)},
            "y-shape": {"y": np.zeros((2, 1), dtype=np.int64)},
        }
        for name, values in cases.items():
            with self.subTest(name=name):
                path = self._write_npz(f"{name}.npz", **values)
                with self.assertRaises(ValueError):
                    load_graph_file(path, clean_graph=self.graph)

    def test_npz_rejects_invalid_graph_values(self) -> None:
        cases = {
            "nan": {
                "x": np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32)
            },
            "self-loop": {"edge_index": np.array([[0], [0]], dtype=np.int64)},
            "asymmetric": {"edge_index": np.array([[0], [1]], dtype=np.int64)},
            "duplicate": {
                "edge_index": np.array(
                    [[0, 1, 0], [1, 0, 1]], dtype=np.int64
                )
            },
            "outside": {
                "edge_index": np.array([[0, 2], [2, 0]], dtype=np.int64)
            },
        }
        for name, values in cases.items():
            with self.subTest(name=name):
                path = self._write_npz(f"{name}.npz", **values)
                with self.assertRaises(ValueError):
                    load_graph_file(path, clean_graph=self.graph)

    def test_unknown_extension_is_rejected(self) -> None:
        path = self.root / "graph.csv"
        path.write_text("graph", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, r"\.pt or \.npz"):
            load_graph_file(path, clean_graph=self.graph)


if __name__ == "__main__":
    unittest.main()
