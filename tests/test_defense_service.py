from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from gnn_attack_detector.defense_service import (
    _load_evaluation_context,
    _load_defense_class,
    _per_input_predictions,
    _SingleGraphDataset,
    defense_profile_status,
    evaluate_robust_probe,
)


def graph() -> Data:
    return Data(
        x=torch.ones(4, 2),
        y=torch.tensor([0, 1, 0, 1]),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        train_ids=torch.tensor([0]),
        val_ids=torch.tensor([1]),
        test_ids=torch.tensor([2, 3]),
        num_nodes=4,
    )


class DefenseServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = SimpleNamespace(
            identifier="cora-lcc",
            expected_num_nodes=4,
            clean_reference="artifacts/profiles/cora-lcc/clean.pt",
            probe_dir="artifacts/profiles/cora-lcc/probes",
        )

    def test_pyg_defenses_use_their_actual_deeprobust_modules(self) -> None:
        fake_class = type("SGC", (), {})
        with patch(
            "gnn_attack_detector.defense_service.import_module",
            return_value=SimpleNamespace(SGC=fake_class),
        ) as importer:
            self.assertIs(_load_defense_class("sgc"), fake_class)
        importer.assert_called_once_with("deeprobust.graph.defense_pyg.sgc")

    def test_sgc_and_median_receive_a_one_graph_dataset(self) -> None:
        data = graph()
        dataset = _SingleGraphDataset(data)
        self.assertIs(dataset[0], data)
        self.assertEqual(len(dataset), 1)

    def test_profile_status_allows_per_input_models_without_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = defense_profile_status(root, self.profile, clean_ready=True)

        self.assertTrue(status["gcn"]["ready"])
        self.assertTrue(status["rgcn"]["ready"])
        self.assertTrue(status["sgc"]["ready"])
        self.assertFalse(status["gcn"]["prepared"])
        self.assertEqual(status["gcn"]["execution_mode"], "input_fit")
        self.assertTrue(status["gcn_svd"]["ready"])
        self.assertEqual(status["gcn_svd"]["execution_mode"], "input_fit")
        self.assertEqual(status["prognn"]["execution_mode"], "offline")

    def test_context_uses_trusted_deeprobust_labels_and_split(self) -> None:
        clean = graph()
        uploaded = graph()
        uploaded.y = 1 - clean.y
        config = {"profiles": {"cora-lcc": {}}}
        with tempfile.TemporaryDirectory() as directory, patch(
            "gnn_attack_detector.defense_service.load_config", return_value=config,
        ), patch(
            "gnn_attack_detector.defense_service.load_profiles",
            return_value={"cora-lcc": self.profile},
        ), patch(
            "gnn_attack_detector.defense_service.torch.load", return_value=clean,
        ), patch(
            "gnn_attack_detector.defense_service.load_graph_file", return_value=uploaded,
        ):
            context = _load_evaluation_context(
                "unused.yaml", "input.pt", directory, "cora-lcc",
            )

        self.assertTrue(torch.equal(context["graph"].y, clean.y))
        self.assertEqual(
            context["split"], {"train": [0], "val": [1], "test": [2, 3]},
        )

    def test_each_input_retrains_clean_and_attacked_graph_with_same_split(self) -> None:
        clean = graph()
        attacked = graph()
        split = {"train": [0], "val": [1], "test": [2, 3]}
        context = {
            "config": {"runtime": {"device": "cpu"}, "seed": 42},
            "profile": self.profile,
            "clean": clean,
            "graph": attacked,
            "split": split,
        }
        outputs = [torch.full((4, 2), .5), torch.full((4, 2), .5)]

        def run(operation, **kwargs):
            return operation(torch.device("cpu"))

        with patch(
            "gnn_attack_detector.defense_service.run_with_cpu_fallback",
            side_effect=run,
        ), patch(
            "gnn_attack_detector.defense_service._fit_defense",
            side_effect=outputs,
        ) as fit:
            _per_input_predictions(context, "gcn")

        self.assertEqual(fit.call_count, 2)
        self.assertIs(fit.call_args_list[0].args[1], clean)
        self.assertIs(fit.call_args_list[1].args[1], attacked)
        self.assertEqual(fit.call_args_list[0].args[2], split)
        self.assertEqual(fit.call_args_list[1].args[2], split)

    def test_evaluation_returns_common_metrics_and_execution_mode(self) -> None:
        clean = graph()
        attacked = graph()
        clean_probabilities = torch.tensor([[.9, .1], [.1, .9], [.8, .2], [.2, .8]])
        attacked_probabilities = torch.tensor([[.9, .1], [.1, .9], [.3, .7], [.2, .8]])
        context = {
            "config": {"runtime": {"device": "cpu"}},
            "profile": self.profile,
            "clean": clean,
            "graph": attacked,
            "split": {"train": [0], "val": [1], "test": [2, 3]},
        }
        with patch(
            "gnn_attack_detector.defense_service._load_evaluation_context",
            return_value=context,
        ), patch(
            "gnn_attack_detector.defense_service._per_input_predictions",
            return_value=(clean_probabilities, attacked_probabilities, "cpu"),
        ):
            result = evaluate_robust_probe(
                "unused.yaml", "input.pt", project_root=".",
                profile_id="cora-lcc", defense_name="rgcn",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["execution_mode"], "input_fit")
        self.assertEqual(
            result["split_sizes"], {"train": 1, "val": 1, "test": 2},
        )
        self.assertEqual(result["evaluated_nodes"], 2)
        self.assertEqual(result["clean_accuracy"], 1.0)
        self.assertEqual(result["input_graph_accuracy"], 0.5)
        self.assertEqual(result["accuracy_drop"], 0.5)

if __name__ == "__main__":
    unittest.main()
