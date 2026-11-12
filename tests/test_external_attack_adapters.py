import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from gnn_attack_detector.attacks.deeprobust_adapter import _validate_modified_adjacency


ROOT = Path(__file__).resolve().parents[1]


class ExternalAttackAdapterSourceTests(unittest.TestCase):
    def test_cuda_adjacency_is_copied_to_numpy_before_scipy_conversion(self) -> None:
        values = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        class FakeCudaTensor:
            def __array__(self):
                raise TypeError("can't convert cuda tensor to numpy")

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return values

        with patch(
            "gnn_attack_detector.attacks.deeprobust_adapter.torch.is_tensor",
            return_value=True,
        ):
            matrix = _validate_modified_adjacency(
                FakeCudaTensor(), num_nodes=2
            )

        np.testing.assert_array_equal(matrix.toarray(), values)

    def test_torch_native_attack_paths_accept_selected_device(self) -> None:
        deep_source = (
            ROOT / "src/gnn_attack_detector/attacks/deeprobust_adapter.py"
        ).read_text(encoding="utf-8")
        greatx_source = (
            ROOT / "src/gnn_attack_detector/attacks/greatx_adapter.py"
        ).read_text(encoding="utf-8")

        for function_name in (
            "_run_gradient_targeted",
            "run_deeprobust_sgattack",
            "_run_meta_attack",
            "_run_pgd_family",
            "run_deeprobust_prbcd",
        ):
            function = next(
                node for node in ast.parse(deep_source).body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            self.assertIn("device", [arg.arg for arg in function.args.kwonlyargs])
            self.assertIn("device=device", ast.unparse(function))

        fgattack = next(
            node for node in ast.parse(greatx_source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_greatx_fgattack_untargeted"
        )
        self.assertIn("device", [arg.arg for arg in fgattack.args.kwonlyargs])
        self.assertIn("device=device", ast.unparse(fgattack))

    def test_deeprobust_seed_context_includes_torch_rng(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/deeprobust_adapter.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_isolated_random_seed"
        )
        function_source = ast.unparse(function)

        self.assertIn("torch.random.get_rng_state()", function_source)
        self.assertIn("torch.manual_seed(int(seed))", function_source)
        self.assertIn("torch.random.set_rng_state(torch_state)", function_source)

    def test_greatx_imports_are_lazy_and_errors_are_actionable(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/greatx_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _load_greatx_attack", source)
        self.assertIn("GreatX is required", source)
        self.assertIn("def run_greatx_gfattack", source)
        self.assertIn("def run_greatx_fgattack_untargeted", source)
        self.assertNotIn("from greatx", source.split("def _load_greatx_attack", 1)[0])

    def test_structack_is_low_degree_low_similarity_not_random_alias(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/structack_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def run_structack", source)
        self.assertIn("jaccard", source.lower())
        self.assertIn("degrees", source)
        self.assertNotIn("run_deeprobust_random", source)

    def test_greatx_seeds_all_random_sources_for_distinct_reproducible_runs(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/greatx_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _isolated_random_seed", source)
        self.assertIn("np.random.seed(int(seed))", source)
        self.assertIn("random.seed(int(seed))", source)
        self.assertIn("torch.manual_seed(int(seed))", source)
        self.assertIn("with _isolated_random_seed(seed):", source)

    def test_fgattack_uses_a_seeded_victim_subset_to_avoid_duplicate_full_graph_runs(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/greatx_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _seeded_victim_nodes", source)
        self.assertIn("max_nodes", source)
        self.assertIn("torch.randperm", source)
        self.assertIn("victim_nodes = _seeded_victim_nodes", source)

    def test_greatx_adapter_leaves_budget_mismatch_to_retryable_protocol_layer(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/greatx_adapter.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_validated_graph"
        )

        self.assertEqual(
            [argument.arg for argument in function.args.args],
            ["clean", "candidate"],
        )
        self.assertNotIn("budget", ast.unparse(function))

        protocol_source = (
            ROOT / "src/gnn_attack_detector/graph_attacks.py"
        ).read_text(encoding="utf-8")
        corpus_source = (
            ROOT / "src/gnn_attack_detector/corpus.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if outcome.actual_modifications != budget:", protocol_source)
        self.assertIn("raise AttackCandidateRejected(", protocol_source)
        self.assertIn("except AttackCandidateRejected as exc:", corpus_source)

    def test_pgd_uses_an_exact_seeded_discrete_projection(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/deeprobust_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _select_pgd_flip_indices", source)
        self.assertIn("_select_pgd_flip_indices", source)
        self.assertIn("randomized discrete projection", source.lower())
        self.assertIn("optimized_scores", source)
        self.assertIn("capture_optimized_scores", source)
        self.assertNotIn("stock_random_sample", source)

    def test_pgd_projection_is_exact_reproducible_and_seed_dependent(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/deeprobust_adapter.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_select_pgd_flip_indices"
        )
        namespace = {"np": np}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "<pgd-projection>", "exec"),
            namespace,
        )
        select = namespace["_select_pgd_flip_indices"]
        scores = np.linspace(0.0, 1.0, 1000, dtype=np.float64)

        first = select(scores, 100, 11)
        repeated = select(scores, 100, 11)
        different_seed = select(scores, 100, 12)

        self.assertEqual(len(first), 100)
        self.assertEqual(len(np.unique(first)), 100)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different_seed))
        self.assertGreater(float(scores[first].mean()), float(scores.mean()))

    def test_nettack_feature_compatibility_flattens_ndarrays_without_a1(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/deeprobust_adapter.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_dense_vector"
        )

        class FakeSparse:
            @staticmethod
            def issparse(values) -> bool:
                return False

        namespace = {"Any": object, "np": np, "sp": FakeSparse}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "<dense-vector>", "exec"),
            namespace,
        )
        values = namespace["_dense_vector"](np.array([[3.0, 1.0]]))

        np.testing.assert_array_equal(values, np.array([3.0, 1.0]))
        self.assertIn("attacker.feature_scores =", source)
        self.assertIn("_nettack_feature_scores_compat", source)

    def test_feature_only_nettack_has_a_target_and_empty_constraint_fallback(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/attacks/deeprobust_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn('if mode == "feature":', source)
        self.assertIn(
            "attacker.influencer_nodes = np.array([int(target_node)]",
            source,
        )
        self.assertIn(
            "attacker.potential_edges = np.empty((0, 2), dtype=np.int32)",
            source,
        )
        self.assertIn("Nettack co-occurrence constraint has no candidate", source)
        self.assertIn("unconstrained", source)


if __name__ == "__main__":
    unittest.main()
