from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MultiProfileCorpusSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            ROOT / "src/gnn_attack_detector/corpus.py"
        ).read_text(encoding="utf-8")

    def test_subset_runs_use_canonical_attack_seed_indexes(self) -> None:
        self.assertIn("SUPPORTED_ATTACKS.index(attack_name)", self.source)
        self.assertNotIn("for attack_index, attack_name in enumerate(attacks)", self.source)

    def test_new_artifacts_are_namespaced_by_profile(self) -> None:
        self.assertIn(
            'Path(runtime["output_dir"]) / profile_id / partition / attack_name',
            self.source,
        )

    def test_legacy_artifacts_require_an_explicit_matching_profile(self) -> None:
        self.assertNotIn('item.get("dataset_id") in {None, profile_id}', self.source)
        self.assertIn('item.get("dataset_id") == profile_id', self.source)

    def test_existing_record_path_can_be_reused_without_copying(self) -> None:
        self.assertIn('previous.get("graph_path")', self.source)
        self.assertIn("expected_backend", self.source)

    def test_precomputed_records_can_be_reused_for_the_uniform_quota(self) -> None:
        self.assertIn('spec.generation_mode == "precomputed_or_live"', self.source)
        self.assertIn("_precomputed_meta_candidates", self.source)
        self.assertIn("used_precomputed_paths", self.source)
        self.assertIn('"precomputed_source_path"', self.source)
        self.assertIn('"independent"', self.source)
        self.assertIn("REUSED-PRECOMPUTED", self.source)

    def test_metattack_falls_back_to_live_generation(self) -> None:
        dispatch = (
            ROOT / "src/gnn_attack_detector/graph_attacks.py"
        ).read_text(encoding="utf-8")

        self.assertIn("run_deeprobust_metattack", dispatch)
        self.assertIn('if spec.algorithm == "metattack"', dispatch)
        self.assertIn("if precomputed_graph_path is not None", dispatch)
        self.assertNotIn(
            "Metattack requires a verified precomputed adjacency",
            dispatch,
        )

    def test_corpus_does_not_fallback_to_duplicate_graphs(self) -> None:
        self.assertNotIn("duplicate_outcome", self.source)
        self.assertNotIn("GENERATED-DUPLICATE", self.source)
        self.assertIn("duplicate_graph", self.source)
        self.assertIn("attack_contract", self.source)

    def test_invalid_precomputed_metattack_file_is_skipped_before_live_fallback(self) -> None:
        self.assertIn("precomputed_error", self.source)
        self.assertIn("rejected_precomputed.add", self.source)

    def test_duplicate_hashes_are_rejected_across_data_partitions(self) -> None:
        seen_block = self.source.split("seen_hashes = {", 1)[1].split("}", 1)[0]
        self.assertIn('item.get("attack_name") == attack_name', seen_block)
        self.assertNotIn('item.get("partition") == partition', seen_block)

    def test_manifest_is_persisted_after_each_slot(self) -> None:
        self.assertGreaterEqual(self.source.count("_write_jsonl(manifest_path, records)"), 4)

    def test_verification_filters_selected_attacks(self) -> None:
        self.assertIn("and str(item.get(\"attack_name\")) in selected_attacks", self.source)


if __name__ == "__main__":
    unittest.main()
