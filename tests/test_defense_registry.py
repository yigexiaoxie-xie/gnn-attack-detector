import unittest

from gnn_attack_detector.defenses.registry import (
    CHECKPOINT_DEFENSES,
    DEFENSE_SPECS,
    INPUT_FIT_DEFENSES,
    OFFLINE_REFERENCE_DEFENSES,
    ONLINE_PROBE_DEFENSES,
    SUPPORTED_DEFENSES,
    defense_spec,
)


EXPECTED_DEFENSES = (
    "gcn",
    "gcn_jaccard",
    "gcn_svd",
    "rgcn",
    "median_gcn",
    "simpgcn",
    "prognn",
    "gnnguard",
    "airgnn",
    "sgc",
)


class DefenseRegistryTests(unittest.TestCase):
    def test_registry_contains_ten_unique_deeprobust_models(self) -> None:
        self.assertEqual(SUPPORTED_DEFENSES, EXPECTED_DEFENSES)
        self.assertEqual(len(DEFENSE_SPECS), 10)
        self.assertEqual({item.backend for item in DEFENSE_SPECS}, {"deeprobust"})

    def test_detection_uses_only_three_online_probes(self) -> None:
        self.assertEqual(ONLINE_PROBE_DEFENSES, ("gcn", "gcn_jaccard", "simpgcn"))
        self.assertEqual(OFFLINE_REFERENCE_DEFENSES, ("prognn",))

    def test_catalog_keeps_expensive_models_out_of_the_online_path(self) -> None:
        self.assertEqual(defense_spec("prognn").role, "offline_reference")
        self.assertEqual(defense_spec("gnnguard").role, "benchmark")
        self.assertFalse(defense_spec("prognn").online)

    def test_registry_declares_honest_execution_modes(self) -> None:
        self.assertEqual(CHECKPOINT_DEFENSES, ())
        self.assertEqual(
            INPUT_FIT_DEFENSES,
            tuple(name for name in EXPECTED_DEFENSES if name != "prognn"),
        )
        self.assertEqual(OFFLINE_REFERENCE_DEFENSES, ("prognn",))
        self.assertEqual(defense_spec("gcn").execution_mode, "input_fit")
        self.assertEqual(defense_spec("simpgcn").execution_mode, "input_fit")
        self.assertEqual(defense_spec("gcn_svd").execution_mode, "input_fit")
        self.assertEqual(defense_spec("prognn").execution_mode, "offline")


if __name__ == "__main__":
    unittest.main()
