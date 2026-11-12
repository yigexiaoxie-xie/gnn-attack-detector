import unittest

from gnn_attack_detector.attacks.registry import (
    ATTACK_SPECS,
    GLOBAL_ATTACKS,
    LIVE_GENERATION_ATTACKS,
    SUPPORTED_ATTACKS,
    TARGETED_ATTACKS,
    attack_spec,
)


def test_cuda_capability_is_explicit_for_expensive_torch_attacks() -> None:
    capable = {spec.identifier for spec in ATTACK_SPECS if spec.cuda_capable}

    assert {"prbcd", "fgattack_untargeted", "fga", "sgattack", "pgdattack", "metattack"} <= capable
    assert {"gfattack", "nettack_structure", "structack"}.isdisjoint(capable)


EXPECTED_ATTACKS = (
    "random_add",
    "random_delete",
    "random_flip",
    "dice_add",
    "dice_delete",
    "dice_mix",
    "degree_add",
    "degree_delete",
    "prbcd",
    "structack",
    "gfattack",
    "fgattack_untargeted",
    "rnd",
    "nettack_structure",
    "nettack_feature",
    "nettack_hybrid",
    "fga",
    "sgattack",
    "pgdattack",
    "metattack",
    "igattack",
    "bridge_add",
    "hub_rewire",
)


class AttackRegistryTests(unittest.TestCase):
    def test_registry_contains_the_exact_twenty_three_protocols(self) -> None:
        self.assertEqual(SUPPORTED_ATTACKS, EXPECTED_ATTACKS)
        self.assertEqual(len(ATTACK_SPECS), 23)
        self.assertEqual(len({item.identifier for item in ATTACK_SPECS}), 23)
        self.assertIn("igattack", SUPPORTED_ATTACKS)
        self.assertNotIn("node_embedding_add", SUPPORTED_ATTACKS)

    def test_protocols_record_their_real_backend(self) -> None:
        self.assertEqual(attack_spec("prbcd").backend, "deeprobust")
        self.assertEqual(attack_spec("gfattack").backend, "greatx")
        self.assertEqual(attack_spec("fgattack_untargeted").backend, "greatx")
        self.assertEqual(attack_spec("structack").backend, "structack")
        self.assertEqual(attack_spec("bridge_add").backend, "fast_structural")
        self.assertEqual(attack_spec("hub_rewire").backend, "fast_structural")

    def test_protocol_variants_map_to_their_real_algorithm_and_operation(self) -> None:
        expected = {
            "random_add": ("random", "add"),
            "random_delete": ("random", "remove"),
            "random_flip": ("random", "flip"),
            "dice_add": ("dice", "add"),
            "dice_delete": ("dice", "remove"),
            "dice_mix": ("dice", "mix"),
            "degree_add": ("degree", "add"),
            "degree_delete": ("degree", "remove"),
            "prbcd": ("prbcd", "flip"),
            "structack": ("structack", "add"),
            "gfattack": ("gfattack", "structure"),
            "fgattack_untargeted": ("fgattack", "flip"),
            "nettack_structure": ("nettack", "structure"),
            "nettack_feature": ("nettack", "feature"),
            "nettack_hybrid": ("nettack", "hybrid"),
            "bridge_add": ("bridge_add", "add"),
            "hub_rewire": ("hub_rewire", "flip"),
        }
        for identifier, pair in expected.items():
            spec = attack_spec(identifier)
            self.assertEqual((spec.algorithm, spec.operation), pair)

    def test_families_partition_every_protocol(self) -> None:
        self.assertFalse(set(GLOBAL_ATTACKS) & set(TARGETED_ATTACKS))
        self.assertEqual(
            set(GLOBAL_ATTACKS) | set(TARGETED_ATTACKS), set(EXPECTED_ATTACKS)
        )
        for name in GLOBAL_ATTACKS:
            self.assertEqual(attack_spec(name).strength_unit, "edge_rate")
        for name in TARGETED_ATTACKS:
            self.assertEqual(attack_spec(name).strength_unit, "target_nodes")

    def test_metattack_reuses_precomputed_graphs_then_generates_missing_graphs(self) -> None:
        self.assertEqual(
            attack_spec("metattack").generation_mode,
            "precomputed_or_live",
        )
        self.assertIn("metattack", LIVE_GENERATION_ATTACKS)
        self.assertEqual(len(LIVE_GENERATION_ATTACKS), 23)


if __name__ == "__main__":
    unittest.main()
