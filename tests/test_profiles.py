import unittest

from gnn_attack_detector.profiles import DatasetProfile, load_profiles


class DatasetProfileTests(unittest.TestCase):
    def test_loads_multiple_profiles_without_changing_detector_identity(self) -> None:
        profiles = load_profiles(
            {
                "cora-lcc": {
                    "name": "Cora",
                    "source": "deeprobust",
                    "setting": "nettack",
                    "expected_num_nodes": 2485,
                    "clean_reference": "artifacts/profiles/cora-lcc/clean.pt",
                    "probe_dir": "artifacts/profiles/cora-lcc/probes",
                },
                "cora-full": {
                    "name": "Cora",
                    "source": "planetoid",
                    "setting": "public",
                    "expected_num_nodes": 2708,
                    "clean_reference": "artifacts/profiles/cora-full/clean.pt",
                    "probe_dir": "artifacts/profiles/cora-full/probes",
                },
            }
        )
        self.assertEqual(tuple(profiles), ("cora-lcc", "cora-full"))
        self.assertNotEqual(profiles["cora-lcc"], profiles["cora-full"])
        self.assertEqual(profiles["cora-lcc"].expected_num_nodes, 2485)

    def test_rejects_absolute_artifact_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be relative"):
            DatasetProfile.from_mapping(
                "cora-lcc",
                {
                    "name": "Cora",
                    "source": "deeprobust",
                    "setting": "nettack",
                    "expected_num_nodes": 2485,
                    "clean_reference": "E:/private/clean.pt",
                    "probe_dir": "artifacts/probes",
                },
            )

    def test_rejects_invalid_profile_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile id"):
            DatasetProfile.from_mapping(
                "Cora LCC",
                {
                    "name": "Cora",
                    "source": "deeprobust",
                    "setting": "nettack",
                    "expected_num_nodes": 2485,
                    "clean_reference": "artifacts/clean.pt",
                    "probe_dir": "artifacts/probes",
                },
            )


if __name__ == "__main__":
    unittest.main()
