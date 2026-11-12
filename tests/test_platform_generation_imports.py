from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PlatformGenerationImportTests(unittest.TestCase):
    def test_attack_registry_symbols_are_imported_from_registry(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/platform_generation.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "from gnn_attack_detector.attacks.registry import (",
            source,
        )
        self.assertIn("LIVE_GENERATION_ATTACKS", source)
        self.assertIn("SUPPORTED_ATTACKS", source)
        self.assertIn("TARGETED_ATTACKS", source)

    def test_platform_service_accepts_current_evaluation_report_version(self) -> None:
        source = (
            ROOT / "src/gnn_attack_detector/platform_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("report.get(\"format_version\") not in {2, 3}", source)


if __name__ == "__main__":
    unittest.main()
