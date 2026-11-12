import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CurrentModuleLayoutTests(unittest.TestCase):
    def test_windows_cuda_setup_keeps_cuda_and_cpu_installers_separate(self) -> None:
        cuda_script = ROOT / "scripts/setup_windows_cuda.ps1"
        cpu_script = ROOT / "scripts/setup_windows_cpu.ps1"

        self.assertTrue(cuda_script.is_file())
        cuda_source = cuda_script.read_text(encoding="utf-8")
        cpu_source = cpu_script.read_text(encoding="utf-8")
        self.assertIn("https://download.pytorch.org/whl/cu121", cuda_source)
        self.assertIn("torch-2.2.0+cu121.html", cuda_source)
        self.assertIn("torch.cuda.is_available()", cuda_source)
        self.assertNotIn("cu121", cpu_source)

    def test_runtime_has_no_day_numbered_python_modules(self) -> None:
        package = ROOT / "src/gnn_attack_detector"
        self.assertTrue((package / "corpus.py").is_file())
        self.assertTrue((package / "pipeline.py").is_file())
        self.assertFalse((package / "day4_corpus.py").exists())
        self.assertFalse((package / "day4_pipeline.py").exists())


if __name__ == "__main__":
    unittest.main()
