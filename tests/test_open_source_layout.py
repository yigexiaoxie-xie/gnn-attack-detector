from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OpenSourceLayoutTests(unittest.TestCase):
    def test_readme_documents_the_only_normal_selected_run(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for text in (
            "configs/run.yaml",
            "run_selected_attacks.bat",
            "cora-lcc",
            "citeseer-lcc",
            "polblogs-lcc",
            "pubmed-lcc",
            "取消注释",
            "数据集数 × 攻击方法数",
            "自动复用",
            "知识智能与安全计算研究组（吴涛团队）",
        ):
            self.assertIn(text, readme)
        self.assertTrue(readme.rstrip().endswith("知识智能与安全计算研究组（吴涛团队）"))

    def test_third_party_notices_record_all_attack_provenance(self) -> None:
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for name in ("DeepRobust", "GreatX", "Structack", "MIT"):
            self.assertIn(name, notices)

    def test_only_current_windows_entry_points_remain(self) -> None:
        scripts = {path.name for path in (ROOT / "scripts").iterdir() if path.is_file()}
        self.assertIn("setup_windows_cpu.bat", scripts)
        self.assertIn("run_selected_attacks.bat", scripts)
        self.assertIn("train_detector.bat", scripts)
        self.assertIn("start_platform.bat", scripts)
        self.assertFalse(any("day" in name.lower() for name in scripts))

    def test_obsolete_backends_and_stage_modules_are_absent(self) -> None:
        package = ROOT / "src" / "gnn_attack_detector"
        self.assertFalse((package / "day4_corpus.py").exists())
        self.assertFalse((package / "day4_pipeline.py").exists())
        self.assertTrue((package / "attacks" / "greatx_adapter.py").exists())
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertTrue((ROOT / "THIRD_PARTY_NOTICES.md").is_file())

    def test_powershell_files_are_windows_powershell_safe_ascii(self) -> None:
        for path in (ROOT / "scripts").glob("*.ps1"):
            path.read_bytes().decode("ascii")

    def test_selected_runner_is_portable_and_uses_the_project_venv(self) -> None:
        batch = (ROOT / "scripts" / "run_selected_attacks.bat").read_text(
            encoding="ascii"
        )
        powershell = (ROOT / "scripts" / "run_selected_attacks.ps1").read_text(
            encoding="ascii"
        )
        self.assertIn("run_selected_attacks.ps1", batch)
        self.assertIn(".venv\\Scripts\\python.exe", powershell)
        self.assertIn("show-selection", powershell)
        self.assertIn("run-selected", powershell)
        self.assertNotIn("E:\\", batch + powershell)
        self.assertNotIn("xqw", (batch + powershell).lower())


if __name__ == "__main__":
    unittest.main()
