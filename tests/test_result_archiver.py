from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell 7 is required")
def test_archiver_saves_report_dataset_metrics_and_average(tmp_path: Path) -> None:
    report = {
        "evaluated_at_utc": "2026-09-17T00:00:00+00:00",
        "profiles": ["cora-lcc", "citeseer-lcc"],
        "selected_attacks": ["random_add"],
        "metrics_by_profile": {
            "cora-lcc": {
                "accuracy": 0.75,
                "balanced_accuracy": 0.8,
                "recall": 1.0,
                "specificity": 0.5,
                "precision": 0.67,
                "f1": 0.8,
                "count": 4,
                "confusion_matrix": {"tp": 2, "tn": 1, "fp": 1, "fn": 0},
            },
            "citeseer-lcc": {
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "recall": 0.5,
                "specificity": 0.5,
                "precision": 0.5,
                "f1": 0.5,
                "count": 6,
                "confusion_matrix": {"tp": 2, "tn": 1, "fp": 2, "fn": 1},
            },
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "archive_detector_result.ps1"

    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(script),
            "-ProjectRoot",
            str(tmp_path),
            "-ReportPath",
            "report.json",
            "-ArchiveRoot",
            "history",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    archive_dirs = list((tmp_path / "history").iterdir())
    assert len(archive_dirs) == 1
    archive = archive_dirs[0]
    assert json.loads((archive / "evaluation.json").read_text(encoding="utf-8")) == report

    summary = json.loads((archive / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["average_accuracy"] == pytest.approx(0.625)
    assert summary["weighted_accuracy"] == pytest.approx(0.6)
    assert summary["datasets"][0]["correct"] == 3
    assert summary["datasets"][1]["correct"] == 3
    assert "cora-lcc" in (archive / "dataset_summary.csv").read_text(encoding="utf-8")
    assert "Average accuracy: 62.50%" in (archive / "summary.txt").read_text(encoding="utf-8")


def test_train_script_invokes_result_archiver() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "train_detector.ps1").read_text(encoding="utf-8")
    assert "archive_detector_result.ps1" in script
