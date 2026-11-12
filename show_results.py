"""Run this file directly in PyCharm to view the measured detector accuracy."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gnn_attack_detector.result_viewer import (  # noqa: E402
    format_evaluation_report,
    load_evaluation_report,
)


REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "reports" / "universal-detector-evaluation.json"
)


def main() -> int:
    try:
        report = load_evaluation_report(REPORT_PATH)
        print(format_evaluation_report(report, report_path=REPORT_PATH))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"无法显示评测结果：{exc}")
        print("请先运行 scripts\\run_selected_attacks.bat 完成训练和测试。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
