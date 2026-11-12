"""Human-readable terminal view of the measured detector evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _correct_and_total(metrics: Mapping[str, Any]) -> tuple[int, int]:
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("test_metrics.confusion_matrix is missing")
    correct = int(matrix.get("tp", 0)) + int(matrix.get("tn", 0))
    return correct, int(metrics.get("count", 0))


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    """Load the real JSON report written by ``evaluate_detector``."""

    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"评测报告不存在：{report_path}")
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"评测报告不是有效 JSON：{report_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("评测报告必须是 JSON 对象")
    return value


def format_evaluation_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path | None = None,
) -> str:
    """Format only values measured from held-out test predictions."""

    metrics = report.get("test_metrics")
    recall = report.get("attacked_complete_graph_recall")
    acceptance = report.get("acceptance")
    if not isinstance(metrics, Mapping):
        raise ValueError("评测报告缺少 test_metrics，不能显示真实准确率")
    if not isinstance(recall, Mapping) or not isinstance(recall.get("by_attack"), Mapping):
        raise ValueError("评测报告缺少 attacked_complete_graph_recall")
    if not isinstance(acceptance, Mapping):
        raise ValueError("评测报告缺少 acceptance")

    correct, total = _correct_and_total(metrics)
    lines = [
        "=" * 72,
        "完整图攻击检测——实际测试结果",
        "=" * 72,
    ]
    if report_path is not None:
        lines.append(f"报告文件: {Path(report_path)}")
    if report.get("detector_model_hash"):
        lines.append(f"本报告模型: {report['detector_model_hash']}（不代表当前激活模型）")
    if report.get("clean_control_warning"):
        lines.append("注意：重复干净对照或未经独立性确认的样本不能证明正式准确率；请核对独立非攻击测试集。")
    lines.extend(
        [
            f"参与评测的数据集: {', '.join(str(value) for value in report.get('profiles', [])) or '未记录'}",
            f"综合准确率: {_percent(metrics['accuracy'])} ({correct}/{total})",
            f"攻击图检出率: {_percent(metrics['recall'])}",
            f"干净图识别率: {_percent(metrics['specificity'])}",
            f"精确率: {_percent(metrics['precision'])}",
            f"F1: {_percent(metrics['f1'])}",
            "",
            "各数据集实际结果:",
        ]
    )
    profile_metrics = report.get("metrics_by_profile", {})
    if isinstance(profile_metrics, Mapping) and profile_metrics:
        for profile_id, values in profile_metrics.items():
            if not isinstance(values, Mapping):
                continue
            profile_correct, profile_total = _correct_and_total(values)
            lines.append(
                f"  {profile_id}: 准确率 {_percent(values['accuracy'])} "
                f"({profile_correct}/{profile_total})，攻击图检出率 {_percent(values['recall'])}，"
                f"干净图识别率 {_percent(values['specificity'])}"
            )
    else:
        lines.append("  当前报告未包含分数据集结果；重新运行评测即可生成。")

    lines.extend(["", "各攻击方法实际检出率:"])
    for attack_name, values in recall["by_attack"].items():
        if not isinstance(values, Mapping):
            continue
        lines.append(
            f"  {attack_name}: {_percent(values['recall'])} "
            f"({int(values['detected'])}/{int(values['total'])})"
        )

    minimum = float(acceptance["minimum"])
    lines.extend(
        [
            "",
            f"配置的验收阈值: {_percent(minimum)}",
            f"最终验收状态: {'通过' if bool(acceptance.get('passed')) else '未通过'}",
            "说明：上面的准确率和检出率均来自已保存测试图的实际预测，不是配置值。",
        ]
    )
    return "\n".join(lines)
