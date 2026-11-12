# Cumulative Universal Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make full and incremental dataset runs train one detector from every saved profile and publish separate current-model evaluation results for every saved dataset.

**Architecture:** Keep the existing per-profile graph and feature artifacts plus the single global detector file. Change only the selected-run orchestration boundary: selection controls expensive preparation, while training and evaluation use the pipeline's existing “all saved profiles” default. Add a model hash and UTC timestamp to the global report so the platform can distinguish current results from a retained historical report.

**Tech Stack:** Python 3.12, PyTorch, Streamlit, unittest/pytest, JSON/JSONL artifacts.

---

## File map

- Modify `src/gnn_attack_detector/workflow.py`: stop restricting universal training and evaluation to `selection.profiles`.
- Modify `src/gnn_attack_detector/pipeline.py`: write model identity metadata and replace prediction JSONL atomically.
- Modify `src/gnn_attack_detector/platform_service.py`: compare the saved report hash with the current detector hash.
- Modify `src/gnn_attack_detector/web_app.py`: show evaluation time/model hash and a historical-result warning.
- Modify `tests/test_selected_workflow.py`: prove selection scopes generation but not universal training/evaluation.
- Modify `tests/test_detector_core.py`: prove reports identify the detector and prediction writes preserve the previous file on replacement failure.
- Modify `tests/test_platform_workspace.py`: prove current/historical classification and UI metadata rendering.
- Modify `README.md`: document one-model cumulative behavior for full and incremental runs.

The worktree already contains unrelated uncommitted edits in several of these files. Preserve them, use `apply_patch` for narrow changes, and do not stage or commit overlapping files as a whole.

### Task 1: Make selected runs train and evaluate every saved profile

**Files:**
- Modify: `tests/test_selected_workflow.py`
- Modify: `src/gnn_attack_detector/workflow.py:145-150,189-194`

- [ ] **Step 1: Write the failing orchestration test**

In `test_two_profiles_run_in_order_with_one_universal_training`, replace `universal_profile_scopes` with `universal_call_kwargs` and record complete keyword arguments:

```python
        universal_call_kwargs: list[dict[str, object]] = []

        def train(config, **kwargs):
            events.append("train:universal")
            seen_attacks.append(tuple(config["attacks"]["protocols"]))
            universal_call_kwargs.append(dict(kwargs))
            return {"status": "ok", "training_rows": 10}

        def evaluate(config, **kwargs):
            events.append("evaluate:universal")
            universal_call_kwargs.append(dict(kwargs))
            return {"status": "ok", "acceptance": {"passed": False}}
```

Replace the old profile-scope assertion with:

```python
        self.assertEqual(len(universal_call_kwargs), 2)
        self.assertTrue(
            all("profile_ids" not in kwargs for kwargs in universal_call_kwargs)
        )
        self.assertTrue(
            all(kwargs["project_root"] == ROOT.resolve() for kwargs in universal_call_kwargs)
        )
```

This expresses the contract: profile selection still drives both profile preparation loops, but the universal operations discover all saved profile feature files through their existing default behavior.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_selected_workflow.py::SelectedWorkflowTests::test_two_profiles_run_in_order_with_one_universal_training -q
```

Expected: FAIL because both calls currently include `profile_ids=selection.profiles`.

- [ ] **Step 3: Remove only the two narrowing arguments**

Change the training call in `run_selected_workflow` to:

```python
            results["detector_training"] = active_operations.train_universal(
                selected_configs[selection.profiles[0]],
                project_root=root,
            )
```

Change the evaluation call to:

```python
            results["evaluation"] = active_operations.evaluate_universal(
                selected_configs[selection.profiles[0]],
                project_root=root,
            )
```

Do not change the profile preparation or test-generation loops. `train_detector` and `evaluate_detector` already interpret omitted `profile_ids` as all valid files under `artifacts/features/profiles/`.

- [ ] **Step 4: Run the workflow tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_selected_workflow.py tests/test_detector_core.py::DetectorCoreTests::test_available_profile_datasets_only_loads_explicit_selection -q
```

Expected: PASS. The explicit-selection pipeline API remains available, while selected workflow calls now use its cumulative default.

- [ ] **Step 5: Inspect the narrow diff checkpoint**

Run:

```powershell
git diff -- src/gnn_attack_detector/workflow.py tests/test_selected_workflow.py
```

Expected: the new changes only remove the two `profile_ids` arguments and update the matching test. Leave these overlapping dirty files unstaged so pre-existing user changes are not accidentally committed.

### Task 2: Bind each evaluation report to the current detector

**Files:**
- Modify: `tests/test_detector_core.py`
- Modify: `src/gnn_attack_detector/pipeline.py:3-8,407-412,491-502`

- [ ] **Step 1: Write a failing report-metadata test**

Add the standard-library imports below and replace the existing pipeline import in `tests/test_detector_core.py` with the shown grouped import:

```python
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from gnn_attack_detector.integrity import hash_file
from gnn_attack_detector.pipeline import (
    _available_profile_datasets,
    _build_profile_dataset,
    evaluate_detector,
)
```

Add this test to `DetectorCoreTests`:

```python
    def test_evaluation_report_identifies_current_detector(self) -> None:
        config = load_config(ROOT / "configs" / "base.yaml")
        dataset = GraphDetectorTensorDataset(
            feature_names=("signal",),
            x=torch.tensor([[1.0], [0.0]]),
            y=torch.tensor([1, 0]),
            partitions=("test", "test"),
            graph_ids=("attack", "clean-reference-test"),
            dataset_ids=("cora-lcc", "cora-lcc"),
            attack_names=("random_add", "clean"),
            strength_values=(0.05, None),
            strength_units=("edge_rate", "none"),
            independent_clean=torch.tensor([False, True]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / config["detector"]["model_path"]
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"current-detector")
            bundle = SimpleNamespace(feature_names=("signal",), threshold=0.5)
            with (
                patch(
                    "gnn_attack_detector.pipeline._available_profile_datasets",
                    return_value=[dataset],
                ),
                patch(
                    "gnn_attack_detector.pipeline.load_graph_detector_bundle",
                    return_value=bundle,
                ),
                patch(
                    "gnn_attack_detector.pipeline.predict_graph_probabilities",
                    return_value=torch.tensor([0.9, 0.1]),
                ),
            ):
                report = evaluate_detector(config, project_root=root)

            self.assertEqual(report["detector_model_hash"], hash_file(model_path))
            evaluated_at = datetime.fromisoformat(report["evaluated_at_utc"])
            self.assertIsNotNone(evaluated_at.tzinfo)
            self.assertEqual(evaluated_at.utcoffset().total_seconds(), 0)
```

- [ ] **Step 2: Run the metadata test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector_core.py::DetectorCoreTests::test_evaluation_report_identifies_current_detector -q
```

Expected: FAIL with missing `detector_model_hash` or `evaluated_at_utc`.

- [ ] **Step 3: Add model hash and UTC evaluation time**

Add the import in `pipeline.py`:

```python
from datetime import datetime, timezone
```

Resolve the model path once before loading it:

```python
    model_path = _resolve(root, runtime["model_path"], "detector model")
    bundle = load_graph_detector_bundle(model_path)
```

Add these fields to the report dictionary immediately after `"status": "ok"`:

```python
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_model_hash": hash_file(model_path),
```

Keep report format version 3 because these are backward-compatible additive fields.

- [ ] **Step 4: Run the metadata and metric tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector_core.py::DetectorCoreTests::test_evaluation_report_identifies_current_detector tests/test_actual_metric_reporting_source.py tests/test_result_viewer.py -q
```

Expected: PASS.

- [ ] **Step 5: Inspect the narrow diff checkpoint**

Run:

```powershell
git diff -- src/gnn_attack_detector/pipeline.py tests/test_detector_core.py
```

Expected: report metadata plus its behavior test, without model-format or network-architecture changes. Leave the overlapping files unstaged.

### Task 3: Detect and display historical reports

**Files:**
- Modify: `tests/test_platform_workspace.py`
- Modify: `src/gnn_attack_detector/platform_service.py:12-18,94-133`
- Modify: `src/gnn_attack_detector/web_app.py:253-285`

- [ ] **Step 1: Write failing service tests for matching and changed models**

Replace the existing `graph_fingerprint` import in `tests/test_platform_workspace.py` with:

```python
from gnn_attack_detector.integrity import graph_fingerprint, hash_file
```

Add the following two tests to `PlatformWorkspaceTests`:

```python
    def test_platform_marks_report_current_when_model_hash_matches(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.write_bytes(b"detector-a")
            report["detector_model_hash"] = hash_file(model)
            report["evaluated_at_utc"] = "2026-09-07T12:00:00+00:00"
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = load_platform_evaluation(path, model_path=model)

        self.assertFalse(loaded["historical"])

    def test_platform_marks_report_historical_when_model_hash_changes(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.write_bytes(b"detector-a")
            report["detector_model_hash"] = hash_file(model)
            report["evaluated_at_utc"] = "2026-09-07T12:00:00+00:00"
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            model.write_bytes(b"detector-b")

            loaded = load_platform_evaluation(path, model_path=model)

        self.assertTrue(loaded["historical"])

    def test_platform_marks_legacy_report_without_hash_historical(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.write_bytes(b"current-detector")
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = load_platform_evaluation(path, model_path=model)

        self.assertTrue(loaded["historical"])
```

- [ ] **Step 2: Run the service tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_platform_workspace.py::PlatformWorkspaceTests::test_platform_marks_report_current_when_model_hash_matches tests/test_platform_workspace.py::PlatformWorkspaceTests::test_platform_marks_report_historical_when_model_hash_changes tests/test_platform_workspace.py::PlatformWorkspaceTests::test_platform_marks_legacy_report_without_hash_historical -q
```

Expected: FAIL because `load_platform_evaluation` does not accept `model_path` and does not return `historical`.

- [ ] **Step 3: Implement hash comparison in the service**

Add this import in `platform_service.py`:

```python
from gnn_attack_detector.integrity import hash_file
```

Extend the function signature:

```python
def load_platform_evaluation(
    report_path: str | Path,
    *,
    prediction_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
```

Immediately before the successful return, classify the report:

```python
    historical = False
    if model_path is not None:
        current_model = Path(model_path)
        recorded_hash = report.get("detector_model_hash")
        historical = bool(
            not current_model.is_file()
            or not isinstance(recorded_hash, str)
            or hash_file(current_model) != recorded_hash
        )
    return {
        "available": True,
        "report": report,
        "reason": None,
        "historical": historical,
    }
```

Missing hashes in old reports deliberately become historical when the platform supplies a model path. Calls that omit `model_path` retain their existing behavior.

- [ ] **Step 4: Run the service tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_platform_workspace.py::PlatformWorkspaceTests::test_platform_marks_report_current_when_model_hash_matches tests/test_platform_workspace.py::PlatformWorkspaceTests::test_platform_marks_report_historical_when_model_hash_changes tests/test_platform_workspace.py::PlatformWorkspaceTests::test_platform_marks_legacy_report_without_hash_historical tests/test_platform_workspace.py::PlatformWorkspaceTests::test_old_report_uses_matching_predictions_without_changing_saved_files -q
```

Expected: PASS.

- [ ] **Step 5: Write the failing UI metadata test**

Add this test to `WorkspaceUiTests`:

```python
    def test_evaluation_shows_model_identity_and_historical_warning(self):
        report = report_fixture()
        report["detector_model_hash"] = "0123456789abcdef"
        report["evaluated_at_utc"] = "2026-09-07T12:00:00+00:00"
        payload = {"available": True, "report": report, "historical": True}
        with (
            patch(
                "gnn_attack_detector.web_app.collect_platform_status",
                return_value=self.status,
            ),
            patch(
                "gnn_attack_detector.web_app.load_platform_evaluation",
                return_value=payload,
            ),
        ):
            app = self.app().run()
            app.radio(key="platform-page").set_value("准确率评测").run()

        self.assertTrue(any("历史模型" in item.value for item in app.warning))
        self.assertTrue(any("0123456789ab" in item.value for item in app.caption))
        self.assertTrue(any("2026-09-07T12:00:00+00:00" in item.value for item in app.caption))
```

- [ ] **Step 6: Run the UI test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_platform_workspace.py::WorkspaceUiTests::test_evaluation_shows_model_identity_and_historical_warning -q
```

Expected: FAIL because the evaluation page does not render report identity metadata.

- [ ] **Step 7: Pass the model path and render metadata**

Update the service call in `_evaluation`:

```python
        payload = load_platform_evaluation(
            root / status["evaluation_report"],
            prediction_path=(
                root / status["prediction_report"]
                if status.get("prediction_report")
                else None
            ),
            model_path=(
                root / status["detector_model"]
                if status.get("detector_model")
                else None
            ),
        )
```

Immediately after `report = payload["report"]`, add:

```python
    if payload.get("historical"):
        st.warning(
            "当前显示的是历史模型评测结果；下一次成功完成通用训练和全量评测后会自动更新。"
        )
    evaluated_at = report.get("evaluated_at_utc")
    model_hash = report.get("detector_model_hash")
    if isinstance(evaluated_at, str) and isinstance(model_hash, str):
        st.caption(f"评测时间（UTC）：{evaluated_at} · 检测器：{model_hash[:12]}")
```

- [ ] **Step 8: Run platform service and UI tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_platform_workspace.py -q
```

Expected: PASS, including the existing Cora/Citeseer dropdown-switch test.

- [ ] **Step 9: Inspect the narrow diff checkpoint**

Run:

```powershell
git diff -- src/gnn_attack_detector/platform_service.py src/gnn_attack_detector/web_app.py tests/test_platform_workspace.py
```

Expected: model/report identity handling only. Leave these overlapping dirty files unstaged.

### Task 4: Make prediction publication recoverable

**Files:**
- Modify: `tests/test_detector_core.py`
- Modify: `src/gnn_attack_detector/pipeline.py:74-79`

- [ ] **Step 1: Write a failing interruption test**

Add `_write_jsonl` to the pipeline imports in `tests/test_detector_core.py`, then add:

```python
    def test_jsonl_replacement_failure_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text("previous\n", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    _write_jsonl(path, [{"sample_id": "new"}])
            self.assertEqual(path.read_text(encoding="utf-8"), "previous\n")
```

- [ ] **Step 2: Run the interruption test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector_core.py::DetectorCoreTests::test_jsonl_replacement_failure_preserves_previous_file -q
```

Expected: FAIL because the current writer directly replaces the destination contents and never calls `Path.replace`.

- [ ] **Step 3: Write JSONL through a temporary sibling**

Replace `_write_jsonl` with:

```python
def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
```

- [ ] **Step 4: Run the interruption and report tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_detector_core.py::DetectorCoreTests::test_jsonl_replacement_failure_preserves_previous_file tests/test_detector_core.py::DetectorCoreTests::test_evaluation_report_identifies_current_detector -q
```

Expected: PASS.

- [ ] **Step 5: Inspect the narrow diff checkpoint**

Run:

```powershell
git diff -- src/gnn_attack_detector/pipeline.py tests/test_detector_core.py
```

Expected: temporary-file publication and the two new pipeline tests, with existing feature-cache changes preserved.

### Task 5: Document behavior and run full verification

**Files:**
- Modify: `README.md:50-68`
- Verify: all files modified in Tasks 1-4

- [ ] **Step 1: Update the workflow documentation**

Replace the paragraph that says training and evaluation only merge selected datasets with:

```markdown
脚本按“准备本次所选数据集 → 为所选数据集生成训练/验证图 → 合并所有已准备
数据集并训练一次通用检测器 → 为所选数据集生成测试图 → 用当前通用检测器重新
评测所有已准备数据集”的顺序运行。`configs/run.yaml` 的数据集选择只控制本次
需要生成或更新的数据；不同数据集仍分别保存攻击图、清单和特征文件。一次全选
会直接联合训练，分批运行则自动复用此前数据并逐步扩展同一个检测器。平台按
数据集显示独立准确率，所有当前结果都来自报告标记的同一个模型版本。
```

Keep the existing resumability and graph-count documentation below it.

- [ ] **Step 2: Run focused regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_selected_workflow.py tests/test_detector_core.py tests/test_platform_workspace.py tests/test_result_viewer.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the complete test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass with no new warnings or errors.

- [ ] **Step 4: Check formatting and accidental changes**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` is clean. Status still contains the user's pre-existing changes plus the files intentionally modified by this plan; no artifact, cache, model, report, or unrelated source file is newly tracked.

- [ ] **Step 5: Review requirement coverage**

Confirm from test output and the final diff:

```text
one detector path
selection-scoped generation
all-saved-profile training
all-saved-profile evaluation
per-profile platform metrics
current/historical model identity
recoverable report and prediction publication
```

Do not create per-dataset model files or a model-history subsystem. Because the touched source files had pre-existing uncommitted edits, hand off the verified working-tree diff without staging those files as whole-file commits.
