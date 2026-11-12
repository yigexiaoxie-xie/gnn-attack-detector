# Selected Dataset Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure selected runs never train or evaluate against unselected saved dataset profiles.

**Architecture:** Add an optional profile filter at the shared dataset-loading boundary and pass the current run selection through the workflow. Keep each profile artifact unchanged and stop evaluation when selected test generation fails.

**Tech Stack:** Python 3.10+, PyTorch, unittest/pytest

---

### Task 1: Lock workflow selection behavior with failing tests

**Files:**
- Modify: `tests/test_selected_workflow.py`

- [ ] Add a test asserting `profile_ids=("cora-lcc", "citeseer-lcc")` is passed to the single training and evaluation calls.
- [ ] Add a test asserting evaluation is not called when one selected profile fails test generation.
- [ ] Run `python -m pytest tests/test_selected_workflow.py -q` and verify both tests fail for the missing behavior.

### Task 2: Filter shared profile dataset loading

**Files:**
- Modify: `tests/test_detector_core.py`
- Modify: `src/gnn_attack_detector/pipeline.py`

- [ ] Add a test with saved Cora, Citeseer, and Polblogs feature files that requests only Cora and Citeseer.
- [ ] Run that test and verify it fails because Polblogs is still returned.
- [ ] Add optional `profile_ids` validation/filtering to `_available_profile_datasets`, `train_detector`, and `evaluate_detector`.
- [ ] Run the focused detector and workflow tests and verify they pass.

### Task 3: Pass the selected scope and stop stale evaluation

**Files:**
- Modify: `src/gnn_attack_detector/workflow.py`

- [ ] Pass `profile_ids=selection.profiles` to universal training and evaluation.
- [ ] Return the collected test error without calling evaluation when any selected test preparation fails.
- [ ] Run `python -m pytest tests/test_selected_workflow.py tests/test_detector_core.py -q`.

### Task 4: Verify compatibility

**Files:**
- Test: full `tests/` suite

- [ ] Run `python -m pytest -q`.
- [ ] Run `git diff --check` for the modified source and test files.
- [ ] Confirm no artifact deletion or migration is required.
