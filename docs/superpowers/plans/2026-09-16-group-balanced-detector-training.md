# Group-Balanced Detector Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate each dataset's actually available attack corpus and train/select the shared detector by per-dataset/per-attack validation quality instead of aggregate loss alone.

**Architecture:** Add one corpus-policy helper that classifies selected attacks as complete or unavailable per profile and rejects partial groups. Extend detector metrics with deterministic group-aware threshold calibration, then reuse those results in the existing detector training loop for hierarchical sample weights, checkpoint ranking, and target-based early stopping.

**Tech Stack:** Python 3.10, PyTorch, PyTorch Geometric, pytest, YAML.

---

### Task 1: Corpus availability and completeness

**Files:**
- Modify: `src/gnn_attack_detector/corpus_policy.py`
- Modify: `src/gnn_attack_detector/pipeline.py`
- Test: `tests/test_corpus_policy_current.py`

- [ ] Write failing tests proving absent attacks are reported as unavailable while partially present attacks fail with partition/strength details.
- [ ] Run `pytest tests/test_corpus_policy_current.py -q` and verify the new tests fail because the helper is missing.
- [ ] Implement the minimal policy helper and call it before `_build_profile_dataset` in `rebuild_detector`.
- [ ] Filter profile records to complete attacks and include the availability matrix in the rebuild result.
- [ ] Run `pytest tests/test_corpus_policy_current.py tests/test_detector_core.py -q`.

### Task 2: Hierarchical dataset/attack loss weights

**Files:**
- Modify: `src/gnn_attack_detector/detector_model.py`
- Test: `tests/test_detector_core.py`

- [ ] Write a failing unit test with unequal profile/attack sample counts and assert equal total attacked weight per profile and per attack within that profile, plus equal clean/attacked class totals.
- [ ] Run the focused test and verify it fails under the current profile/label-only weighting.
- [ ] Replace `balanced_sample_weights` with the minimum hierarchical implementation while preserving its existing call shape where practical.
- [ ] Run the focused tests and verify they pass.

### Task 3: Group-aware threshold, checkpoint ranking, and stopping

**Files:**
- Modify: `src/gnn_attack_detector/detector_metrics.py`
- Modify: `src/gnn_attack_detector/detector_model.py`
- Modify: `src/gnn_attack_detector/pipeline.py`
- Modify: `src/gnn_attack_detector/config.py`
- Modify: `configs/base.yaml`
- Test: `tests/test_detector_metrics.py`
- Test: `tests/test_detector_core.py`

- [ ] Write failing tests for one global threshold that satisfies clean specificity and maximizes worst `(profile, attack)` recall.
- [ ] Write a failing training test proving three consecutive all-group passes stop training and group-aware metadata is saved.
- [ ] Add validated detector model settings for target recall, clean specificity, consecutive epochs, and minimum validation group size.
- [ ] Implement deterministic calibration/ranking and replace aggregate-loss patience stopping without using test rows.
- [ ] Print epoch progress containing worst-group recall, passed/total groups, clean specificity, and threshold.
- [ ] Run focused detector tests.

### Task 4: Regression verification

**Files:**
- Modify only if a failing regression reveals a scoped compatibility issue.

- [ ] Run `pytest tests/test_corpus_policy_current.py tests/test_detector_metrics.py tests/test_detector_core.py tests/test_reference_free_detector.py tests/test_selected_workflow.py -q`.
- [ ] Run the complete test suite with `pytest -q`.
- [ ] Review `git diff` and confirm attack generation, detector architecture, and unrelated user changes were not modified.
