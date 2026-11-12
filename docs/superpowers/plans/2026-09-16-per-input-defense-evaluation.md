# Per-Input Defense Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the selected DeepRobust defense independently for every uploaded perturbed graph using the authoritative DeepRobust node split.

**Architecture:** `defense_service.py` loads the selected profile's clean graph, restores its trusted labels and split onto the uploaded topology/features, and routes every catalog model through one per-input fitting function. Platform readiness depends on the clean profile rather than cached defense checkpoints; detector and attack code remain untouched.

**Tech Stack:** Python 3.10, PyTorch 2.2.2, PyTorch Geometric 2.5.3, DeepRobust 0.2.9, pytest.

---

### Task 1: Define per-input execution semantics

**Files:**
- Modify: `src/gnn_attack_detector/defenses/registry.py`
- Test: `tests/test_defense_registry.py`

- [x] Add a failing test asserting every model except ProGNN uses `input_fit`, ProGNN uses `offline`, and no defense requires a checkpoint.
- [x] Run `pytest tests/test_defense_registry.py -q` and confirm the existing checkpoint modes fail the test.
- [x] Change only the catalog execution modes required by the test.
- [x] Re-run the focused test and confirm it passes.

### Task 2: Retrain every defense on each graph

**Files:**
- Modify: `src/gnn_attack_detector/defense_service.py`
- Test: `tests/test_defense_service.py`

- [x] Add failing tests showing evaluation obtains the split from the clean DeepRobust graph, replaces uploaded labels with trusted labels, and calls the per-input trainer for both clean and attacked graphs.
- [x] Add a failing test showing checkpoint paths and `prepare_profile_defenses` are not used by evaluation.
- [x] Run the focused tests and confirm they fail for the old checkpoint route.
- [x] Add DeepRobust class routes for GCN, GCN-Jaccard and SimPGCN.
- [x] Implement one `_fit_defense` dispatcher for scipy-style, PyG-style and ProGNN models, preserving CUDA-to-CPU fallback.
- [x] Make `evaluate_robust_probe` always call that dispatcher twice with identical seed and split.
- [x] Remove checkpoint loading and preparation code that is no longer called.
- [x] Re-run `pytest tests/test_defense_service.py -q` and confirm it passes.

### Task 3: Remove checkpoint preparation from platform behavior

**Files:**
- Modify: `src/gnn_attack_detector/preparation.py`
- Modify: `src/gnn_attack_detector/platform_service.py`
- Modify: `src/gnn_attack_detector/web_app.py`
- Test: `tests/test_probe_contract_source.py`
- Test: `tests/test_platform_workspace.py`

- [x] Add failing source and service tests asserting defense readiness requires only the clean graph and UI text says every graph is retrained.
- [x] Run the focused tests and confirm the old preparation/checkpoint wording fails.
- [x] Stop `prepare_project` from training optional defense checkpoints while preserving ordinary GCN preparation.
- [x] Update platform readiness, execution labels and instructions.
- [x] Re-run the focused tests and confirm they pass.

### Task 4: Documentation and regression verification

**Files:**
- Modify: `README.md`

- [x] Replace checkpoint preparation instructions with the per-input training protocol and DeepRobust split requirement.
- [x] Run `python -m py_compile` for modified source files.
- [x] Run the focused defense/platform test set.
- [x] Run `pytest -q` and require the full suite to pass.
- [x] Run `git diff --check` on the files changed by this plan.
