# Defense Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all ten configured defense models honestly selectable and executable without changing the shared attack detector.

**Architecture:** Keep the existing three probes intact, add execution metadata to the defense registry, and route evaluation through one defense service. Cache one-time model artifacts under each profile while treating GCN-SVD and ProGNN as per-input jobs.

**Tech Stack:** Python 3.10, PyTorch 2.2, PyTorch Geometric, DeepRobust 0.2.9, Streamlit, unittest/pytest.

---

### Task 1: Define execution contracts

**Files:**
- Modify: `src/gnn_attack_detector/defenses/registry.py`
- Test: `tests/test_defense_registry.py`

- [ ] Add failing assertions for `execution_mode`, cached model names, and per-input model names.
- [ ] Run `python -m pytest tests/test_defense_registry.py -q` and confirm the new assertions fail.
- [ ] Add the minimal registry fields and derived constants.
- [ ] Re-run the registry test and confirm it passes.

### Task 2: Add the unified defense service

**Files:**
- Modify: `src/gnn_attack_detector/defense_service.py`
- Create: `tests/test_defense_service.py`

- [ ] Add failing tests for artifact paths, existing-probe routing, unavailable optional dependencies, and result metadata.
- [ ] Run `python -m pytest tests/test_defense_service.py -q` and confirm failure is caused by missing service behavior.
- [ ] Implement graph validation, lazy DeepRobust imports, cached artifact loading, input-fit execution, and common accuracy reporting.
- [ ] Run the service tests and confirm they pass.

### Task 3: Prepare cached defense models independently

**Files:**
- Modify: `src/gnn_attack_detector/preparation.py`
- Modify: `src/gnn_attack_detector/cli.py`
- Test: `tests/test_defense_service.py`

- [ ] Add a failing test showing optional-model failure is reported without aborting existing probe preparation.
- [ ] Implement `prepare_profile_defenses` and call it from `prepare-defenses` after the existing probes are ready.
- [ ] Re-run the focused tests.

### Task 4: Expose real readiness and all models in Streamlit

**Files:**
- Modify: `src/gnn_attack_detector/platform_service.py`
- Modify: `src/gnn_attack_detector/web_app.py`
- Modify: `tests/test_platform_workspace.py`

- [ ] Add failing tests that expect all configured models and their execution/readiness fields.
- [ ] Replace the three-model hard-coded select box and status table with registry-driven data.
- [ ] Show checkpoint/input-fit/offline descriptions and actionable dependency errors.
- [ ] Run the platform tests.

### Task 5: Regression verification

**Files:**
- Modify: `README.md`

- [ ] Document one-time preparation, direct-call models, GCN-SVD, ProGNN, and optional GNNGuard.
- [ ] Run `python -m pytest -q`.
- [ ] Inspect `git diff --check` and the scoped diff to ensure detector training code and model paths are unchanged.

