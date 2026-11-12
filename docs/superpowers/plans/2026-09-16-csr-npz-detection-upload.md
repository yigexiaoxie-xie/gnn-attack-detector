# CSR-NPZ Detection Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make attack-detection uploads select a dataset first, accept SciPy CSR-NPZ by attaching that dataset's trusted attributes, remove the attribute-type control, and rename the defense page.

**Architecture:** Extend the shared detection input loader so an explicit profile loads `clean.pt` and passes it to the existing validated graph loader. Keep complete NPZ/PT behavior unchanged, while the Streamlit upload tab always supplies the selected profile and relies on checkpoint-controlled automatic attribute handling.

**Tech Stack:** Python 3.10, PyTorch, SciPy, Streamlit, pytest.

---

### Task 1: Load CSR-NPZ through an explicit dataset profile

**Files:**
- Modify: `src/gnn_attack_detector/pipeline.py`
- Test: `tests/test_detector_core.py`

- [x] Add a test that saves a SciPy CSR adjacency, supplies `profile_id="cora-lcc"`, and proves `detect_graph` receives a graph whose attributes came from that profile's `clean.pt`.
- [x] Run the focused test and confirm it fails because `_load_detection_input` currently calls `load_graph_file(path)` without a trusted graph for NPZ.
- [x] Update `_load_detection_input` to load the explicit profile's clean graph and call `load_graph_file(path, clean_graph=clean)` for NPZ and complete graph inputs.
- [x] Keep the existing legacy compact-PT provenance validation unchanged.
- [x] Run the focused graph-input tests and confirm they pass.

### Task 2: Change the attack-detection upload controls

**Files:**
- Modify: `src/gnn_attack_detector/web_app.py`
- Test: `tests/test_platform_workspace.py`

- [x] Add source assertions that `_upload_tab` selects a dataset before the uploader, no longer renders `节点属性类型`, passes `profile_id` to detection, and uses the label `鲁棒防御模型`.
- [x] Run the focused platform test and confirm the old page fails those assertions.
- [x] Replace the attribute-type selector with `_select_dataset`, include the selected profile in the result signature, and pass it to `detect_platform_graphs`.
- [x] Remove the upload-time `attribute_mode` override so saved checkpoint policy remains authoritative.
- [x] Rename only the navigation and page heading text from `防御工具` to `鲁棒防御模型`.
- [x] Run the focused platform tests and confirm they pass.

### Task 3: Documentation and regression verification

**Files:**
- Modify: `README.md`

- [x] Document that upload detection requires selecting the source dataset and that CSR-NPZ borrows only the trusted node attributes.
- [x] Run `python -m compileall -q src tests`.
- [x] Run the focused graph/platform tests.
- [x] Run the full `pytest -q` suite and require zero failures.
- [x] Run `git diff --check` on the files changed by this plan.
