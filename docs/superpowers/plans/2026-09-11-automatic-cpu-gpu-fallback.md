# Automatic CPU/GPU Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run compatible training and attack work on CUDA when available, use CPU otherwise, and retry the current unit once on CPU after a recognized CUDA failure.

**Architecture:** Add one small device-policy module used by victim training, detector training, and corpus attack execution. Keep explicit CPU-only boundaries for NumPy/SciPy attacks, pass the resolved device only to adapters known to support it, and keep persisted tensors on CPU.

**Tech Stack:** Python 3.10+, PyTorch 2.2.2, PyTorch Geometric 2.5.3, DeepRobust 0.2.9, GreatX 0.0.1b0, unittest/pytest.

---

### Task 1: Shared device policy

**Files:**
- Create: `src/gnn_attack_detector/device.py`
- Create: `tests/test_device.py`
- Modify: `src/gnn_attack_detector/victim.py`

- [ ] **Step 1: Write failing tests** for `resolve_device("auto")`, forced CPU, requested CUDA without availability falling back to CPU, and `is_cuda_failure()` accepting CUDA OOM/runtime availability errors while rejecting ordinary `ValueError` and unrelated `RuntimeError`.
- [ ] **Step 2: Run `python -m pytest tests/test_device.py -q`** and verify failure because the module does not exist.
- [ ] **Step 3: Implement** `resolve_device(requested, *, cuda_available=torch.cuda.is_available)` and `is_cuda_failure(exc)` with only `auto`, `cpu`, and `cuda` accepted. CUDA requests resolve to CPU when CUDA is unavailable.
- [ ] **Step 4: Replace** the private victim resolver with the shared resolver and print `[DEVICE] victim training: <device>`.
- [ ] **Step 5: Run `python -m pytest tests/test_device.py tests/test_victim.py -q`** and verify pass.

### Task 2: CUDA-aware detector training with one CPU retry

**Files:**
- Modify: `src/gnn_attack_detector/detector_model.py`
- Modify: `src/gnn_attack_detector/pipeline.py`
- Modify: `tests/test_detector_core.py`

- [ ] **Step 1: Write failing tests** showing detector tensors/model use a supplied CUDA-like resolved device and a recognized CUDA failure causes exactly one CPU retry while ordinary errors propagate.
- [ ] **Step 2: Run the focused detector tests** and confirm the new expectations fail.
- [ ] **Step 3: Add** a `device="auto"` argument to detector training, move normalized values, labels, masks, and weights to the resolved device, retain best state and bundle tensors on CPU, and wrap only the training unit with recognized CUDA fallback.
- [ ] **Step 4: Pass** `config["runtime"]["device"]` from the pipeline and print `[DEVICE] detector training: <device>` plus `[GPU-FALLBACK]` when retried.
- [ ] **Step 5: Run `python -m pytest tests/test_detector_core.py tests/test_selected_workflow.py -q`** and verify pass.

### Task 3: Attack device capability and corpus fallback

**Files:**
- Modify: `src/gnn_attack_detector/attacks/registry.py`
- Modify: `src/gnn_attack_detector/graph_attacks.py`
- Modify: `src/gnn_attack_detector/corpus.py`
- Modify: `tests/test_attack_registry.py`
- Modify: `tests/test_multi_profile_corpus.py`

- [ ] **Step 1: Write failing tests** for an explicit `cuda_capable` attack property and corpus behavior that invokes a CUDA-capable attack with `device="cuda"`, retries once with `device="cpu"` after a CUDA error, does not retry non-CUDA errors, and routes CPU-only attacks directly to CPU.
- [ ] **Step 2: Run the focused tests** and confirm failures identify the missing capability/device arguments.
- [ ] **Step 3: Add** `cuda_capable` to `AttackSpec`; mark only implementations that can safely keep their optimization tensors in PyTorch on CUDA. Leave NumPy/SciPy/spectral implementations CPU-only.
- [ ] **Step 4: Thread** an optional `device` argument through `run_complete_graph_attack` into the compatible adapter calls without changing saved graph format.
- [ ] **Step 5: Resolve and print** the attack device in corpus generation. On a recognized CUDA exception, clear CUDA cache and retry that same candidate once from the clean graph on CPU; preserve normal rejection attempts and re-raise unrelated exceptions.
- [ ] **Step 6: Run the attack registry and corpus tests** and verify pass.

### Task 4: Make compatible adapters honor the selected device

**Files:**
- Modify: `src/gnn_attack_detector/attacks/deeprobust_adapter.py`
- Modify: `src/gnn_attack_detector/attacks/greatx_adapter.py`
- Modify: `tests/test_external_attack_adapters.py`

- [ ] **Step 1: Write failing source/behavior tests** proving GPU-capable constructors receive the requested device and returned portable graphs are moved back to CPU.
- [ ] **Step 2: Run `python -m pytest tests/test_external_attack_adapters.py -q`** and verify failure.
- [ ] **Step 3: Replace hard-coded `device="cpu"` only in the verified PyTorch-native attack paths** with the supplied device. Keep adjacency/features converted for NumPy/SciPy attacks on CPU and keep validation/persistence outputs on CPU.
- [ ] **Step 4: Run adapter tests** and verify pass.

### Task 5: CUDA environment setup and regression verification

**Files:**
- Create: `scripts/setup_windows_cuda.ps1`
- Create: `scripts/setup_windows_cuda.bat`
- Modify: `README.md`
- Modify: `tests/test_windows_scripts.py` if present, otherwise add the checks to `tests/test_current_module_layout.py`.

- [ ] **Step 1: Write a failing script contract test** requiring the CUDA setup script to install `torch==2.2.2` from the `cu121` index and PyG sparse wheels from `torch-2.2.0+cu121.html`, without modifying the CPU setup script.
- [ ] **Step 2: Run the focused script test** and verify failure because the CUDA setup script does not exist.
- [ ] **Step 3: Add the CUDA setup scripts** mirroring the established CPU setup flow, with CUDA-specific PyTorch/PyG sources and a final assertion that CUDA is available.
- [ ] **Step 4: Document** `runtime.device: auto`, CPU and CUDA setup commands, actual-device log messages, and CPU fallback behavior.
- [ ] **Step 5: Run `python -m pytest -q`** and verify the complete suite passes.
- [ ] **Step 6: Run `git diff --check`** and inspect `git status --short` so only intended files are reported as this feature's changes.
