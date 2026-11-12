# DeepRobust Universal Detector Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed-backend 22-method prototype with a smaller open-source core containing 20 DeepRobust protocols, dataset profiles, a dataset-independent detector contract, and a robust-model catalog.

**Architecture:** Keep the proven complete-graph detector and PT/NPZ boundary, but make attacks, profiles, probes, and the final classifier separate modules. Remove legacy backends and stage-specific naming only after callers are migrated and tests prove the new public contracts.

**Tech Stack:** Python 3.10, PyTorch 2.2, PyTorch Geometric 2.5, DeepRobust 0.2.9, Streamlit, unittest.

---

### Task 1: Freeze the new registries

**Files:**
- Modify: `tests/test_attack_registry.py`
- Create: `tests/test_defense_registry.py`
- Modify: `src/gnn_attack_detector/attacks/registry.py`
- Create: `src/gnn_attack_detector/defenses/__init__.py`
- Create: `src/gnn_attack_detector/defenses/registry.py`

- [ ] Write tests asserting the exact 20 protocol IDs, the single `deeprobust` backend, parameter variants, the 10 defense IDs, three online probes, and offline ProGNN.
- [ ] Run the two registry tests and confirm they fail against the mixed 22-method registry and missing defense package.
- [ ] Implement frozen dataclass registries with unique-ID checks and lookup helpers.
- [ ] Run the registry tests and confirm they pass.

### Task 2: Add dataset profiles and universal dataset metadata

**Files:**
- Create: `tests/test_profiles.py`
- Modify: `tests/test_detector_dataset.py`
- Create: `src/gnn_attack_detector/profiles.py`
- Modify: `src/gnn_attack_detector/detector_dataset.py`
- Modify: `src/gnn_attack_detector/detector_model.py`

- [ ] Write failing tests for relative profile paths, distinct Cora/Cora-LCC profiles, per-row `dataset_ids`, and detector bundles without a reference-graph or victim-model hash.
- [ ] Implement the minimal `DatasetProfile` loader and version-3 detector dataset/bundle formats.
- [ ] Preserve explicit rejection of incompatible feature schemas while allowing rows from multiple dataset IDs.
- [ ] Run focused profile, dataset, and model tests.

### Task 3: Collapse attack execution to DeepRobust

**Files:**
- Modify: `tests/test_deeprobust_adapter.py`
- Modify: `tests/test_graph_attacks.py`
- Modify: `src/gnn_attack_detector/attacks/deeprobust_adapter.py`
- Modify: `src/gnn_attack_detector/graph_attacks.py`
- Delete: `src/gnn_attack_detector/attacks/greatx_adapter.py`
- Delete: `src/gnn_attack_detector/attacks/pyg_adapter.py`
- Delete: `src/gnn_attack_detector/attacks/project_adapter.py`
- Delete: `tests/test_greatx_adapter.py`

- [ ] Write failing dispatch tests for Random add/delete/flip, DICE add/delete/mix, embedding add/delete, degree add/delete, eigenvector delete, Nettack structure/feature/hybrid, and IGAttack.
- [ ] Add the smallest parameterized DeepRobust adapters and map all 20 IDs through one dispatcher.
- [ ] Remove injection/backdoor and mixed-backend branches.
- [ ] Delete adapters only after `rg` shows no production callers.
- [ ] Run focused attack tests.

### Task 4: Migrate configuration, corpus, pipeline, and platform

**Files:**
- Modify: `configs/base.yaml`
- Modify: `src/gnn_attack_detector/config.py`
- Modify: `src/gnn_attack_detector/corpus_policy.py`
- Modify: `src/gnn_attack_detector/day4_corpus.py`
- Modify: `src/gnn_attack_detector/day4_pipeline.py`
- Modify: `src/gnn_attack_detector/platform_generation.py`
- Modify: `src/gnn_attack_detector/platform_service.py`
- Modify: `src/gnn_attack_detector/web_app.py`
- Modify: corresponding tests under `tests/`

- [ ] Write failing tests proving configuration uses `profiles`, `attacks`, `corpus`, and `detector` rather than injection/backdoor and fixed-Cora fields.
- [ ] Route strength selection by protocol `strength_unit`; ensure feature attacks persist `x`.
- [ ] Add defense-catalog status to the platform without running all defenses during detection.
- [ ] Remove exact victim/reference hashes from the universal detector contract; retain profile-level integrity checks.
- [ ] Run focused configuration, corpus, pipeline, service, and UI tests.

### Task 5: Simplify deployment and open-source files

**Files:**
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `requirements.txt`
- Modify: `README.md`
- Modify: `scripts/setup_day4_venv.ps1`
- Modify: `scripts/run_day4_train.ps1`
- Delete: obsolete environment files, cleanup scripts, daily logs, and superseded specs/plans.

- [ ] Write source-level tests asserting setup installs DeepRobust but not GreatX or PyG extension wheels, and the runtime registry has exactly 20 protocols.
- [ ] Add MIT and DeepRobust notices and document the data/profile/detector distinction.
- [ ] Remove files only when `rg` confirms no active code or current documentation references them.
- [ ] Run Windows script compatibility tests.

### Task 6: Verification

**Files:**
- Test: `tests/`

- [ ] Compile every Python file with `compileall`.
- [ ] Run the complete lightweight suite with `PYTHONPATH=src` in the available environment.
- [ ] If heavy dependencies are unavailable locally, record that fact and provide the exact server command for the dependency-backed suite.
- [ ] Search for forbidden active references: `greatx`, `grbcd`, `prbcd`, `graphfool`, `random_injection`, `backdoor`, `twenty-two`, and old node-level artifacts.
- [ ] Inspect `git diff --check`, deletion list, and final file tree before reporting completion.

