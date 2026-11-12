# Multi-Profile Selected Attack Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow one resumable run to select multiple known dataset profiles and multiple attacks from commented YAML lists, while retaining one universal detector and the full 21-attack catalog.

**Architecture:** Keep `configs/base.yaml` as the authoritative catalog and add a small `configs/run.yaml` selection file. A workflow module validates the selection, creates an in-memory/per-run profile configuration, generates train/validation data for every selected profile, trains the universal detector once, then generates held-out test data and evaluates. Corpus identity uses the canonical registry index so selecting subsets never changes sample IDs, and existing unselected manifest rows remain intact.

**Tech Stack:** Python 3.10, PyYAML, PyTorch 2.2, PyTorch Geometric 2.5, DeepRobust 0.2.9, optional pinned GreatX adapter, `unittest`, Windows PowerShell 5.1.

---

## File Structure

- Create `configs/run.yaml`: human-edited commented selection list.
- Create `src/gnn_attack_detector/run_selection.py`: validate selection and build selected per-profile configurations.
- Create `src/gnn_attack_detector/workflow.py`: orchestrate preparation, corpus generation, one training pass, test generation, evaluation, and failure summary.
- Create `src/gnn_attack_detector/attacks/greatx_adapter.py`: isolate GreatX conversion and GFAttack/FGAttack calls.
- Create `src/gnn_attack_detector/attacks/structack_adapter.py`: isolate Structack structure-only attack.
- Create `scripts/run_selected_attacks.bat` and `scripts/run_selected_attacks.ps1`: Windows entry point.
- Modify `configs/base.yaml`: four profiles and the fixed 21-attack catalog.
- Modify `src/gnn_attack_detector/attacks/registry.py`: replace slow removed protocols, preserve IGAttack, record backend.
- Modify `src/gnn_attack_detector/attacks/deeprobust_adapter.py`: add PRBCD adapter and remove unused node-embedding/eigenvector functions.
- Modify `src/gnn_attack_detector/graph_attacks.py`: dispatch all three backends.
- Modify `src/gnn_attack_detector/config.py`: validate the authoritative catalog and selected subsets separately.
- Modify `src/gnn_attack_detector/data.py`: handle Polblogs deterministically when node features are absent or empty.
- Modify `src/gnn_attack_detector/preparation.py`: accept an already-validated selected configuration.
- Modify `src/gnn_attack_detector/corpus.py`: stable IDs, explicit profile paths, subset-preserving manifests, selected verification.
- Modify `src/gnn_attack_detector/pipeline.py`: accept selected configuration/profile and report accumulated versus formal coverage.
- Modify `src/gnn_attack_detector/cli.py`: add `run-selected` and selection inspection.
- Modify `scripts/setup_windows_cpu.ps1`, `requirements.txt`, `THIRD_PARTY_NOTICES.md`: install/check required backends and document provenance.
- Modify `README.md`: explain the only normal operation, editing `configs/run.yaml`.
- Create `tests/test_run_selection.py`, `tests/test_selected_workflow.py`, `tests/test_multi_profile_corpus.py`.
- Modify attack, configuration, data, script, and source-contract tests.

### Task 1: Add the human-edited multi-selection configuration

**Files:**
- Create: `configs/run.yaml`
- Create: `src/gnn_attack_detector/run_selection.py`
- Test: `tests/test_run_selection.py`

- [ ] **Step 1: Write the failing selection tests**

```python
from pathlib import Path
import tempfile
import unittest

from gnn_attack_detector.run_selection import load_run_selection


class RunSelectionTests(unittest.TestCase):
    def test_loads_multiple_profiles_and_attacks_in_order(self):
        text = """run:
  profiles: [cora-lcc, citeseer-lcc]
  attacks: [random_add, dice_mix]
  partitions: [train, val, test]
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.yaml"
            path.write_text(text, encoding="utf-8")
            selected = load_run_selection(
                path,
                known_profiles=("cora-lcc", "citeseer-lcc", "pubmed-lcc"),
                known_attacks=("random_add", "dice_mix", "igattack"),
            )
        self.assertEqual(selected.profiles, ("cora-lcc", "citeseer-lcc"))
        self.assertEqual(selected.attacks, ("random_add", "dice_mix"))
        self.assertEqual(selected.partitions, ("train", "val", "test"))

    def test_rejects_empty_duplicate_and_unknown_items(self):
        cases = (
            ("profiles: []", "profiles must contain"),
            ("profiles: [cora-lcc, cora-lcc]", "profiles must contain unique"),
            ("profiles: [unknown]", "unknown profile"),
        )
        for profiles, message in cases:
            text = f"run:\n  {profiles}\n  attacks: [random_add]\n  partitions: [train]\n"
            with self.subTest(profiles=profiles), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "run.yaml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_run_selection(path, known_profiles=("cora-lcc",), known_attacks=("random_add",))
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_run_selection -v`

Expected: `ModuleNotFoundError: No module named 'gnn_attack_detector.run_selection'`.

- [ ] **Step 3: Implement the minimal immutable selection loader**

```python
@dataclass(frozen=True)
class RunSelection:
    profiles: tuple[str, ...]
    attacks: tuple[str, ...]
    partitions: tuple[str, ...]


def _selected(values, known, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"run.{label} must contain at least one item")
    normalized = tuple(str(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"run.{label} must contain unique items")
    unknown = set(normalized) - set(known)
    if unknown:
        raise ValueError(f"unknown {label[:-1]}: {sorted(unknown)}")
    return normalized


def load_run_selection(path, *, known_profiles, known_attacks):
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    run = loaded.get("run") if isinstance(loaded, dict) else None
    if not isinstance(run, dict):
        raise ValueError("run is required")
    return RunSelection(
        profiles=_selected(run.get("profiles"), known_profiles, "profiles"),
        attacks=_selected(run.get("attacks"), known_attacks, "attacks"),
        partitions=_selected(run.get("partitions"), ("train", "val", "test"), "partitions"),
    )
```

- [ ] **Step 4: Add `configs/run.yaml` with every item visible**

List all four profiles and all 21 attacks. Leave a small fast example uncommented and prefix every inactive list item with `# -`. Do not create alternate run files.

- [ ] **Step 5: Run the selection tests**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_run_selection -v`

Expected: all selection tests pass.

- [ ] **Step 6: Commit**

```powershell
git add configs/run.yaml src/gnn_attack_detector/run_selection.py tests/test_run_selection.py
git commit -m "feat: add multi-profile attack run selection"
```

### Task 2: Register the four datasets and fixed 21 attacks

**Files:**
- Modify: `configs/base.yaml`
- Modify: `src/gnn_attack_detector/attacks/registry.py`
- Modify: `src/gnn_attack_detector/data.py`
- Modify: `tests/test_attack_registry.py`
- Modify: `tests/test_configuration.py`
- Modify: `tests/test_profiles.py`
- Create: `tests/test_known_datasets.py`

- [ ] **Step 1: Write failing catalog tests**

```python
EXPECTED = (
    "random_add", "random_delete", "random_flip",
    "dice_add", "dice_delete", "dice_mix",
    "degree_add", "degree_delete", "prbcd", "structack",
    "gfattack", "fgattack_untargeted", "rnd",
    "nettack_structure", "nettack_feature", "nettack_hybrid",
    "fga", "sgattack", "pgdattack", "metattack", "igattack",
)

def test_registry_matches_the_fixed_21_protocol_contract(self):
    self.assertEqual(SUPPORTED_ATTACKS, EXPECTED)
    self.assertEqual(len(SUPPORTED_ATTACKS), 21)
    self.assertIn("igattack", SUPPORTED_ATTACKS)
    self.assertNotIn("node_embedding_add", SUPPORTED_ATTACKS)

def test_base_config_registers_four_lcc_profiles(self):
    config = load_config(ROOT / "configs" / "base.yaml")
    self.assertEqual(
        tuple(config["profiles"]),
        ("cora-lcc", "citeseer-lcc", "polblogs-lcc", "pubmed-lcc"),
    )
```

Add a fake DeepRobust dataset test whose `features` is `None` and confirm Polblogs receives a deterministic `[num_nodes, 2]` degree-based feature matrix rather than crashing.

- [ ] **Step 2: Run and verify expected failures**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_attack_registry tests.test_configuration tests.test_known_datasets -v`

Expected: registry/profile/feature assertions fail against the current 20-protocol, Cora-only configuration.

- [ ] **Step 3: Replace the registry and add profiles**

Use `AttackSpec.backend` values `deeprobust`, `greatx`, and `structack`. Mark Metattack `precomputed_only`; keep PGD/PRBCD/IG and external adapters non-parallel. Set the known LCC expectations to Cora 2485, Citeseer 2110, Polblogs 1222, and Pubmed 19717, each with its own clean/probe path.

- [ ] **Step 4: Add deterministic Polblogs structural features**

When DeepRobust returns missing or zero-width features, compute exactly two normalized columns from the clean adjacency: node degree divided by maximum degree and a constant-one bias column. Preserve supplied features for all other datasets.

- [ ] **Step 5: Run catalog and dataset tests**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_attack_registry tests.test_configuration tests.test_profiles tests.test_known_datasets -v`

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add configs/base.yaml src/gnn_attack_detector/attacks/registry.py src/gnn_attack_detector/data.py tests/test_attack_registry.py tests/test_configuration.py tests/test_profiles.py tests/test_known_datasets.py
git commit -m "feat: register four datasets and 21 attacks"
```

### Task 3: Add real backend adapters without mixing libraries into the corpus

**Files:**
- Modify: `src/gnn_attack_detector/attacks/deeprobust_adapter.py`
- Create: `src/gnn_attack_detector/attacks/greatx_adapter.py`
- Create: `src/gnn_attack_detector/attacks/structack_adapter.py`
- Modify: `src/gnn_attack_detector/graph_attacks.py`
- Modify: `tests/test_attack_dispatch_current.py`
- Create: `tests/test_external_attack_adapters.py`

- [ ] **Step 1: Write failing dispatch tests**

Test that `prbcd` dispatches to the DeepRobust adapter, `gfattack` and `fgattack_untargeted` dispatch to GreatX, `structack` dispatches to its isolated adapter, and each returns exactly the requested number of unique undirected changes. Inject fake adapter callables so the unit test does not import or run expensive libraries.

```python
def test_prbcd_uses_global_budget_and_backend_adapter(self):
    calls = []
    result = run_global_protocol(
        make_graph(), torch.nn.Linear(1, 1), "prbcd", 0.34, 7,
        adapter=lambda graph, **kwargs: one_edge_change(graph, calls, kwargs),
    )
    self.assertEqual(result.actual_modifications, 1)
    self.assertEqual(calls[0]["operation"], "flip")
```

- [ ] **Step 2: Run and verify unknown-protocol/dispatch failures**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_attack_dispatch_current tests.test_external_attack_adapters -v`

Expected: new registry names have no adapters.

- [ ] **Step 3: Implement the DeepRobust PRBCD adapter**

Import PRBCD lazily, convert the PyG graph and trained surrogate to the API required by DeepRobust 0.2.9, seed NumPy/Torch through `_isolated_random_seed`, request the exact undirected budget, and convert `modified_edge_index` or `modified_adj` back through the shared adjacency validator. Raise a clear `RuntimeError` if the installed DeepRobust build does not expose PRBCD.

- [ ] **Step 4: Implement isolated GreatX adapters**

Keep all GreatX imports inside `_load_greatx_attack`. Convert only at this boundary. GFAttack is targeted and must be accumulated through `run_multi_target_protocol`; FGAttack Untargeted uses the global edge-rate budget. Validate symmetry, binary edges, node range, no self-loops, and exact unique-change budget after conversion.

- [ ] **Step 5: Implement Structack as a deterministic structure-only adapter**

Use the published low-degree/low-similarity addition rule: rank nodes by degree, split the lowest-centrality candidate set deterministically from the seed, greedily add non-edges with minimal neighbor-set Jaccard similarity, and stop at the exact budget. Never label random edge addition as Structack; preserve the structural ranking and record backend `structack`.

- [ ] **Step 6: Remove dead node-embedding/eigenvector adapter code**

Delete only functions and loaders with no registered caller. Keep degree add/delete because they remain in the catalog.

- [ ] **Step 7: Run adapter tests**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_attack_dispatch_current tests.test_external_attack_adapters -v`

Expected: all pass without performing network downloads.

- [ ] **Step 8: Commit**

```powershell
git add src/gnn_attack_detector/attacks src/gnn_attack_detector/graph_attacks.py tests/test_attack_dispatch_current.py tests/test_external_attack_adapters.py
git commit -m "feat: dispatch the 21 attack backends"
```

### Task 4: Make corpus generation subset-safe and profile-safe

**Files:**
- Modify: `src/gnn_attack_detector/corpus.py`
- Modify: `src/gnn_attack_detector/corpus_policy.py`
- Create: `tests/test_multi_profile_corpus.py`

- [ ] **Step 1: Write failing identity and preservation tests**

Create temporary manifests with `cora-lcc/random_add` and `cora-lcc/dice_mix`. Run a selected `random_add` regeneration and assert the DICE row remains. Generate the same attack once as a singleton selection and once as part of a larger list; assert sample IDs and seeds match. Assert new graph paths contain `<profile>/<partition>/<attack>`.

- [ ] **Step 2: Run and verify current behavior deletes or renumbers rows**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_multi_profile_corpus -v`

Expected: failures show unselected requested-partition rows are removed, selection order changes attack index, and profile is absent from the artifact path.

- [ ] **Step 3: Preserve accumulated rows**

Change the retained-record condition to keep a row when its partition is not requested **or** its attack is not selected. Filter verification counts to requested partitions plus selected attacks. Do not count accumulated unselected attacks as quota errors.

- [ ] **Step 4: Stabilize IDs and paths**

Derive `attack_index` from `SUPPORTED_ATTACKS.index(attack_name)`, never `enumerate(runtime["attacks"])`. Write new artifacts below `output_dir/profile_id/partition/attack`. Continue accepting an existing manifest row whose recorded legacy path passes all hash and graph-contract checks, so existing Cora graphs are reused without copying or deletion.

- [ ] **Step 5: Accept registered backend provenance**

Verify each record backend equals `attack_spec(record["attack_name"]).backend` rather than requiring every row to say `deeprobust`.

- [ ] **Step 6: Run corpus tests**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_multi_profile_corpus tests.test_corpus_policy_current -v`

Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add src/gnn_attack_detector/corpus.py src/gnn_attack_detector/corpus_policy.py tests/test_multi_profile_corpus.py tests/test_corpus_policy_current.py
git commit -m "fix: preserve selected multi-profile corpus artifacts"
```

### Task 5: Orchestrate selected profiles with one detector training pass

**Files:**
- Create: `src/gnn_attack_detector/workflow.py`
- Modify: `src/gnn_attack_detector/preparation.py`
- Modify: `src/gnn_attack_detector/pipeline.py`
- Create: `tests/test_selected_workflow.py`

- [ ] **Step 1: Write a failing workflow-order test**

Inject fake operations and assert this exact order for two profiles: prepare both, train/val generation for both, one universal train, test generation for both, one evaluation. Assert the configuration passed to each profile contains all selected attacks, not just the first.

```python
self.assertEqual(
    events,
    [
        "prepare:cora-lcc", "prepare:citeseer-lcc",
        "trainval:cora-lcc", "trainval:citeseer-lcc",
        "train:universal",
        "test:cora-lcc", "test:citeseer-lcc",
        "evaluate:universal",
    ],
)
```

- [ ] **Step 2: Run and verify the missing workflow failure**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_selected_workflow -v`

Expected: workflow module or entry function missing.

- [ ] **Step 3: Implement configuration selection without duplicate user files**

Use `copy.deepcopy(base_config)`, set `active_profile`, set `attacks.protocols` to the selected tuple, and set `attacks.require_all_protocols = False`. Update preparation/corpus/pipeline public functions to accept a validated mapping in addition to a file path through one shared `load_config_source` helper. Do not write persistent per-profile YAML files.

- [ ] **Step 4: Implement ordered workflow and aggregated errors**

Run every selected profile for each phase. Catch exceptions at profile/phase boundaries, append `{profile, phase, error}`, continue safe independent combinations, and skip universal training if any selected train/validation profile failed integrity validation. Return status `ok` or `failed` with generated/reused totals and errors; do not erase artifacts.

- [ ] **Step 5: Separate partial progress from formal acceptance**

Evaluation reports both `selected_attacks` and `catalog_attacks`. Formal `passed` remains false until at least `evaluation.minimum_attacks` unique attacks have eligible test rows and every required per-attack threshold passes. Extra accumulated attacks do not make selected-run verification fail.

- [ ] **Step 6: Run workflow and detector tests**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_selected_workflow tests.test_detector_core tests.test_detector_metrics -v`

Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add src/gnn_attack_detector/workflow.py src/gnn_attack_detector/preparation.py src/gnn_attack_detector/pipeline.py tests/test_selected_workflow.py
git commit -m "feat: run selected profiles through one detector"
```

### Task 6: Add the CLI and Windows one-command entry point

**Files:**
- Modify: `src/gnn_attack_detector/cli.py`
- Create: `scripts/run_selected_attacks.bat`
- Create: `scripts/run_selected_attacks.ps1`
- Modify: `scripts/setup_windows_cpu.ps1`
- Modify: `requirements.txt`
- Create or modify: `tests/test_cli_selection.py`
- Modify: `tests/test_open_source_layout.py`

- [ ] **Step 1: Write failing CLI/script tests**

Assert `build_parser()` accepts:

```text
run-selected --config configs/base.yaml --run-config configs/run.yaml
show-selection --config configs/base.yaml --run-config configs/run.yaml
```

Source-contract tests must assert the batch file invokes the PowerShell file, PowerShell uses `.venv\Scripts\python.exe`, and neither file hard-codes `E:` or a username.

- [ ] **Step 2: Run and verify command-not-found failures**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_cli_selection tests.test_open_source_layout -v`

Expected: `run-selected` and scripts are missing.

- [ ] **Step 3: Add CLI commands**

`show-selection` prints profiles, attacks, partitions, combination count, and expected attacked-graph count before any expensive work. `run-selected` invokes the workflow and returns exit code `1` when status is failed.

- [ ] **Step 4: Add ASCII-only Windows scripts**

The PowerShell script sets the same cache/proxy environment as existing scripts, prints the selection first, then runs it. The BAT file contains only `@echo off`, resolves `%~dp0`, invokes PowerShell with `-ExecutionPolicy Bypass`, and propagates `%ERRORLEVEL%`.

- [ ] **Step 5: Update setup verification**

Verify 21 registry entries and import every lazy backend that is installed. Keep GreatX isolated: if its installation is required for an uncommented attack but unavailable, `show-selection` reports the specific missing dependency before graph generation begins.

- [ ] **Step 6: Run CLI/script tests**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_cli_selection tests.test_open_source_layout -v`

Expected: all pass.

- [ ] **Step 7: Commit**

```powershell
git add src/gnn_attack_detector/cli.py scripts/run_selected_attacks.bat scripts/run_selected_attacks.ps1 scripts/setup_windows_cpu.ps1 requirements.txt tests/test_cli_selection.py tests/test_open_source_layout.py
git commit -m "feat: add selected attack Windows runner"
```

### Task 7: Document provenance, quantities, and server operation

**Files:**
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `tests/test_open_source_layout.py`

- [ ] **Step 1: Write the failing documentation assertions**

Assert README contains `configs/run.yaml`, `run_selected_attacks.bat`, all four profile IDs, the comment/uncomment rule, the count formula, resumption behavior, and the warning that partial runs do not prove the 80% requirement. Assert third-party notices identify DeepRobust, GreatX, and Structack sources and licenses.

- [ ] **Step 2: Run and verify missing-documentation failures**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_open_source_layout -v`

Expected: the new usage/provenance assertions fail.

- [ ] **Step 3: Write concise beginner-facing instructions**

Document exactly:

```powershell
notepad .\configs\run.yaml
.\scripts\run_selected_attacks.bat
```

Explain that uncommented profiles and attacks form a Cartesian product; ordinary combinations produce 20 graphs, PGD 12, and Meta 4; existing valid artifacts are reused; the user should not delete `.venv` or `artifacts` between batches.

- [ ] **Step 4: Run documentation tests**

Run: `\.venv\Scripts\python.exe -m unittest tests.test_open_source_layout -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add README.md THIRD_PARTY_NOTICES.md tests/test_open_source_layout.py
git commit -m "docs: explain selected multi-dataset runs"
```

### Task 8: Full verification and server handoff

**Files:**
- Verify all modified files.

- [ ] **Step 1: Compile all Python sources**

Run: `\.venv\Scripts\python.exe -m compileall -q src tests`

Expected: exit code 0 with no syntax errors.

- [ ] **Step 2: Run the complete automated suite**

Run: `\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass, with only explicitly documented optional real-backend integration tests skipped.

- [ ] **Step 3: Validate both YAML files and preview a multi-selection**

Run:

```powershell
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli check-config --config configs/base.yaml
.\.venv\Scripts\python.exe -m gnn_attack_detector.cli show-selection --config configs/base.yaml --run-config configs/run.yaml
```

Expected: four profiles, 21 supported attacks, every selected item displayed, correct combination and expected graph counts.

- [ ] **Step 4: Run a cheap end-to-end smoke batch**

Temporarily select only `cora-lcc`, `random_add`, and `train/val`; run `scripts\run_selected_attacks.bat`; confirm profile preparation/reuse, generated/reused counts, manifest path, and no test generation before detector training. Restore the user-facing example selection afterwards.

- [ ] **Step 5: Inspect the final diff for accidental data/dependency changes**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; no `.venv`, `artifacts`, uploaded graphs, or machine-specific paths are staged.

- [ ] **Step 6: Commit final verification adjustments if any**

```powershell
git add README.md configs scripts src tests requirements.txt THIRD_PARTY_NOTICES.md
git commit -m "test: verify multi-profile selected attack workflow"
```

## Plan Self-Review

- Spec coverage: multi-profile selection, multi-attack selection, commented full catalog, four known datasets, 21 attacks with IGAttack, stable resumption, one detector, held-out ordering, partial-versus-formal evaluation, Windows entry point, and provenance are each covered by an implementation task.
- Placeholder scan: the plan contains no deferred implementation markers; optional real-backend tests have an explicit skip boundary and lightweight adapters still require unit coverage.
- Type consistency: `RunSelection.profiles`, `.attacks`, and `.partitions` are ordered tuples throughout; selected profile configurations retain the existing `active_profile` and `attacks.protocols` keys used by corpus/pipeline code.
