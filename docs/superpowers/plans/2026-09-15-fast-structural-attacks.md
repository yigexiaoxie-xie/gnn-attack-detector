# Fast Structural Attacks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `bridge_add` and `hub_rewire` as fast, deterministic, structure-only attacks across all four configured graph profiles.

**Architecture:** A new dependency-free adapter operates only on `edge_index` and returns a cloned PyG graph. Existing global attack dispatch, budget auditing, corpus resumption, and platform registry discovery remain the single integration path. New registry entries are appended so every existing attack keeps its stable index and seed namespace.

**Tech Stack:** Python 3.10, PyTorch/PyG graph containers, standard-library seeded randomness, pytest/unittest.

---

### Task 1: Specify and implement the fast structural adapter

**Files:**
- Create: `src/gnn_attack_detector/attacks/fast_structural_adapter.py`
- Create: `tests/test_fast_structural_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

Create tests that build small bidirectional graphs and require both public functions to be deterministic, preserve `x` and `y`, return no self-loops or duplicate undirected edges, and make the exact requested number of edge changes. Include an attribute-free-style graph using identity features to prove the implementation never reads feature semantics.

```python
first = run_bridge_add(clean, budget=3, seed=17)
second = run_bridge_add(clean, budget=3, seed=17)
assert torch.equal(first.edge_index, second.edge_index)
assert len(changed_edges(clean, first)) == 3
assert torch.equal(first.x, clean.x)
assert torch.equal(first.y, clean.y)

rewired = run_hub_rewire(clean, budget=4, seed=19)
assert len(changed_edges(clean, rewired)) == 4
assert abs(edge_count(rewired) - edge_count(clean)) <= 1
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run:

```powershell
python -m pytest tests/test_fast_structural_adapter.py -q
```

Expected: collection fails because `fast_structural_adapter` does not exist.

- [ ] **Step 3: Implement the minimum adapter**

Implement local helpers that convert bidirectional `edge_index` to an undirected edge set, build neighbor sets and degrees, sample unique clean non-edges with `random.Random(seed)`, rank candidates by fewest shared neighbors then degree contrast, and convert the result back to sorted bidirectional `edge_index`.

```python
def run_bridge_add(graph, *, budget: int, seed: int):
    clean = _edges(graph)
    additions = _ranked_non_edges(clean, graph.num_nodes, budget, seed)
    return _with_edges(graph, clean | set(additions))

def run_hub_rewire(graph, *, budget: int, seed: int):
    clean = _edges(graph)
    remove_count = budget // 2
    removals = _rank_hub_edges(clean)[:remove_count]
    additions = _ranked_non_edges(
        clean, graph.num_nodes, budget - remove_count, seed
    )
    return _with_edges(graph, (clean - set(removals)) | set(additions))
```

Candidate generation must be bounded by `max(10_000, budget * 200)` attempts and raise `RuntimeError` when it cannot satisfy the exact budget. Additions must be non-edges of the original clean graph so removed edges are never re-added.

- [ ] **Step 4: Run adapter tests**

Run:

```powershell
python -m pytest tests/test_fast_structural_adapter.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the adapter**

```powershell
git add src/gnn_attack_detector/attacks/fast_structural_adapter.py tests/test_fast_structural_adapter.py
git commit -m "feat: add fast structural attack adapters"
```

### Task 2: Register and dispatch both protocols

**Files:**
- Modify: `src/gnn_attack_detector/attacks/registry.py`
- Modify: `src/gnn_attack_detector/graph_attacks.py`
- Modify: `tests/test_attack_registry.py`
- Modify: `tests/test_attack_dispatch_current.py`

- [ ] **Step 1: Extend registry and dispatch tests first**

Append both identifiers to `EXPECTED_ATTACKS`, change exact registry counts from 21 to 23, assert backend `fast_structural`, and require global dispatch to preserve operations `add` and `flip`.

```python
assert attack_spec("bridge_add").backend == "fast_structural"
assert attack_spec("hub_rewire").backend == "fast_structural"
assert attack_spec("bridge_add").operation == "add"
assert attack_spec("hub_rewire").operation == "flip"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
python -m pytest tests/test_attack_registry.py tests/test_attack_dispatch_current.py -q
```

Expected: failures because the new protocols are not registered or dispatched.

- [ ] **Step 3: Append registry entries without shifting old attack indices**

Append after `igattack`:

```python
_global("bridge_add", "Bridge Add", "bridge_add", "add", backend="fast_structural"),
_global("hub_rewire", "Hub Rewire", "hub_rewire", "flip", backend="fast_structural"),
```

Do not insert them among existing entries; corpus seeds derive from registry position.

- [ ] **Step 4: Wire the existing global dispatcher**

Import the two adapter functions and add two `_global_adapter` branches. Ignore surrogate and device because these attacks are CPU graph operations.

```python
if spec.algorithm == "bridge_add":
    return run_bridge_add(graph, budget=common["budget"], seed=common["seed"])
if spec.algorithm == "hub_rewire":
    return run_hub_rewire(graph, budget=common["budget"], seed=common["seed"])
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_fast_structural_adapter.py tests/test_attack_registry.py tests/test_attack_dispatch_current.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit registry and dispatch**

```powershell
git add src/gnn_attack_detector/attacks/registry.py src/gnn_attack_detector/graph_attacks.py tests/test_attack_registry.py tests/test_attack_dispatch_current.py
git commit -m "feat: expose bridge and hub attacks"
```

### Task 3: Expose configuration and documentation

**Files:**
- Modify: `configs/base.yaml`
- Modify: `configs/run.yaml`
- Modify: `src/gnn_attack_detector/config.py`
- Modify: `tests/test_corpus_policy_current.py`
- Modify: `README.md`

- [ ] **Step 1: Write configuration expectations**

Require `SUPPORTED_ATTACKS` to contain 23 entries and the loaded base protocol list to equal the registry. Keep existing protocol order and append the two new entries.

- [ ] **Step 2: Run configuration tests and verify failure**

Run:

```powershell
python -m pytest tests/test_configuration.py tests/test_corpus_policy_current.py -q
```

Expected: failures reporting the old 21-entry contract.

- [ ] **Step 3: Update user-facing configuration**

Append to `configs/base.yaml`:

```yaml
    - bridge_add
    - hub_rewire
```

Append commented choices to `configs/run.yaml` so the current active selection is not changed:

```yaml
    # - bridge_add
    # - hub_rewire
```

Change the configuration error to derive the count dynamically:

```python
raise ValueError(
    f"attacks.protocols must match the fixed {len(SUPPORTED_ATTACKS)}-protocol registry"
)
```

Update README from 21 to 23 protocols and describe the two fast local structural baselines.

- [ ] **Step 4: Run configuration and full tests**

Run:

```powershell
python -m pytest tests/test_configuration.py tests/test_corpus_policy_current.py -q
python -m pytest -q
```

Expected: focused tests and the complete suite pass.

- [ ] **Step 5: Verify source and configuration consistency**

Run:

```powershell
python -m gnn_attack_detector.cli --config configs/base.yaml show-selection --run-config configs/run.yaml
git diff --check
```

Expected: selection preview succeeds, the fixed registry has 23 protocols, and no whitespace errors are reported.

- [ ] **Step 6: Commit integration changes**

```powershell
git add configs/base.yaml configs/run.yaml src/gnn_attack_detector/config.py tests/test_corpus_policy_current.py README.md
git commit -m "docs: expose fast attacks in configuration"
```

## Self-review

- The plan covers both attack algorithms, exact budgets, determinism, all-profile compatibility, failure behavior, registry/config/platform exposure, artifact reuse, tests, and documentation.
- Existing attack order remains unchanged, so existing deterministic seed namespaces are stable.
- No external dependency, GPU branch, surrogate model, eigendecomposition, or unrelated detector change is introduced.
- The plan contains no deferred placeholders; accuracy improvement remains an evaluation condition rather than an unsupported implementation guarantee.
