# Skip Incomplete Attacks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip an entire dataset/attack when its selected corpus is absent or incomplete instead of aborting detector rebuilding.

**Architecture:** Keep `classify_attack_availability` as the single policy boundary. Return available attacks, unavailable attacks, and skip reasons; the existing pipeline filters records to available attacks only.

**Tech Stack:** Python 3.10, unittest/pytest.

---

### Task 1: Define skipping behavior

**Files:**
- Modify: `tests/test_corpus_policy_current.py`
- Modify: `src/gnn_attack_detector/corpus_policy.py`

- [ ] Replace the partial-corpus exception test with assertions that the attack is unavailable and its reason reports the missing quota.
- [ ] Run the focused test and verify it fails against the current exception behavior.
- [ ] Change the classifier to collect a reason and skip the whole attack on its first completeness mismatch.
- [ ] Run the focused test and verify it passes.

### Task 2: Expose skip reasons and verify regression safety

**Files:**
- Modify: `src/gnn_attack_detector/pipeline.py`
- Test: `tests/test_detector_core.py`

- [ ] Preserve skip reasons in the rebuild result and print one concise skip line per attack.
- [ ] Run corpus-policy and detector tests.
- [ ] Run the complete test suite and `git diff --check`.

