# Reference-free shared GCN implementation plan

**Goal:** Replace reference/probe-difference detection with a shared 12→32→32 GCN, 96-dimensional pooled embedding plus four consistency statistics, while retaining selected-profile workflows and separately versioned per-profile metrics.

**Architecture:** Attribute/structure preprocessing is independent of labels and references. Disk-backed prepared graph inputs feed one jointly trained encoder/head. Legacy compact attacks may hydrate missing attributes from their verified source solely during decoding; standalone uploads do not require a reference. Keep defense benchmarks and attack-surrogate code because they are still used outside detection.

**Tech stack:** existing PyTorch, PyG, NumPy, SciPy, pytest, Streamlit. No new runtime dependencies.

## Execution (same session, user has approved implementation)
- [x] Tests: label/reference independence; variable dimensions and node permutation; invalid edges and absent/identity attributes; encoder gradients; save/load; CPU/CUDA fallback; selective training/report preservation; standalone and legacy uploads.
- [x] Replace detector_features.py: strict graph validation, 12 node descriptors, four edge-consistency summaries, chunked edge-wise attribute comparisons, no NxN matrices.
- [x] Replace detector_dataset.py: portable disk-backed prepared inputs, metadata separated from model input; optional nonattack manifest; report repeated clean controls honestly, never manufacture benign labels.
- [x] Replace detector_model.py: shared GCN pooling, joint training, training-only normalization, profile/class balance, validation-only threshold, versioned bundles and checks.
- [x] Adapt pipeline.py/workflow.py/cli.py: cache re-extraction without attack generation, selected-only training/evaluation, candidate models and historical reports, explicit activation, no probe/reference in standalone detection.
- [x] Adapt graph_io.py: optional labels, safe PT loader and self-contained exports; legacy compact hydration validates source fingerprints; CSR-only upload requires attributes rather than silently treating it as a full graph.
- [x] Adapt preparation/platform: attacks need only their GCN surrogate; robust defense page remains separate. Upload no longer selects a dataset; provide attribute semantics and unknown-source support messaging.
- [x] Remove replaced detector-difference code and obsolete detector tests only; preserve a prechange snapshot at output/shared-gcn-prechange.
- [x] Run full pytest, tiny end-to-end training/detection and CLI/platform checks; write server migration instructions with exact changed/deleted files. No real datasets regenerated, no CUDA performance claim on this CPU-only machine.

## Acceptance boundaries
Only benchmark clean controls currently exist unless a nonattack manifest is provided. Reused clean controls must be flagged, not counted as independent negatives. Metadata/labels never enter the graph encoder. Future datasets require evaluation; readable input does not imply validated generalization. No automatic deletion of attacks, models, reports, or the user's unrelated changes.

## Verification (2026-09-15)
- Full regression: 142 passed, 3 skipped (CUDA hardware unavailable locally).
- Real tiny-graph joint training, save/load/activation, independent NPZ inference, selected-only training and profile-history retention exercised.
- Simulated CUDA failure retries on CPU; actual CUDA smoke tests remain for the server.
- Independent code review findings fixed and rechecked: restricted upload loading, canonical split checks, explicit nonattack provenance, saved preprocessing policies, independent-only formal metrics, permutation-matrix masking and NPZ expansion/header limits.
- CLI configuration/help, Python compilation and PowerShell syntax checks passed; git diff --check passed (line-ending warnings only).
- Runtime dependencies, existing attacks and unrelated preexisting workspace changes preserved. No real-dataset performance claim; server training and independent negative collection remain deployment/evaluation work.
