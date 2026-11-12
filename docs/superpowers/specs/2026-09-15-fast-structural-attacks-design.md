# Fast Structural Attacks Design

## Goal

Add two deterministic, dependency-free structural attack protocols that run on Cora, CiteSeer, PubMed, and Polblogs without model fitting, gradients, or spectral decomposition:

- `bridge_add`: add long-range bridge edges between structurally dissimilar regions.
- `hub_rewire`: remove hub-adjacent edges and replace them with distant non-edges while keeping the requested change count exact.

The attacks expand corpus coverage. They may improve detector generalization after retraining, but they do not replace the separate work needed to improve sparse feature-attack detection.

## Protocol contracts

Both protocols are global, structure-only attacks. Their strength unit remains `edge_rate`, and the existing `structural_budget` function determines the exact number of undirected edge changes. They must:

- preserve node count, node attributes, and node labels;
- return a simple undirected graph with no self-loops or duplicate edges;
- produce exactly the requested number of unique edge changes;
- be deterministic for a fixed seed;
- support all four configured dataset profiles;
- run on CPU without DeepRobust, GreatX, GPU, surrogate training, gradients, or eigendecomposition.

### Bridge Add

`bridge_add` samples non-edge candidates using the seeded generator and ranks them by structural separation. Candidates with no shared neighbors are preferred, followed by lower neighborhood overlap and larger degree contrast. It adds the highest-ranked unique non-edges until the budget is met. Candidate sampling is bounded so PubMed does not require an all-pairs scan.

### Hub Rewire

`hub_rewire` ranks existing edges by endpoint degree and removes hub-adjacent edges. It then adds seeded, structurally distant non-edges. Removal and addition counts are split to keep the total number of changed undirected edges equal to the requested budget; when the budget is odd, the extra change is an addition. The final edge count may differ by at most one.

## Integration

The protocols use a small local adapter and the existing attack dispatch path. They are registered in `ATTACK_SPECS`, listed in the base attack configuration, and automatically become available to corpus generation and the platform through the existing registry-derived lists.

Existing attack artifacts remain reusable. A selected workflow generates only missing `bridge_add` and `hub_rewire` train/validation/test artifacts, then retrains the shared detector when those partitions are selected.

## Failure handling

If the graph cannot provide enough valid unique candidates, the adapter raises a clear runtime error instead of silently returning a smaller perturbation. The existing corpus retry and failure-summary behavior remains unchanged.

## Verification

Tests cover registry metadata, fixed-seed determinism, exact budgets, graph invariants, unchanged attributes and labels, and successful dispatch on small attributed and attribute-free graphs. The existing test suite must continue to pass.

## Acceptance criteria

- Both attacks are selectable everywhere existing attacks are selectable.
- Both generate valid samples for all four profiles.
- Existing attacks and artifacts are unchanged.
- Runtime is bounded by sparse graph processing and sampled candidates, avoiding the hours-long behavior of gradient and meta attacks.
- Detector results are accepted only after comparison with the previous model; no accuracy improvement is guaranteed by adding attacks alone.
