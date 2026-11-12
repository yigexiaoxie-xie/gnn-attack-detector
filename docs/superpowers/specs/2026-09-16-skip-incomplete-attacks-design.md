# Skip Incomplete Attacks Design

## Goal

Allow detector rebuilding to continue when a selected dataset has an absent or partially generated attack corpus.

## Behavior

- A dataset/attack is available only when every selected partition and configured strength has its full configured quota.
- An entirely absent or partially complete dataset/attack is unavailable and is skipped as a whole.
- Partial rows are never included in feature preparation or detector training.
- The result records unavailable attacks and a human-readable reason for each skipped attack.
- Rebuilding still fails when a selected dataset has no complete attacks at all.

## Scope

Change only corpus availability classification, pipeline reporting, and focused tests. Do not change generation quotas, existing artifacts, model architecture, balancing, checkpoint selection, or evaluation.

