# Automatic CPU/GPU Fallback Design

## Goal

Use a CUDA GPU wherever the project stage and underlying attack implementation support it, while continuing to run on CPU-only machines. If a supported GPU operation fails because CUDA is unavailable, incompatible, or out of memory, retry only the current unfinished unit on CPU and preserve all completed artifacts.

## Device policy

The existing `runtime.device` setting remains the single user-facing control:

- `auto` selects CUDA when `torch.cuda.is_available()` is true and otherwise selects CPU.
- `cpu` forces CPU for every stage.
- `cuda` requests CUDA, but unsupported attack implementations and recognized CUDA runtime failures still fall back to CPU so a batch can finish.

A shared resolver will implement this policy. Components must report the device they actually use rather than merely the requested value.

## Execution boundaries

Victim-model training, compatible PyTorch attack implementations, and neural detector training may use CUDA. Operations backed by NumPy, SciPy, CPU-only spectral routines, or third-party implementations that cannot safely accept CUDA tensors remain on CPU.

Attack capability is explicit rather than inferred by blindly replacing `.cpu()` calls. Saved graphs, manifests, metrics, and serialized model state remain CPU tensors for portability.

## Fallback behavior

Each graph or training unit starts on its resolved device. A recognized CUDA availability, compatibility, or out-of-memory failure triggers these actions:

1. discard the incomplete in-memory result;
2. clear the CUDA cache when possible;
3. print a `[GPU-FALLBACK]` message containing the stage and reason;
4. retry that unit once from its original clean input on CPU.

Previously completed and saved graphs are not regenerated. Ordinary Python errors, invalid inputs, and algorithm errors are not treated as CUDA failures and continue to raise normally. A failed CPU retry also raises normally.

## Reporting

Long-running stages print `[DEVICE]` with the selected device. CPU-only attacks say they are using CPU; fallback messages identify the affected dataset split, attack, and graph where available.

## Testing

Tests cover automatic CPU selection without CUDA, automatic CUDA selection when available, forced CPU, CPU-only attack routing, recognized CUDA failure followed by one CPU retry, preservation of non-CUDA exceptions, and detector/victim use of the shared policy. Existing corpus reuse and persistence tests must continue to pass.

## Deployment note

The source code supports both devices, but GPU execution still requires a CUDA-enabled PyTorch/PyG environment. The existing CPU setup script remains valid for CPU-only machines; a separate CUDA setup path must not overwrite it.
