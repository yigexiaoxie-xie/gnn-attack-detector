import torch
from pathlib import Path

from gnn_attack_detector.device import (
    is_cuda_failure,
    resolve_device,
    run_with_cpu_fallback,
)


def test_resolve_device_auto_and_cuda_fall_back_without_cuda() -> None:
    unavailable = lambda: False

    assert resolve_device("auto", cuda_available=unavailable).type == "cpu"
    assert resolve_device("cuda", cuda_available=unavailable).type == "cpu"
    assert resolve_device("cpu", cuda_available=lambda: True).type == "cpu"
    assert resolve_device("auto", cuda_available=lambda: True).type == "cuda"


def test_only_cuda_failures_are_classified_for_fallback() -> None:
    assert is_cuda_failure(torch.cuda.OutOfMemoryError("out of memory"))
    assert is_cuda_failure(RuntimeError("CUDA driver initialization failed"))
    assert is_cuda_failure(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert not is_cuda_failure(RuntimeError("invalid graph shape"))
    assert not is_cuda_failure(ValueError("CUDA was misspelled in user input"))


def test_cuda_failure_retries_once_on_cpu() -> None:
    calls: list[str] = []
    messages: list[str] = []

    def operation(device: torch.device) -> str:
        calls.append(device.type)
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "done"

    result = run_with_cpu_fallback(
        operation,
        requested="auto",
        label="test stage",
        cuda_available=lambda: True,
        printer=messages.append,
    )

    assert result == "done"
    assert calls == ["cuda", "cpu"]
    assert messages[0] == "[DEVICE] test stage: cuda"
    assert messages[1].startswith("[GPU-FALLBACK] test stage:")


def test_non_cuda_failure_is_not_retried() -> None:
    calls: list[str] = []

    def operation(device: torch.device) -> None:
        calls.append(device.type)
        raise RuntimeError("invalid graph shape")

    try:
        run_with_cpu_fallback(
            operation,
            requested="auto",
            label="test stage",
            cuda_available=lambda: True,
            printer=lambda _message: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "invalid graph shape"
    else:
        raise AssertionError("ordinary errors must propagate")

    assert calls == ["cuda"]


def test_victim_training_uses_shared_cpu_fallback_policy() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/gnn_attack_detector/victim.py"
    ).read_text(encoding="utf-8")

    assert "run_with_cpu_fallback(" in source
    assert 'label="victim training"' in source
