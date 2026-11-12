"""Shared CPU/CUDA selection and safe fallback policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch


T = TypeVar("T")


def resolve_device(
    requested: str,
    *,
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
) -> torch.device:
    value = str(requested).lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if value == "cpu" or not cuda_available():
        return torch.device("cpu")
    return torch.device("cuda")


def is_cuda_failure(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("cuda", "cublas", "cudnn", "cusparse", "device-side assert")
    )


def run_with_cpu_fallback(
    operation: Callable[[torch.device], T],
    *,
    requested: str,
    label: str,
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
    printer: Callable[[str], None] = print,
    announce: bool = True,
) -> T:
    device = resolve_device(requested, cuda_available=cuda_available)
    if announce:
        printer(f"[DEVICE] {label}: {device}")
    try:
        return operation(device)
    except Exception as exc:
        if device.type != "cuda" or not is_cuda_failure(exc):
            raise
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        printer(
            f"[GPU-FALLBACK] {label}: {type(exc).__name__}: {exc}; "
            "retrying on cpu"
        )
        return operation(torch.device("cpu"))
