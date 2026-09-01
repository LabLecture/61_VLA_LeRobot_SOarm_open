"""Shared PyTorch compute-device selection helpers.

Automatic selection intentionally requires a supported GPU.  CPU execution is
available only when the caller explicitly requests it, which prevents a long
training job from silently falling back to the CPU.
"""

from __future__ import annotations

from typing import TypeAlias

import torch


DEVICE_CHOICES = ("auto", "xpu", "cuda", "cpu")
DeviceLike: TypeAlias = str | torch.device


def _normalize_device_type(device: DeviceLike) -> str:
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device).strip().lower()

    if device_type not in DEVICE_CHOICES:
        choices = ", ".join(DEVICE_CHOICES)
        raise ValueError(f"Unsupported device '{device}'. Choose one of: {choices}.")
    return device_type


def _backend(device_type: str):
    """Return a PyTorch accelerator namespace without assuming it exists."""
    return getattr(torch, device_type, None)


def _unavailable_message(device_type: str) -> str:
    if device_type == "cuda":
        return (
            "CUDA is not available. Install a CUDA-enabled PyTorch build and a "
            "compatible NVIDIA driver, then verify the environment with "
            "'python hello_lerobot.py --device cuda'. Use '--device xpu' for an "
            "Intel GPU or '--device cpu' only when CPU execution is intentional."
        )
    if device_type == "xpu":
        return (
            "Intel XPU is not available. Install an XPU-enabled PyTorch build and "
            "a compatible Intel graphics driver, then verify the environment with "
            "'python hello_lerobot.py --device xpu'. Use '--device cuda' for an "
            "NVIDIA GPU or '--device cpu' only when CPU execution is intentional."
        )
    return f"Device backend '{device_type}' is not available."


def backend_is_available(device_type: str) -> bool:
    """Return whether a PyTorch backend can currently be used.

    The implementation tolerates builds of PyTorch that do not expose
    ``torch.xpu`` or ``torch.cuda`` at all.
    """
    normalized = _normalize_device_type(device_type)
    if normalized == "auto":
        return backend_is_available("cuda") or backend_is_available("xpu")
    if normalized == "cpu":
        return True

    backend = _backend(normalized)
    is_available = getattr(backend, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except (AttributeError, RuntimeError):
        return False


def resolve_device_type(requested: str) -> str:
    """Resolve ``auto`` or validate an explicitly requested device type.

    ``auto`` prefers CUDA, then XPU.  It never silently selects CPU.
    """
    normalized = _normalize_device_type(requested)
    if normalized == "auto":
        for candidate in ("cuda", "xpu"):
            if backend_is_available(candidate):
                return candidate
        raise RuntimeError(
            "No supported GPU backend is available. Install a CUDA- or XPU-enabled "
            "PyTorch build and its matching GPU driver. If CPU execution is truly "
            "intended, request it explicitly with '--device cpu'."
        )

    if normalized == "cpu":
        return normalized
    if not backend_is_available(normalized):
        raise RuntimeError(_unavailable_message(normalized))
    return normalized


def resolve_torch_device(requested: str) -> torch.device:
    """Return a validated ``torch.device`` for a requested device type."""
    return torch.device(resolve_device_type(requested))


def get_device_count(device: DeviceLike) -> int:
    """Return the device count without failing when a backend namespace is absent."""
    device_type = _normalize_device_type(device)
    if device_type == "auto":
        device_type = resolve_device_type(device_type)
    if device_type == "cpu":
        return 1

    backend = _backend(device_type)
    device_count = getattr(backend, "device_count", None)
    if not callable(device_count):
        return 0
    try:
        return int(device_count())
    except (AttributeError, RuntimeError):
        return 0


def get_device_name(device: DeviceLike) -> str:
    """Return the selected processor name with actionable backend failures."""
    device_type = _normalize_device_type(device)
    if device_type == "auto":
        device_type = resolve_device_type(device_type)
        device = torch.device(device_type)
    if device_type == "cpu":
        return "CPU"
    if not backend_is_available(device_type):
        raise RuntimeError(_unavailable_message(device_type))

    backend = _backend(device_type)
    get_name = getattr(backend, "get_device_name", None)
    if not callable(get_name):
        raise RuntimeError(
            f"PyTorch backend '{device_type}' cannot report a device name. "
            "Check that the matching PyTorch build is installed correctly."
        )

    if isinstance(device, torch.device) and device.index is not None:
        index = device.index
    else:
        index = 0
    try:
        return str(get_name(index))
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to query the {device_type.upper()} device name: {exc}"
        ) from exc


def synchronize_device(device: DeviceLike) -> None:
    """Wait for queued work on a CUDA/XPU device; CPU is already synchronous."""
    device_type = _normalize_device_type(device)
    if device_type == "auto":
        device_type = resolve_device_type(device_type)
        device = torch.device(device_type)
    if device_type == "cpu":
        return
    if not backend_is_available(device_type):
        raise RuntimeError(_unavailable_message(device_type))

    backend = _backend(device_type)
    synchronize = getattr(backend, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError(
            f"PyTorch backend '{device_type}' does not provide synchronize(). "
            "Check that the matching PyTorch build is installed correctly."
        )
    try:
        synchronize(device)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to synchronize the {device_type.upper()} device: {exc}"
        ) from exc
