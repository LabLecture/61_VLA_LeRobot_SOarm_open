"""Verify that LeRobot and PyTorch can run a training step on a device.

This check does not connect to or move the robot.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import lerobot
import torch

from compute_device import (
    DEVICE_CHOICES,
    get_device_count,
    get_device_name,
    resolve_device_type,
    synchronize_device,
)


def require_xpu() -> torch.device:
    """Return an Intel GPU device (kept for backwards-compatible imports)."""
    return torch.device(resolve_device_type("xpu"))


def run_training_smoke_test(device: torch.device) -> float:
    """Run one forward/backward/optimizer step on the selected device."""
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())

    observation = torch.randn(4, 8, device=device)
    target = torch.randn(4, 2, device=device)
    loss = torch.nn.functional.mse_loss(model(observation), target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    synchronize_device(device)
    return loss.item()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a PyTorch CUDA, XPU, or explicitly selected CPU device."
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help=(
            "Compute device (default: auto). auto prefers CUDA, then XPU, and "
            "fails instead of silently falling back to CPU."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    selected_type = resolve_device_type(args.device)
    device = torch.device(selected_type)
    loss = run_training_smoke_test(device)

    print(f"Python executable : {sys.executable}")
    print(f"LeRobot version   : {lerobot.__version__}")
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA build        : {torch.version.cuda}")
    print(f"Requested device  : {args.device}")
    print(f"Selected device   : {device}")
    print(f"Device count      : {get_device_count(device)}")
    print(f"Device name       : {get_device_name(device)}")
    print(f"Training test loss: {loss:.6f}")
    print("Device smoke test : PASS")


if __name__ == "__main__":
    main()
