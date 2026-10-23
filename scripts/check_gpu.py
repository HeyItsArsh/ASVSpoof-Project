"""Verify real deterministic MobileNet updates on CUDA; record resource usage."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch import nn

from secure_audio.common import read_config, save_json, seed_all
from secure_audio.models import build_mobile, optimization_step


def main() -> None:
    """Exercise the configured four-second batch through forward/backward/save."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    cfg = read_config(REPO / "configs/default.yaml")
    seed_all(cfg["seeds"][0], cfg["deep"]["threads"])
    device = torch.device("cuda")
    model = build_mobile().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["deep"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    frames = 1 + round(cfg["audio"]["seconds"] * cfg["audio"]["sample_rate"]) // cfg["features"]["hop_length"]
    shape = (cfg["deep"]["batch_size"], 1, cfg["features"]["n_mels"], frames)
    x = torch.randn(shape, device=device)
    y = torch.arange(shape[0], device=device).remainder(2).float()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimization_step(model, optimizer, criterion, x, y)
    torch.cuda.synchronize()
    started = time.perf_counter()
    losses = [optimization_step(model, optimizer, criterion, x, y) for _ in range(10)]
    torch.cuda.synchronize()
    step_seconds = (time.perf_counter() - started) / len(losses)
    model.eval()
    with torch.inference_mode():
        first = model(x)
        second = model(x)
    assert torch.isfinite(first).all() and np.isfinite(losses).all()
    assert torch.equal(first, second), "Repeated CUDA inference differed"
    checkpoint = REPO / "artifacts/gpu_check/model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored = build_mobile().to(device).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    with torch.inference_mode():
        assert torch.equal(first, restored(x)), "Reloaded CUDA checkpoint differed"
    result = {"passed": True, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
              "batch_shape": list(shape), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "finite_backward_losses": losses, "mean_update_seconds_synthetic": step_seconds,
              "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "total_vram_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
              "scope": "Synthetic CUDA mechanics/resource check; not ASVspoof metrics or an epoch estimate"}
    save_json(REPO / "artifacts/gpu_check/verification.json", result)
    print(f"CUDA verified: {result['device']} | {result['torch']} | shape={shape}", flush=True)
    print(f"Synthetic update: {step_seconds:.3f}s; peak allocated VRAM {result['peak_allocated_vram_mib']:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
