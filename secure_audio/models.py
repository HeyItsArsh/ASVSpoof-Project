"""CPU-friendly models and reproducible DataLoaders."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.mobilenetv2 import MobileNetV2

from .audio import augment, fix_length, load_audio, mel_spectrogram
from .data import Record


class MonoMobileNet(MobileNetV2):
    """MobileNetV2 with equivalent spatial mean pooling for deterministic CUDA."""
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features, globally average, and produce one logit."""
        x = self.features(x)
        # Same result as AdaptiveAvgPool2d(1), with deterministic mean backward.
        x = x.mean(dim=(-2, -1))
        return self.classifier(x)


def build_mobile() -> nn.Module:
    """Standard width-1 MobileNetV2, one input channel, one binary logit.

    Train from scratch: no ImageNet weights or hidden network download. The
    architecture uses far fewer parameters than ResNet-18; benchmark runtime on
    the target laptop rather than assuming depthwise kernels are always faster.
    """
    model = MonoMobileNet(num_classes=1)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    return model


class SpectrogramDataset(Dataset):
    """Cached center-cropped evaluation; augmented waveform training when enabled."""
    def __init__(self, records: list[Record], cfg: dict, *, seed: int = 0,
                 training: bool = False, cached: np.ndarray | None = None) -> None:
        """Bind protocol records and optionally an aligned read-only cache."""
        self.records, self.cfg, self.seed = records, cfg, seed
        self.training, self.cached, self.epoch = training, cached, 0

    def __len__(self) -> int:
        """Return the exact protocol trial count."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an input and label, with deterministic per-epoch augmentation."""
        augmented = self.training and self.cfg["augmentation"]["enabled"]
        if self.cached is not None and not augmented:
            spec = np.array(self.cached[index], copy=True)
        else:
            wave = load_audio(self.records[index].path, self.cfg["audio"])
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            wave = fix_length(wave, self.cfg["audio"], rng if augmented else None)
            if augmented:
                wave = augment(wave, self.cfg["audio"]["sample_rate"], self.cfg["augmentation"], rng)
            spec = mel_spectrogram(wave, self.cfg)
        return torch.from_numpy(spec), torch.tensor(float(self.records[index].label))


def seed_worker(worker_id: int) -> None:
    """Seed loader workers from Torch's generator (Windows-spawn compatible)."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset: Dataset, cfg: dict, seed: int, *, shuffle: bool = False) -> DataLoader:
    """Construct a deterministic loader; default workers=0 is laptop friendly."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg["deep"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["deep"]["workers"], worker_init_fn=seed_worker,
                      generator=generator, drop_last=False, persistent_workers=False)


def resolve_device(cfg: dict) -> torch.device:
    """Select CUDA only when available and requested or automatic."""
    name = cfg["deep"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def optimization_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                      criterion: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Shared finite-loss update for main training and diagnostic controls."""
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x).flatten(), y)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss; inspect input scale and training configuration")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def predict_deep(model: nn.Module, batches: DataLoader, device: torch.device) -> np.ndarray:
    """Produce continuous bona fide logits in protocol order."""
    model.eval()
    chunks = [model(x.to(device)).flatten().cpu().numpy() for x, _ in batches]
    return np.concatenate(chunks)
