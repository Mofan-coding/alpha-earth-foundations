"""Synthetic dataset for quick end-to-end verification.

You can replace this dataset with real satellite data while keeping
output tensor shapes unchanged.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticEarthDataset(Dataset):
    """Generate pseudo Earth-observation time series.

    Returns:
      x: [T, C, H, W]
      target: [C, H, W]  (here we use the last frame as reconstruction target)
    """

    def __init__(
        self,
        num_samples: int = 512,
        time_steps: int = 6,
        channels: int = 6,
        image_size: int = 64,
    ) -> None:
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.channels = channels
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Low-frequency baseline + random temporal variation.
        base = torch.randn(self.channels, self.image_size, self.image_size) * 0.25
        seq = []
        for t in range(self.time_steps):
            drift = (t / max(self.time_steps - 1, 1)) * 0.2
            noise = torch.randn_like(base) * 0.1
            frame = base + drift + noise
            seq.append(frame)

        x = torch.stack(seq, dim=0)
        target = x[-1].clone()  # example target design
        return x, target
