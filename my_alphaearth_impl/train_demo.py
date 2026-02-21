"""Training entry for the custom AlphaEarth-style implementation.

The script is intentionally explicit and step-by-step for readability.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

# Make script runnable both as module and as direct file.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from my_alphaearth_impl.dataset import SyntheticEarthDataset
from my_alphaearth_impl.model import SimpleSTPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple AlphaEarth-style model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--time_steps", type=int, default=6)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="runs/my_alphaearth_impl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: Reproducibility + device setup.
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: Build dataset and dataloader.
    dataset = SyntheticEarthDataset(
        num_samples=args.num_samples,
        time_steps=args.time_steps,
        channels=args.channels,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Step 3: Build model, optimizer, and objective.
    model = SimpleSTPModel(in_channels=args.channels, embed_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Step 4: Train loop.
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0

        for x, target in loader:
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[Epoch {epoch}/{args.epochs}] reconstruction_loss={avg_loss:.6f}")

    # Step 5: Save checkpoint.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
