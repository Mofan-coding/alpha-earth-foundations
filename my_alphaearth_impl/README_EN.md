# My AlphaEarth-Style Implementation (from scratch)

This is **my own simplified implementation** inspired by AlphaEarth Foundations.
It is reorganized for readability and reproducibility, so it can be used as a practical starter for your own remote-sensing experiments.

> Goal: provide a version that is easy to read, easy to run, and easy to adapt to your own dataset.

## 1. What is included

- A simplified `STP` (Space-Time-Precision) encoder:
  - **Space**: spatial convolutional encoding for each time step.
  - **Time**: lightweight temporal aggregation (attention pooling).
  - **Precision**: residual-style convolutional refinement for local details.
- A simple decoder for reconstruction pretraining.
- A runnable training demo using synthetic EO-like tensors.

## 2. Project structure

```text
my_alphaearth_impl/
├── README.md          # Chinese guide
├── README_EN.md       # English guide (this file)
├── model.py           # Model definition with detailed comments
├── dataset.py         # Synthetic spatiotemporal dataset scaffold
└── train_demo.py      # Step-by-step training entry
```

## 3. Environment setup

Run from repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```

## 4. Run the training demo

```bash
python my_alphaearth_impl/train_demo.py \
  --epochs 3 \
  --batch_size 8 \
  --time_steps 6 \
  --image_size 64
```

You should see reconstruction loss printed per epoch.
A checkpoint is saved to `runs/my_alphaearth_impl/model.pt`.

## 5. How to replace with your own dataset

1. Open `dataset.py` and replace `SyntheticEarthDataset` with your real data loader.
2. Keep output shapes consistent:
   - `x`: `[T, C, H, W]` (multi-temporal, multi-channel imagery)
   - `target`: `[C, H, W]` (e.g., last frame or your supervision target)
3. In `train_demo.py`, replace `loss_fn` based on task type:
   - Reconstruction: `MSELoss`
   - Representation learning: contrastive losses
   - Supervised tasks: `CrossEntropyLoss`, `DiceLoss`, etc.
4. Adjust channels and embedding size in `SimpleSTPModel` according to your GPU memory and task complexity.

## 6. Suggested next steps

- Connect to real EO data (Sentinel-2 / Landsat / MODIS).
- Add cloud/no-data masking logic.
- Upgrade temporal modeling with Transformer blocks.
- Add downstream heads for classification / segmentation / regression.

---

If you want, I can also help convert this demo into a production-style training repo with:
- GeoTIFF dataloader,
- multi-sensor support,
- YAML config + logging + visualization pipeline.
