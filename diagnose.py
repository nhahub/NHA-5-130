# -*- coding: utf-8 -*-
"""Diagnose the model: compare model.eval() vs model.train()
(batch statistics) on validation tiles.

Usage:
    python diagnose.py [--checkpoint PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.data_discovery import build_samples, download_dataset
from src.dataset import EvalTileDataset, samples_for_split
from src.metrics import ConfusionMeter
from src.model import SiameseUNet


def find_best_checkpoint() -> Path:
    """Find the checkpoint with best val_iou across all runs."""
    runs = sorted(config.OUTPUT_DIR.glob("run_*"))
    best_path, best_val = None, -1.0
    for r in runs:
        ck = r / "best_model.pt"
        if not ck.exists():
            continue
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        v = float(meta.get("val_iou", -1.0))
        if v > best_val:
            best_path, best_val = ck, v
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Diagnose model behaviour")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: best across runs)")
    args = parser.parse_args()

    device = torch.device("cpu")  # diagnose on CPU for safety

    # find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_best_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        sys.exit("ERROR: no checkpoint found. Train first or pass --checkpoint.")
    print(f"Checkpoint: {ckpt_path}")

    # load data
    root = download_dataset()
    samples, _ = build_samples(root)
    val_s = samples_for_split(samples, "val")
    if not val_s:
        sys.exit("ERROR: empty validation split.")
    ds = EvalTileDataset(val_s, image_size=256)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    # load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    base = int(cfg.get("base-channels", cfg.get("base_channels", config.BASE_CHANNELS)))
    fusion = cfg.get("fusion", getattr(config, "FUSION_MODE", "concat"))
    model = SiameseUNet(in_ch=config.IN_CHANNELS, base=base, fusion=fusion)
    model.load_state_dict(ckpt["model"])

    batch = next(iter(dl))
    a, b, y = batch["a"], batch["b"], batch["mask"]
    print(f"input a mean/std: {float(a.mean()):.4f} / {float(a.std()):.4f}")
    print(f"mask changed frac in batch: {float(y.mean()):.4f}")

    for mode in ("eval", "train"):
        if mode == "eval":
            model.eval()
        else:
            model.train()
        with torch.no_grad():
            logits = model(a, b)
        probs = torch.sigmoid(logits)
        meter = ConfusionMeter()
        meter.update(logits, y, 0.5)
        m = meter.metrics()
        qs = [float(torch.quantile(probs, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)]
        print(f"[{mode:5}] prob quantiles 1/25/50/75/99%: "
              + " ".join(f"{q:.3f}" for q in qs)
              + f" | IoU={m['iou']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")
        # best threshold on these tiles (diagnostic only)
        best = (0.0, 0.0)
        for thr in np.arange(0.3, 0.95, 0.05):
            mm = ConfusionMeter()
            mm.update(logits, y, float(thr))
            if mm.metrics()["iou"] > best[1]:
                best = (float(thr), mm.metrics()["iou"])
        print(f"[{mode:5}] best threshold on this batch: {best[0]:.2f} -> IoU {best[1]:.3f}")


if __name__ == "__main__":
    main()
