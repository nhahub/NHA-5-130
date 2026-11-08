# -*- coding: utf-8 -*-
"""Standalone CLI Predictor for Aerial Change Detection.

Allows running inference on arbitrary satellite image pairs without launching
the web server. Supports Ensemble blending, TTA, and morphological post-processing.

Usage:
    python predict.py --before A.png --after B.png --output change_mask.png --overlay overlay.png
    python predict.py --before A.png --after B.png --tta --postprocess
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from scipy.ndimage import binary_opening, binary_closing, label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.model import SiameseUNet

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(fusion: str = "diff_cat") -> SiameseUNet:
    """Load trained checkpoint for the specified fusion mode."""
    candidates = [
        BASE_DIR / f"models/best_{'diffcat' if fusion == 'diff_cat' else 'concat'}_model.pt",
        BASE_DIR / (f"outputs/run_20261012_141834/best_model_inference.pt" if fusion == "diff_cat"
                    else "outputs/run_20261011_073203/best_model.pt"),
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if not model_path:
        raise FileNotFoundError(f"Could not find checkpoint for fusion={fusion}")

    ck = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = SiameseUNet(in_ch=3, base=32, fusion=fusion).to(DEVICE).eval()
    model.load_state_dict(ck["model"])
    return model


def tta_forward(model: SiameseUNet, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """4-fold Test-Time Augmentation."""
    p0 = torch.sigmoid(model(a, b))
    p1 = torch.flip(torch.sigmoid(model(torch.flip(a, [-1]), torch.flip(b, [-1]))), [-1])
    p2 = torch.flip(torch.sigmoid(model(torch.flip(a, [-2]), torch.flip(b, [-2]))), [-2])
    p3 = torch.flip(torch.sigmoid(model(torch.flip(a, [-2, -1]), torch.flip(b, [-2, -1]))), [-2, -1])
    return (p0 + p1 + p2 + p3) / 4.0


def postprocess_mask(mask: np.ndarray, min_size: int = 25) -> np.ndarray:
    """Apply morphological opening/closing and filter micro-noise components."""
    if not HAS_SCIPY:
        return mask
    # 1. Opening: remove isolated pixel noise
    opened = binary_opening(mask > 0, structure=np.ones((3, 3)))
    # 2. Closing: fill small holes in detected roofs
    closed = binary_closing(opened, structure=np.ones((3, 3)))
    # 3. Filter components smaller than min_size
    labeled, num_features = label(closed)
    cleaned = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_features + 1):
        comp = labeled == i
        if comp.sum() >= min_size:
            cleaned |= comp
    return (cleaned.astype(np.uint8)) * 255


def predict_pair(img_a: Image.Image, img_b: Image.Image,
                 models: dict, model_name: str = "ensemble",
                 threshold: float = 0.50, use_tta: bool = True,
                 use_postprocess: bool = True,
                 tile_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled inference with blending on arbitrary resolution images."""
    w, h = img_a.size
    if img_b.size != (w, h):
        img_b = img_b.resize((w, h), Image.BILINEAR)

    arr_a = (np.asarray(img_a.convert("RGB"), dtype=np.float32) / 255.0 - 0.5) / 0.5
    arr_b = (np.asarray(img_b.convert("RGB"), dtype=np.float32) / 255.0 - 0.5) / 0.5

    prob_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    step = tile_size
    y_coords = list(range(0, h, step))
    x_coords = list(range(0, w, step))

    tiles_a, tiles_b, slices = [], [], []
    for y in y_coords:
        for x in x_coords:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            th, tw = y_end - y, x_end - x

            ta = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            tb = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            ta[:th, :tw] = arr_a[y:y_end, x:x_end]
            tb[:th, :tw] = arr_b[y:y_end, x:x_end]

            tiles_a.append(torch.from_numpy(ta).permute(2, 0, 1).unsqueeze(0))
            tiles_b.append(torch.from_numpy(tb).permute(2, 0, 1).unsqueeze(0))
            slices.append((y, y_end, x, x_end, th, tw))

    batch_size = 16
    n_tiles = len(tiles_a)
    with torch.no_grad():
        for i in range(0, n_tiles, batch_size):
            b_a = torch.cat(tiles_a[i : i + batch_size], dim=0).to(DEVICE)
            b_b = torch.cat(tiles_b[i : i + batch_size], dim=0).to(DEVICE)

            if model_name == "ensemble":
                m1 = models.get("concat")
                m2 = models.get("diff_cat")
                p1 = tta_forward(m1, b_a, b_b) if use_tta else torch.sigmoid(m1(b_a, b_b))
                p2 = tta_forward(m2, b_a, b_b) if use_tta else torch.sigmoid(m2(b_a, b_b))
                p = 0.45 * p1 + 0.55 * p2
            elif model_name == "diff_cat":
                m = models["diff_cat"]
                p = tta_forward(m, b_a, b_b) if use_tta else torch.sigmoid(m(b_a, b_b))
            else:
                m = models["concat"]
                p = tta_forward(m, b_a, b_b) if use_tta else torch.sigmoid(m(b_a, b_b))

            p_np = p.squeeze(1).cpu().numpy()
            for k in range(p_np.shape[0]):
                y1, y2, x1, x2, th, tw = slices[i + k]
                prob_map[y1:y2, x1:x2] += p_np[k, :th, :tw]
                weight_map[y1:y2, x1:x2] += 1.0

    weight_map = np.maximum(weight_map, 1.0)
    prob_map = prob_map / weight_map
    mask = (prob_map >= threshold).astype(np.uint8) * 255

    if use_postprocess:
        mask = postprocess_mask(mask)

    return prob_map, mask


def make_overlay(img_b: Image.Image, mask: np.ndarray, color=(0, 230, 255), alpha=0.55) -> Image.Image:
    """Create glow overlay of change mask on Image B."""
    b_rgb = img_b.convert("RGB")
    overlay = np.array(b_rgb, dtype=np.float32)
    change_pixels = mask > 127
    r, g, b = color
    overlay[change_pixels, 0] = overlay[change_pixels, 0] * (1.0 - alpha) + r * alpha
    overlay[change_pixels, 1] = overlay[change_pixels, 1] * (1.0 - alpha) + g * alpha
    overlay[change_pixels, 2] = overlay[change_pixels, 2] * (1.0 - alpha) + b * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Aerial Image Change Detection CLI")
    parser.add_argument("--before", "-a", type=str, required=True, help="Path to Before image (T1)")
    parser.add_argument("--after", "-b", type=str, required=True, help="Path to After image (T2)")
    parser.add_argument("--output", "-o", type=str, default="change_mask.png", help="Path to save binary mask")
    parser.add_argument("--overlay", type=str, default=None, help="Path to save visual overlay")
    parser.add_argument("--model", type=str, default="ensemble", choices=["ensemble", "diff_cat", "concat"])
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification probability threshold")
    parser.add_argument("--tta", action="store_true", default=True, help="Use 4-fold Test-Time Augmentation")
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA for faster execution")
    parser.add_argument("--postprocess", action="store_true", default=True, help="Apply morphological noise filtering")
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false")
    args = parser.parse_args()

    path_a = Path(args.before)
    path_b = Path(args.after)
    if not path_a.exists():
        sys.exit(f"Error: Before image '{path_a}' does not exist.")
    if not path_b.exists():
        sys.exit(f"Error: After image '{path_b}' does not exist.")

    print(f"[*] Loading models on {DEVICE}...")
    models = {}
    if args.model in ("ensemble", "concat"):
        models["concat"] = load_model("concat")
        print("  [+] Loaded Baseline Concat model")
    if args.model in ("ensemble", "diff_cat"):
        models["diff_cat"] = load_model("diff_cat")
        print("  [+] Loaded Diff-Cat model")

    img_a = Image.open(path_a)
    img_b = Image.open(path_b)
    print(f"[*] Processing image pair: {img_a.size[0]}x{img_a.size[1]} pixels")

    t0 = time.time()
    prob_map, mask = predict_pair(
        img_a, img_b, models=models, model_name=args.model,
        threshold=args.threshold, use_tta=args.tta, use_postprocess=args.postprocess
    )
    elapsed = time.time() - t0

    # Save output mask
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)
    print(f"  [+] Saved change mask to: {out_path}")

    # Save overlay if requested
    if args.overlay:
        ov_path = Path(args.overlay)
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_img = make_overlay(img_b, mask)
        overlay_img.save(ov_path)
        print(f"  [+] Saved visual overlay to: {ov_path}")

    # Compute statistics
    change_pixels = int((mask > 0).sum())
    total_pixels = mask.size
    change_pct = (change_pixels / total_pixels) * 100
    est_area_m2 = change_pixels * 0.25  # LEVIR-CD+ 0.5m GSD

    print("\n" + "=" * 50)
    print("         INFERENCE SUMMARY")
    print("=" * 50)
    print(f"  Model Engine       : {args.model.upper()}")
    print(f"  TTA Enabled        : {args.tta}")
    print(f"  Post-processing    : {args.postprocess}")
    print(f"  Threshold          : {args.threshold:.2f}")
    print(f"  Inference Latency  : {elapsed:.2f} s")
    print(f"  Changed Pixels     : {change_pixels:,} ({change_pct:.2f}%)")
    print(f"  Est. Ground Area   : {est_area_m2:,.1f} m^2 ({est_area_m2 / 10000:.3f} ha)")
    print("=" * 50)


if __name__ == "__main__":
    main()
