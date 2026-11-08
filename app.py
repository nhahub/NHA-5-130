# -*- coding: utf-8 -*-
"""Interactive Web Application for Aerial Image Change Detection.

DEPI Graduation Project - AI Track
Team Members:
  - Ahmed Refaat Abd-Elmotelb (Lead ML Engineer & Pipeline Architect)
  - Ahmed Reda Mohamed (Deep Learning Specialist & Loss Design)
  - Raeed Youssef Mahfouz (Data Systems & Augmentation Specialist)
  - Ahmed Mohamed Qotb (Evaluation Metrics & Quality Assurance)
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import binary_opening, binary_closing, label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Project directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.model import SiameseUNet

PORT = 5000
HOST = "0.0.0.0"

# Global device and models cache
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
MODELS: Dict[str, SiameseUNet] = {}
SAMPLES_CACHE: List[dict] = []

TEAM_MEMBERS = [
    {"name": "Ahmed Refaat Abd-Elmotelb", "role": "Lead ML Engineer & Pipeline Architect"},
    {"name": "Ahmed Reda Mohamed", "role": "Deep Learning Specialist & Loss Design"},
    {"name": "Raeed Youssef Mahfouz", "role": "Data Systems & Augmentation Specialist"},
    {"name": "Ahmed Mohamed Qotb", "role": "Evaluation Metrics & Quality Assurance"},
]


def load_cached_models():
    """Load trained models into memory."""
    global MODELS
    print(f"[*] Initializing models on {DEVICE} ({DEVICE_NAME})...")

    # 1. Baseline Concat Model
    m1_candidates = [
        BASE_DIR / "models/best_concat_model.pt",
        BASE_DIR / "outputs/run_20261011_073203/best_model.pt",
    ]
    for p in m1_candidates:
        if p.exists():
            try:
                ck1 = torch.load(p, map_location=DEVICE, weights_only=False)
                m1 = SiameseUNet(in_ch=3, base=32, fusion="concat").to(DEVICE).eval()
                m1.load_state_dict(ck1["model"])
                MODELS["concat"] = m1
                print(f"  [+] Loaded Baseline Concat Model from {p}")
                break
            except Exception as e:
                print(f"  [!] Failed to load model 1 from {p}: {e}")

    # 2. Diff-Cat Model
    m2_candidates = [
        BASE_DIR / "models/best_diffcat_model.pt",
        BASE_DIR / "outputs/run_20261012_141834/best_model_inference.pt",
        BASE_DIR / "outputs/run_20261012_141834/best_model.pt",
    ]
    for p in m2_candidates:
        if p.exists():
            try:
                ck2 = torch.load(p, map_location=DEVICE, weights_only=False)
                m2 = SiameseUNet(in_ch=3, base=32, fusion="diff_cat").to(DEVICE).eval()
                m2.load_state_dict(ck2["model"])
                MODELS["diff_cat"] = m2
                print(f"  [+] Loaded Diff-Enhanced U-Net (Diff-Cat) from {p}")
                break
            except Exception as e:
                print(f"  [!] Failed to load model 2 from {p}: {e}")


def init_dataset_samples():
    """Discover offline demo samples for instant zero-download startup."""
    global SAMPLES_CACHE
    SAMPLES_CACHE = []

    demo_dir = BASE_DIR / "demo_samples"
    if demo_dir.is_dir():
        for sub in sorted(demo_dir.iterdir()):
            if not sub.is_dir():
                continue
            a_p = sub / "A.png"
            b_p = sub / "B.png"
            mask_p = sub / "label.png"
            if not mask_p.exists():
                mask_p = sub / "mask.png"
            if a_p.exists() and b_p.exists():
                SAMPLES_CACHE.append({
                    "stem": sub.name,
                    "split": "demo",
                    "a_path": str(a_p),
                    "b_path": str(b_p),
                    "mask_path": str(mask_p) if mask_p.exists() else None,
                })
        if SAMPLES_CACHE:
            print(f"  [+] Loaded {len(SAMPLES_CACHE)} bundled demo scenes from {demo_dir}.")


def tta_forward(model: SiameseUNet, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """4-fold Test-Time Augmentation (Horizontal, Vertical, Dihedral)."""
    p0 = torch.sigmoid(model(a, b))
    p1 = torch.flip(torch.sigmoid(model(torch.flip(a, [-1]), torch.flip(b, [-1]))), [-1])
    p2 = torch.flip(torch.sigmoid(model(torch.flip(a, [-2]), torch.flip(b, [-2]))), [-2])
    p3 = torch.flip(torch.sigmoid(model(torch.flip(a, [-2, -1]), torch.flip(b, [-2, -1]))), [-2, -1])
    return (p0 + p1 + p2 + p3) / 4.0


def postprocess_mask(mask: np.ndarray, min_size: int = 25) -> np.ndarray:
    """Apply morphological opening/closing and filter micro-noise components."""
    if not HAS_SCIPY:
        return mask
    opened = binary_opening(mask > 0, structure=np.ones((3, 3)))
    closed = binary_closing(opened, structure=np.ones((3, 3)))
    labeled, num_features = label(closed)
    cleaned = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_features + 1):
        comp = labeled == i
        if comp.sum() >= min_size:
            cleaned |= comp
    return (cleaned.astype(np.uint8)) * 255


def predict_pair(
    img_a: Image.Image,
    img_b: Image.Image,
    model_name: str = "ensemble",
    threshold: float = 0.50,
    use_tta: bool = False,
    use_postprocess: bool = True,
    tile_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run fast tiled inference on arbitrary size bi-temporal pairs."""
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
            y1 = y
            y2 = min(y + tile_size, h)
            x1 = x
            x2 = min(x + tile_size, w)

            tile_a = arr_a[y1:y2, x1:x2, :]
            tile_b = arr_b[y1:y2, x1:x2, :]

            th, tw = y2 - y1, x2 - x1
            if th < tile_size or tw < tile_size:
                pad_h = tile_size - th
                pad_w = tile_size - tw
                tile_a = np.pad(tile_a, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                tile_b = np.pad(tile_b, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

            t_a = torch.from_numpy(tile_a).permute(2, 0, 1).unsqueeze(0)
            t_b = torch.from_numpy(tile_b).permute(2, 0, 1).unsqueeze(0)

            tiles_a.append(t_a)
            tiles_b.append(t_b)
            slices.append((y1, y2, x1, x2, th, tw))

    batch_size = 32
    n_tiles = len(tiles_a)
    with torch.inference_mode():
        for i in range(0, n_tiles, batch_size):
            b_a = torch.cat(tiles_a[i : i + batch_size], dim=0).to(DEVICE)
            b_b = torch.cat(tiles_b[i : i + batch_size], dim=0).to(DEVICE)

            if model_name == "ensemble":
                m1 = MODELS.get("concat")
                m2 = MODELS.get("diff_cat")
                if m1 and m2:
                    p1 = tta_forward(m1, b_a, b_b) if use_tta else torch.sigmoid(m1(b_a, b_b))
                    p2 = tta_forward(m2, b_a, b_b) if use_tta else torch.sigmoid(m2(b_a, b_b))
                    p = 0.45 * p1 + 0.55 * p2
                elif m2:
                    p = tta_forward(m2, b_a, b_b) if use_tta else torch.sigmoid(m2(b_a, b_b))
                elif m1:
                    p = tta_forward(m1, b_a, b_b) if use_tta else torch.sigmoid(m1(b_a, b_b))
                else:
                    raise RuntimeError("No model weights loaded in memory.")
            elif model_name == "diff_cat":
                m = MODELS.get("diff_cat")
                p = tta_forward(m, b_a, b_b) if use_tta else torch.sigmoid(m(b_a, b_b))
            else:
                m = MODELS.get("concat")
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
    """Create a semi-transparent cyan glow overlay on Image B."""
    b_rgb = img_b.convert("RGB")
    overlay = np.array(b_rgb, dtype=np.float32)
    change_pixels = mask > 127
    r, g, b = color
    overlay[change_pixels, 0] = overlay[change_pixels, 0] * (1.0 - alpha) + r * alpha
    overlay[change_pixels, 1] = overlay[change_pixels, 1] * (1.0 - alpha) + g * alpha
    overlay[change_pixels, 2] = overlay[change_pixels, 2] * (1.0 - alpha) + b * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def make_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Image.Image:
    """Create error analysis map: TP=Green, FP=Red, FN=Blue, TN=Dark."""
    h, w = gt_mask.shape
    err = np.zeros((h, w, 3), dtype=np.uint8)
    gt_pos = gt_mask > 127
    pred_pos = pred_mask > 127

    tp = gt_pos & pred_pos
    fp = (~gt_pos) & pred_pos
    fn = gt_pos & (~pred_pos)
    tn = (~gt_pos) & (~pred_pos)

    err[tn] = [18, 24, 38]     # Navy dark
    err[tp] = [0, 230, 118]    # Neon Green
    err[fp] = [255, 23, 68]    # Crimson Red
    err[fn] = [41, 121, 255]   # Electric Blue
    return Image.fromarray(err)


def pil_to_base64_png(img: Image.Image) -> str:
    """Encode PIL Image to base64 PNG data URI string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for fast concurrent requests."""
    daemon_threads = True


class ChangeDetectionHandler(BaseHTTPRequestHandler):
    """Clean HTTP request dispatcher and API handler."""

    def log_message(self, format, *args):
        try:
            code = args[0] if (args and isinstance(args[0], int)) else (args[1] if len(args) > 1 else 200)
            req = getattr(self, "requestline", "") or self.path
            print(f"  [HTTP {code}] {req}")
        except Exception:
            pass

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. Favicon (204 No Content)
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        # 2. Main Page (Index)
        elif path == "/" or path == "/index.html":
            html_file = BASE_DIR / "static/index.html"
            if html_file.exists():
                content = html_file.read_text(encoding="utf-8")
                rendered = content.replace("__DEVICE_NAME__", DEVICE_NAME)
                body = rendered.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            else:
                self.send_error(500, "static/index.html not found")
                return

        # 3. Static Assets (CSS, JS)
        elif path.startswith("/static/"):
            rel_file = path.replace("/static/", "")
            static_file = BASE_DIR / "static" / rel_file
            if static_file.exists() and static_file.is_file():
                mime, _ = mimetypes.guess_type(str(static_file))
                self.serve_file(static_file, mime or "application/octet-stream")
                return
            else:
                self.send_error(404, f"Static asset '{rel_file}' not found")
                return

        # 4. API: Status
        elif path == "/api/status":
            self.send_json({
                "status": "ready",
                "device": str(DEVICE),
                "device_name": DEVICE_NAME,
                "models_loaded": list(MODELS.keys()),
                "samples_count": len(SAMPLES_CACHE),
                "team": TEAM_MEMBERS,
            })
            return

        # 5. API: Samples List
        elif path == "/api/samples":
            sample_list = []
            for s in SAMPLES_CACHE:
                sample_list.append({"stem": s["stem"], "split": s.get("split", "demo")})
            self.send_json({"samples": sample_list})
            return

        # 6. API: Sample Image
        elif path == "/api/sample":
            stem = query.get("id", [None])[0]
            img_type = query.get("type", ["a"])[0]

            sample = next((s for s in SAMPLES_CACHE if s["stem"] == stem), None)
            if not sample:
                self.send_error(404, f"Sample '{stem}' not found")
                return

            path_key = "mask_path" if img_type == "mask" else f"{img_type}_path"
            img_path = sample.get(path_key)
            if not img_path or not Path(img_path).exists():
                if img_type == "mask":
                    # Return 1×1 blank transparent PNG
                    blank = bytes([
                        0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,
                        0x49,0x48,0x44,0x52,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,
                        0x08,0x00,0x00,0x00,0x00,0x3A,0x7E,0x9B,0x55,0x00,0x00,0x00,
                        0x0A,0x49,0x44,0x41,0x54,0x78,0x9C,0x62,0x00,0x00,0x00,0x02,
                        0x00,0x01,0xE2,0x21,0xBC,0x33,0x00,0x00,0x00,0x00,0x49,0x45,
                        0x4E,0x44,0xAE,0x42,0x60,0x82
                    ])
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(blank)))
                    self.end_headers()
                    self.wfile.write(blank)
                    return
                self.send_error(404, f"Image not found for {img_type}")
                return

            self.serve_file(Path(img_path), "image/png")
            return

        # 7. Documentation Downloads
        elif path.startswith("/docs/"):
            filename = path.replace("/docs/", "")
            doc_file = BASE_DIR / "documentation" / filename
            if doc_file.exists() and doc_file.is_file():
                mime, _ = mimetypes.guess_type(str(doc_file))
                self.serve_file(doc_file, mime or "application/octet-stream")
                return
            else:
                self.send_error(404, f"Document '{filename}' not found")
                return

        # 8. Documentation Figures
        elif path.startswith("/figures/"):
            filename = path.replace("/figures/", "")
            fig_file = BASE_DIR / "documentation/figures" / filename
            if fig_file.exists() and fig_file.is_file():
                mime, _ = mimetypes.guess_type(str(fig_file))
                self.serve_file(fig_file, mime or "image/png")
                return
            else:
                self.send_error(404, f"Figure '{filename}' not found")
                return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            try:
                data = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self.send_json({"error": f"Invalid JSON payload: {e}"}, status=400)
                return

            model_name = data.get("model", "ensemble")
            threshold = float(data.get("threshold", 0.50))
            use_tta = bool(data.get("tta", False))
            use_postprocess = bool(data.get("postprocess", True))
            sample_id = data.get("sample_id")

            gt_mask_np = None

            try:
                t0 = time.time()
                if sample_id:
                    sample = next((s for s in SAMPLES_CACHE if s["stem"] == sample_id), None)
                    if not sample:
                        self.send_json({"error": f"Sample '{sample_id}' not found"}, status=404)
                        return
                    img_a = Image.open(sample["a_path"])
                    img_b = Image.open(sample["b_path"])
                    if "mask_path" in sample and sample["mask_path"] and Path(sample["mask_path"]).exists():
                        gt_raw = np.array(Image.open(sample["mask_path"]).convert("L"))
                        th = gt_raw.max() / 2.0 if gt_raw.max() > 0 else 0.5
                        gt_mask_np = (gt_raw > th).astype(np.uint8) * 255
                else:
                    raw_a = data.get("image_a", "").split(",")[-1]
                    raw_b = data.get("image_b", "").split(",")[-1]
                    if not raw_a or not raw_b:
                        self.send_json({"error": "Both images A and B are required"}, status=400)
                        return
                    img_a = Image.open(io.BytesIO(base64.b64decode(raw_a)))
                    img_b = Image.open(io.BytesIO(base64.b64decode(raw_b)))

                # Run inference
                prob_map, pred_mask = predict_pair(
                    img_a, img_b,
                    model_name=model_name,
                    threshold=threshold,
                    use_tta=use_tta,
                    use_postprocess=use_postprocess,
                )
                t1 = time.time()
                elapsed_ms = int((t1 - t0) * 1000)

                # Visual outputs
                overlay_img = make_overlay(img_b, pred_mask)
                mask_pil = Image.fromarray(pred_mask)

                changed_pixels = int((pred_mask > 127).sum())
                total_pixels = int(pred_mask.size)
                changed_percent = float((changed_pixels / total_pixels) * 100.0)
                est_ground_area_m2 = changed_pixels * 0.25
                est_ground_area_ha = est_ground_area_m2 / 10000.0

                cluster_count = 0
                avg_footprint_m2 = 0.0
                max_footprint_m2 = 0.0
                if HAS_SCIPY and changed_pixels > 0:
                    labeled, num_features = label(pred_mask > 127)
                    cluster_count = int(num_features)
                    if cluster_count > 0:
                        sizes = np.bincount(labeled.ravel())[1:]
                        avg_footprint_m2 = float(sizes.mean() * 0.25)
                        max_footprint_m2 = float(sizes.max() * 0.25)

                if changed_percent < 2.0:
                    density_label = "Sparse / Low Activity"
                    density_color = "#00E676"
                elif changed_percent < 10.0:
                    density_label = "Suburban Development"
                    density_color = "#00E5FF"
                else:
                    density_label = "High Urban Densification"
                    density_color = "#FF9100"

                resp = {
                    "overlay_png": pil_to_base64_png(overlay_img),
                    "mask_png": pil_to_base64_png(mask_pil),
                    "inference_time_ms": elapsed_ms,
                    "metrics": {
                        "changed_pixels": changed_pixels,
                        "total_pixels": total_pixels,
                        "changed_percent": changed_percent,
                    },
                    "geo_stats": {
                        "est_ground_area_m2": round(est_ground_area_m2, 1),
                        "est_ground_area_ha": round(est_ground_area_ha, 3),
                        "cluster_count": cluster_count,
                        "avg_footprint_m2": round(avg_footprint_m2, 1),
                        "max_footprint_m2": round(max_footprint_m2, 1),
                        "density_label": density_label,
                        "density_color": density_color,
                    }
                }

                if gt_mask_np is not None:
                    err_img = make_error_map(gt_mask_np, pred_mask)
                    resp["error_map_png"] = pil_to_base64_png(err_img)

                    p_pos = pred_mask > 127
                    g_pos = gt_mask_np > 127
                    tp = float((p_pos & g_pos).sum())
                    fp = float((p_pos & (~g_pos)).sum())
                    fn = float(((~p_pos) & g_pos).sum())

                    eps = 1e-7
                    prec = tp / max(tp + fp, eps)
                    rec = tp / max(tp + fn, eps)
                    f1 = 2 * prec * rec / max(prec + rec, eps)
                    iou = tp / max(tp + fp + fn, eps)

                    resp["metrics"]["precision"] = float(prec)
                    resp["metrics"]["recall"] = float(rec)
                    resp["metrics"]["f1"] = float(f1)
                    resp["metrics"]["iou"] = float(iou)

                self.send_json(resp)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json({"error": str(e)}, status=500)
            return

        self.send_error(404, "Not Found")

    def serve_file(self, filepath: Path, mime: str):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            if mime and mime.startswith("image/"):
                self.send_header("Cache-Control", "public, max-age=3600")
            else:
                self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")


def main():
    print("=" * 70)
    print("  AERIAL IMAGE CHANGE DETECTION STUDIO - DEPI GRADUATION PROJECT")
    print("  Team Members:")
    for m in TEAM_MEMBERS:
        print(f"   - {m['name']:<28} : {m['role']}")
    print("=" * 70)

    load_cached_models()
    init_dataset_samples()

    server = ThreadingHTTPServer((HOST, PORT), ChangeDetectionHandler)
    print(f"\n[+] Web Application successfully running at: http://localhost:{PORT}")
    print("[*] Press Ctrl+C in terminal to stop server.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server gracefully...")
        server.server_close()


if __name__ == "__main__":
    main()
