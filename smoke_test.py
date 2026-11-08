# -*- coding: utf-8 -*-
"""Quick end-to-end smoke test with synthetic data (no real dataset needed).

Builds a fake LEVIR-like tree (train/val/test each with A/, B/, label/),
then runs discovery -> datasets -> model -> loss -> metrics -> checkpoint
save/load, to catch pipeline bugs early.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
import src.data_discovery as dd
from src.dataset import EvalTileDataset, TrainChangeDataset
from src.losses import BCEDiceLoss
from src.metrics import ConfusionMeter
from src.model import SiameseUNet, count_parameters


def make_img(size=512, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    if seed % 3 != 0:  # some samples have change, some not
        mask[100:150, 200:300] = 255
        mask[400:430, 100:180] = 255
    return img, mask


def build_fake(root: Path):
    for split in ("train", "val", "test"):
        n = {"train": 4, "val": 2, "test": 2}[split]
        for sub in ("A", "B", "label"):
            (root / split / sub).mkdir(parents=True)
        for i in range(n):
            seed = hash((split, i)) % 1000
            img, mask = make_img(seed=seed)
            for sub in ("A", "B"):
                from PIL import Image
                Image.fromarray(img + (1 if sub == "B" else 0)).save(
                    root / split / sub / f"{split}_{i}.png")
            from PIL import Image
            Image.fromarray(mask).save(root / split / "label" / f"{split}_{i}.png")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="levir_smoke_"))
    build_fake(tmp)
    print(f"fake dataset at {tmp}")

    dd.OVERRIDES = {}  # force automatic detection
    samples, notes = dd.build_samples(tmp)
    print(f"discovered {len(samples)} samples, notes={notes}")
    assert len(samples) == 8, samples
    assert all(s.split in ("train", "val", "test") for s in samples)

    train_s = [s for s in samples if s.split == "train"]
    val_s = [s for s in samples if s.split == "val"]

    # training dataset path
    ds = TrainChangeDataset(train_s, image_size=256, augment=True)
    dl = DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(dl))
    assert batch["a"].shape == (2, 3, 256, 256), batch["a"].shape
    assert batch["mask"].shape == (2, 1, 256, 256)
    uniq = torch.unique(batch["mask"])
    assert set(uniq.tolist()) <= {0.0, 1.0}, f"mask not binary: {uniq}"
    print("train dataset OK", batch["a"].shape, batch["mask"].shape, uniq)

    # eval tiles path (512 -> 4 tiles per image)
    eds = EvalTileDataset(val_s, image_size=256)
    assert len(eds) == 8, len(eds)
    edl = DataLoader(eds, batch_size=3, num_workers=0)
    ebatch = next(iter(edl))
    m = ebatch["meta"]
    print("eval tiles OK:", ebatch["a"].shape, "valid_hw:", list(zip(
        m["valid_h"].tolist(), m["valid_w"].tolist())),
        "origins:", list(zip(m["top"].tolist(), m["left"].tolist())))

    # model forward/backward
    model = SiameseUNet(base=32)
    print(f"params: {count_parameters(model)/1e6:.2f}M")
    crit = BCEDiceLoss()
    logits = model(batch["a"], batch["b"])
    assert logits.shape == (2, 1, 256, 256)
    loss = crit(logits, batch["mask"])
    loss.backward()
    print("loss:", float(loss))

    # metrics
    meter = ConfusionMeter()
    meter.update(logits, batch["mask"], 0.5)
    print("metrics:", meter.summary())

    # eval forward with tile slicing
    model.eval()
    with torch.no_grad():
        elogits = model(ebatch["a"], ebatch["b"])
    for j in range(elogits.shape[0]):
        assert int(m["valid_h"][j]) == 256 and int(m["valid_w"][j]) == 256
    print("eval forward OK", elogits.shape)

    # test metric formulas accuracy
    m_test = ConfusionMeter()
    # 10 TP, 2 FP, 3 FN, 85 TN (total 100 pixels)
    # Target: 10 TP, 3 FN -> 13 changed; 85 TN, 2 FP -> 87 unchanged
    targets = torch.cat([torch.ones(10), torch.zeros(2), torch.ones(3), torch.zeros(85)])
    # Sigmoid(10.0) > 0.5 -> 1, Sigmoid(-10.0) < 0.5 -> 0
    logits_m = torch.cat([torch.full((12,), 10.0), torch.full((88,), -10.0)])
    m_test.update(logits_m, targets, threshold=0.5)
    res = m_test.metrics()
    assert res["tp"] == 10 and res["fp"] == 2 and res["fn"] == 3 and res["tn"] == 85
    # IoU change: 10 / (10 + 2 + 3) = 10 / 15 = 2/3
    assert abs(res["iou"] - (10.0 / 15.0)) < 1e-5
    # IoU unchanged: 85 / (85 + 2 + 3) = 85 / 90
    assert abs(res["iou_unchanged"] - (85.0 / 90.0)) < 1e-5, f"iou_unchanged incorrect: {res['iou_unchanged']}"
    print("metric formulas unit test PASSED (IoU change & unchanged verified)")

    # test split preservation when only train and test exist (LEVIR-CD case)
    tmp_splits = Path(tempfile.mkdtemp(prefix="levir_splits_"))
    for sp, count in (("train", 10), ("test", 6)):
        for sub in ("A", "B", "label"):
            (tmp_splits / sp / sub).mkdir(parents=True)
        for i in range(count):
            for sub in ("A", "B"):
                from PIL import Image
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(
                    tmp_splits / sp / sub / f"{sp}_{i}.png")
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(
                tmp_splits / sp / "label" / f"{sp}_{i}.png")
    split_samples, split_notes = dd.build_samples(tmp_splits)
    train_stems = {s.stem for s in split_samples if s.split == "train"}
    val_stems = {s.stem for s in split_samples if s.split == "val"}
    test_stems = {s.stem for s in split_samples if s.split == "test"}
    # All 6 test samples must stay strictly in test
    assert len(test_stems) == 6, f"Expected 6 test samples, got {len(test_stems)}"
    assert all(stem.startswith("test_") for stem in test_stems), "Test set polluted with non-test samples!"
    assert all(stem.startswith("train_") for stem in train_stems | val_stems), "Train/val polluted with test samples!"
    assert len(train_stems) > 0 and len(val_stems) > 0, "Val split was not created from train!"
    print(f"split preservation PASSED (train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)})")
    shutil.rmtree(tmp_splits)

    # checkpoint round trip with optimizer / scheduler state
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    scaler = torch.amp.GradScaler(enabled=False)
    ck = {
        "model": model.state_dict(),
        "epoch": 5,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "best_iou": 0.72,
        "config": {"base_channels": 32}
    }
    p = tmp / "best.pt"
    torch.save(ck, p)
    loaded = torch.load(p, weights_only=False)
    assert "optimizer" in loaded and "scheduler" in loaded and "scaler" in loaded
    assert loaded["epoch"] == 5 and loaded["best_iou"] == 0.72
    m2 = SiameseUNet(base=32)
    m2.load_state_dict(loaded["model"])
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.load_state_dict(loaded["optimizer"])
    print("checkpoint round-trip with optimizer/scheduler PASSED")

    shutil.rmtree(tmp)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY")


if __name__ == "__main__":
    sys.exit(main())
