#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento simple y modular para segmentación 3D.
Compatible con:
 - PointNet
 - PointNet++
 - PointNet++ improved (SPFE+WSLFA)
 - DilatedToothSegNet
 - Transformer3D
 - ToothFormer

Versión limpia (v13) para Tesis Amaro.
"""

import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # /home/.../scripts_v5
sys.path.append(str(ROOT))                   # ahora Python ve /models


# ============================================================
# === IMPORTS DE MODELOS DISPONIBLES =========================
# ============================================================

from models.pointnet import PointNetSeg

MODEL_ZOO = {
    "pointnet": PointNetSeg
}


# ==========================================================
# DATASET SIMPLE
# ==========================================================

class CloudDataset(Dataset):
    def __init__(self, X_path, Y_path):
        X = np.load(X_path)["X"].astype(np.float32)
        Y = np.load(Y_path)["Y"].astype(np.int64)
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_loaders(data_dir, batch_size=6, workers=6):
    data_dir = Path(data_dir)
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CloudDataset(
            data_dir / f"X_{split}.npz",
            data_dir / f"Y_{split}.npz",
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=True
        )
    return loaders


# ==========================================================
# NORMALIZACIÓN
# ==========================================================

def normalize_cloud(x):
    """
    Normaliza XYZ por nube → centro=0, radio=1.
    """
    B, N, C = x.shape
    xyz = x[:, :, :3]
    feats = x[:, :, 3:] if C > 3 else None

    center = xyz.mean(dim=1, keepdim=True)
    xyz = xyz - center
    radius = xyz.norm(dim=-1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    xyz = xyz / (radius + 1e-6)

    if feats is not None:
        x = torch.cat([xyz, feats], dim=-1)
    else:
        x = xyz

    return x


# ==========================================================
# MÉTRICAS SIMPLES
# ==========================================================

@torch.no_grad()
def confusion_matrix(logits, y, num_classes):
    preds = logits.argmax(-1).reshape(-1)
    y = y.reshape(-1)

    mask = (y >= 0) & (y < num_classes)
    preds = preds[mask]
    y = y[mask]

    idx = y * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes)


def macro_from_cm(cm):
    tp = torch.diag(cm).float()
    gt = cm.sum(1).float().clamp_min(1)
    pd = cm.sum(0).float().clamp_min(1)

    acc = tp.sum() / cm.sum().clamp_min(1)
    prec = (tp / pd).mean()
    rec = (tp / gt).mean()
    f1 = (2 * prec * rec / (prec + rec + 1e-6))
    iou = (tp / (gt + pd - tp).clamp_min(1)).mean()

    return {
        "acc": acc.item(),
        "prec": prec.item(),
        "rec": rec.item(),
        "f1": f1.item(),
        "iou": iou.item()
    }


# ==========================================================
# LOSS CE + DICE
# ==========================================================

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        self.num_classes = num_classes

    def forward(self, logits, y):
        ce = self.ce(logits.transpose(1, 2), y)

        probs = F.softmax(logits, dim=-1)
        y_one = F.one_hot(y, self.num_classes).float()

        inter = (probs * y_one).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + y_one.sum(dim=(1, 2))
        dice = 1 - (2 * inter + 1e-5) / (union + 1e-5)
        dice = dice.mean()

        return ce + dice


# ==========================================================
# BUILDER DE MODELOS
# ==========================================================

def build_model(name, num_classes, in_ch, device):
    name = name.lower()

    if name == "pointnet":
        return PointNetSeg(num_classes, in_ch).to(device)

    elif name == "pointnetpp":
        return PointNet2Seg(num_classes, in_ch=in_ch).to(device)

    elif name == "pointnetpp_improved":
        return PointNet2Seg_SPFE_WSLFA(num_classes, in_ch=in_ch).to(device)

    elif name == "dilated":
        return DilatedToothSegNet(num_classes, in_ch=in_ch).to(device)

    elif name == "transformer3d":
        return Transformer3D(num_classes, in_ch=in_ch).to(device)

    elif name == "toothformer":
        return ToothFormer(num_classes, in_ch=in_ch).to(device)

    else:
        raise ValueError(f"Modelo desconocido: {name}")


# ==========================================================
# ENTRENAMIENTO
# ==========================================================

def train(model, loaders, device, num_classes, epochs, lr, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class weights automáticos
    Ytrain = loaders["train"].dataset.Y.numpy()
    freq = np.bincount(Ytrain.reshape(-1), minlength=num_classes)
    freq = np.maximum(freq, 1)
    w = 1 / np.log(1.2 + freq)
    w = w / w.mean()
    w = torch.tensor(w, dtype=torch.float32, device=device)

    loss_fn = CombinedLoss(num_classes, class_weights=w)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for ep in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0

        for X, Y in loaders["train"]:
            X = normalize_cloud(X.to(device))
            Y = Y.to(device)

            logits = model(X)
            loss = loss_fn(logits, Y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()

        # ---------------- VAL ----------------
        model.eval()
        val_loss = 0
        cm = torch.zeros((num_classes, num_classes), device=device)

        for X, Y in loaders["val"]:
            X = normalize_cloud(X.to(device))
            Y = Y.to(device)

            logits = model(X)
            loss = loss_fn(logits, Y)
            val_loss += loss.item()

            cm += confusion_matrix(logits, Y, num_classes)

        stats = macro_from_cm(cm)

        # Guardar history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(stats["f1"])

        print(f"[{ep:03d}] tr={train_loss:.4f} va={val_loss:.4f} "
              f"acc={stats['acc']:.3f} f1={stats['f1']:.3f}")

        # Best model
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            torch.save(model.state_dict(), out_dir / "best.pt")

    # Final model
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    json.dump(history, open(out_dir / "history.json", "w"), indent=2)

    print(f"[DONE] Best epoch: {best_epoch}")
    return history


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    loaders = make_loaders(args.data_dir, batch_size=args.batch_size)
    X0, _ = next(iter(loaders["train"]))
    in_ch = X0.shape[2]

    Y = np.load(Path(args.data_dir) / "Y_train.npz")["Y"]
    num_classes = int(Y.max() + 1)

    model = build_model(args.model, num_classes, in_ch, device)

    out_dir = Path("runs_clean") / args.tag
    train(model, loaders, device, num_classes, args.epochs, args.lr, out_dir)
