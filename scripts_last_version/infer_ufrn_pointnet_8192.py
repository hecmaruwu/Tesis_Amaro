#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_cloud(x):
    c = x.mean(dim=1, keepdim=True)
    x = x - c
    r = (x.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    return x / (r + 1e-8)


class STN3d(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1, self.bn1 = nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)
        self.fc1, self.bn4 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc2, self.bn5 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(B, self.k, self.k)

        iden = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return x + iden


class PointNetSeg(nn.Module):
    """
    PointNet compatible con checkpoint entrenado:
      - stn.*
      - fbn1.*
      - fbn2.*
    """

    def __init__(self, num_classes=15, dropout=0.5):
        super().__init__()

        self.stn = STN3d(k=3)

        self.conv1, self.bn1 = nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64)
        self.conv2, self.bn2 = nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128)
        self.conv3, self.bn3 = nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024)

        self.fconv1, self.fbn1 = nn.Conv1d(1152, 512, 1), nn.BatchNorm1d(512)
        self.fconv2, self.fbn2 = nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)
        self.fconv3 = nn.Conv1d(256, num_classes, 1)

    def forward(self, xyz):
        B, P, _ = xyz.shape

        x = xyz.transpose(2, 1)  # (B,3,P)

        T = self.stn(x)
        x = torch.bmm(T, x)

        x1 = F.relu(self.bn1(self.conv1(x)))      # (B,64,P)
        x2 = F.relu(self.bn2(self.conv2(x1)))     # (B,128,P)
        x3 = F.relu(self.bn3(self.conv3(x2)))     # (B,1024,P)

        xg = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, P)
        x_cat = torch.cat([xg, x2], dim=1)        # (B,1152,P)

        x = F.relu(self.fbn1(self.fconv1(x_cat)))
        x = F.relu(self.fbn2(self.fconv2(x)))
        x = self.dropout(x)

        logits = self.fconv3(x).transpose(2, 1)   # (B,P,C)
        return logits


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("No se pudo extraer state_dict del checkpoint.")


def clean_state_dict(state):
    new = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        new[k] = v
    return new


def safe_class_counts(pred_np):
    pred_list = pred_np.tolist()
    counts = {}
    for p in pred_list:
        p = int(p)
        counts[p] = counts.get(p, 0) + 1
    return counts, pred_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=15)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d21_internal", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.x_npz)["X"].astype("float32")

    if X.ndim != 3:
        raise ValueError(f"X debe tener forma (B,P,3), pero viene con forma {X.shape}")
    if X.shape[-1] != 3:
        raise ValueError(f"Última dimensión debe ser 3, pero viene {X.shape}")

    x = torch.tensor(X, dtype=torch.float32, device=device)

    model = PointNetSeg(
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = extract_state_dict(ckpt)
    state = clean_state_dict(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    if len(missing) > 0 or len(unexpected) > 0:
        print("[WARN] Hay keys faltantes o inesperadas. Revise si el checkpoint corresponde exactamente a PointNet clásico.")

    model.eval()

    x = normalize_cloud(x)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

    pred_np = pred[0].detach().cpu().numpy().astype(np.int32)
    prob_np = prob[0].detach().cpu().numpy().astype(np.float32)

    np.save(out_dir / "pred_labels.npy", np.asarray(pred_np, dtype=np.int32))
    np.save(out_dir / "pred_probs.npy", np.asarray(prob_np, dtype=np.float32))

    class_counts, pred_list = safe_class_counts(pred_np)

    d21_count = int(class_counts.get(int(args.d21_internal), 0))
    n_points = int(len(pred_list))

    summary = {
        "x_npz": str(args.x_npz),
        "ckpt": str(args.ckpt),
        "device": str(device),
        "input_shape": [int(v) for v in X.shape],
        "num_points": n_points,
        "num_classes": int(args.num_classes),
        "d21_internal": int(args.d21_internal),
        "pred_class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "pred_d21_points": d21_count,
        "pred_d21_frac": float(d21_count / max(n_points, 1)),
        "missing_keys": [str(k) for k in missing],
        "unexpected_keys": [str(k) for k in unexpected],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Guardado en:", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()