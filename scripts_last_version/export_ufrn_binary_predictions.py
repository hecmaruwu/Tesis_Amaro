#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class UFRNTestDataset(Dataset):
    def __init__(self, data_dir):
        data_dir = Path(data_dir)

        self.X = np.load(
            data_dir / "X_test.npz",
            allow_pickle=True
        )["X"]

        self.Y = np.load(
            data_dir / "Y_test.npz",
            allow_pickle=True
        )["Y"]

        self.rows = []
        with open(data_dir / "index_test.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        if len(self.X) != len(self.rows):
            raise RuntimeError(
                f"X_test tiene {len(self.X)} muestras, pero index_test.csv tiene {len(self.rows)} filas."
            )

    def __len__(self):
        return int(len(self.X))

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32)
        y = np.asarray(self.Y[idx], dtype=np.int64)
        row = self.rows[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            row
        )


# ============================================================
# PointNet binario: MISMA arquitectura del fine-tuning
# ============================================================

class STN3d(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)
        x = x.view(B, self.k, self.k)

        eye = torch.eye(self.k, device=x.device)
        eye = eye.unsqueeze(0).repeat(B, 1, 1)

        return x + eye


class PointNetBinary(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()

        self.stn = STN3d(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fconv1 = nn.Conv1d(1152, 512, 1)
        self.fbn1 = nn.BatchNorm1d(512)

        self.fconv2 = nn.Conv1d(512, 256, 1)
        self.fbn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)

        self.fconv3 = nn.Conv1d(256, 2, 1)

    def forward(self, xyz):
        B, P, _ = xyz.shape

        x = xyz.transpose(2, 1)  # (B,3,P)

        T = self.stn(x)
        x = torch.bmm(T, x)

        x1 = F.relu(self.bn1(self.conv1(x)))      # (B,64,P)
        x2 = F.relu(self.bn2(self.conv2(x1)))     # (B,128,P)
        x3 = F.relu(self.bn3(self.conv3(x2)))     # (B,1024,P)

        xg = torch.max(x3, 2, keepdim=True)[0]
        xg = xg.repeat(1, 1, P)

        x = torch.cat([xg, x2], dim=1)            # (B,1152,P)

        x = F.relu(self.fbn1(self.fconv1(x)))
        x = F.relu(self.fbn2(self.fconv2(x)))
        x = self.dropout(x)

        logits = self.fconv3(x).transpose(2, 1)   # (B,P,2)
        return logits


# ============================================================
# Utils robustos para evitar bugs de NumPy del entorno
# ============================================================

def unwrap_meta_value(meta, key):
    """
    DataLoader con batch_size=1 suele convertir dict[str,str]
    en dict[str, list[str]]. Esta función recupera el valor.
    """
    value = meta[key]

    if isinstance(value, (list, tuple)):
        return value[0]

    return value


def safe_positive_mask_from_prob(prob_class1, threshold):
    """
    Evita operaciones vectorizadas tipo (arr >= thr).sum()
    porque el entorno NumPy ha mostrado errores internos.
    """
    mask = []
    for v in prob_class1:
        mask.append(float(v) >= float(threshold))

    return np.asarray(mask, dtype=bool)


def safe_count_mask(mask):
    c = 0
    for v in mask:
        if bool(v):
            c += 1
    return int(c)


def safe_points_from_mask(points, mask):
    selected = []
    for i, m in enumerate(mask):
        if bool(m):
            selected.append(points[i])

    if len(selected) == 0:
        return np.empty((0, 3), dtype=np.float32)

    return np.asarray(selected, dtype=np.float32)


def compute_binary_counts(pred_labels, gt_labels):
    tp = fp = fn = tn = 0

    for p, t in zip(pred_labels, gt_labels):
        p = int(p)
        t = int(t)

        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
        elif p == 0 and t == 0:
            tn += 1

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_root = out_dir / "test_predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    dataset = UFRNTestDataset(args.data_dir)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = PointNetBinary(dropout=args.dropout).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("[INFO] missing:", missing)
    print("[INFO] unexpected:", unexpected)

    if len(missing) > 0 or len(unexpected) > 0:
        print("[WARN] El checkpoint no calzó perfecto. Revise arquitectura/nombres de capas.")

    model.eval()

    all_summary = []
    global_counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    with torch.no_grad():
        for X, Y, meta in loader:
            X = X.to(device)

            logits = model(X)
            probs = torch.softmax(logits, dim=-1)

            probs_np = np.asarray(
                probs.detach().cpu().numpy()[0],
                dtype=np.float32
            )

            prob_class1 = np.asarray(
                probs_np[:, 1],
                dtype=np.float32
            )

            pts_norm = np.asarray(
                X.detach().cpu().numpy()[0],
                dtype=np.float32
            )

            y_true = np.asarray(
                Y.detach().cpu().numpy()[0],
                dtype=np.int32
            )

            mask_pos = safe_positive_mask_from_prob(
                prob_class1,
                args.threshold
            )

            pred_labels = np.asarray(
                [1 if bool(v) else 0 for v in mask_pos],
                dtype=np.int32
            )

            positive_points_norm = safe_points_from_mask(
                pts_norm,
                mask_pos
            )

            patient_id = str(unwrap_meta_value(meta, "patient_id"))

            center = np.asarray([
                float(unwrap_meta_value(meta, "center_x")),
                float(unwrap_meta_value(meta, "center_y")),
                float(unwrap_meta_value(meta, "center_z")),
            ], dtype=np.float32)

            scale = float(unwrap_meta_value(meta, "scale"))

            # ============================================================
            # Reconstruir coordenadas RAW:
            # raw = norm * scale + center
            # ============================================================

            pts_raw = (pts_norm * scale) + center
            positive_points_raw = safe_points_from_mask(
                pts_raw,
                mask_pos
            )

            gt_mask_pos = safe_positive_mask_from_prob(
                y_true,
                0.5
            )

            gt_positive_points_norm = safe_points_from_mask(
                pts_norm,
                gt_mask_pos
            )

            gt_positive_points_raw = safe_points_from_mask(
                pts_raw,
                gt_mask_pos
            )

            metrics = compute_binary_counts(
                pred_labels,
                y_true
            )

            for k in ["tp", "fp", "fn", "tn"]:
                global_counts[k] += int(metrics[k])

            positive_points = safe_count_mask(mask_pos)
            positive_frac = float(positive_points) / float(len(pred_labels))

            patient_dir = pred_root / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            # Normalizado
            np.save(patient_dir / "pred_points.npy", pts_norm.astype(np.float32))
            np.save(patient_dir / "pred_labels.npy", pred_labels.astype(np.int32))
            np.save(patient_dir / "pred_probs.npy", probs_np.astype(np.float32))
            np.save(patient_dir / "pred_prob_d21.npy", prob_class1.astype(np.float32))
            np.save(patient_dir / "pred_positive_points.npy", positive_points_norm.astype(np.float32))

            # Raw STL
            np.save(patient_dir / "pred_points_raw.npy", pts_raw.astype(np.float32))
            np.save(patient_dir / "pred_positive_points_raw.npy", positive_points_raw.astype(np.float32))

            # GT del dataset 8192, también en normalizado y raw
            np.save(patient_dir / "gt_labels.npy", y_true.astype(np.int32))
            np.save(patient_dir / "gt_positive_points.npy", gt_positive_points_norm.astype(np.float32))
            np.save(patient_dir / "gt_positive_points_raw.npy", gt_positive_points_raw.astype(np.float32))

            summary = {
                "patient_id": patient_id,
                "num_points": int(len(pred_labels)),
                "threshold": float(args.threshold),
                "positive_points": int(positive_points),
                "positive_frac": float(positive_frac),
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "scale": float(scale),
                "metrics": metrics,
                "files": {
                    "pred_points_norm": str(patient_dir / "pred_points.npy"),
                    "pred_positive_points_norm": str(patient_dir / "pred_positive_points.npy"),
                    "pred_points_raw": str(patient_dir / "pred_points_raw.npy"),
                    "pred_positive_points_raw": str(patient_dir / "pred_positive_points_raw.npy"),
                    "gt_positive_points_raw": str(patient_dir / "gt_positive_points_raw.npy"),
                }
            }

            save_json(summary, patient_dir / "summary.json")
            all_summary.append(summary)

            print(
                "[OK]",
                patient_id,
                "positive:",
                positive_points,
                "frac:",
                f"{positive_frac:.4f}",
                "f1:",
                f"{metrics['f1']:.4f}",
                "iou:",
                f"{metrics['iou']:.4f}",
            )

    global_metrics = compute_binary_counts(
        np.asarray(
            [1] * global_counts["tp"]
            + [1] * global_counts["fp"]
            + [0] * global_counts["fn"]
            + [0] * global_counts["tn"],
            dtype=np.int32
        ),
        np.asarray(
            [1] * global_counts["tp"]
            + [0] * global_counts["fp"]
            + [1] * global_counts["fn"]
            + [0] * global_counts["tn"],
            dtype=np.int32
        )
    )

    final_summary = {
        "data_dir": args.data_dir,
        "ckpt": args.ckpt,
        "out_dir": str(out_dir),
        "threshold": float(args.threshold),
        "device": str(device),
        "patients": all_summary,
        "global_metrics": global_metrics,
    }

    save_json(final_summary, out_dir / "all_predictions_summary.json")

    print("=" * 80)
    print("[DONE]")
    print(json.dumps({
        "n_patients": len(all_summary),
        "global_metrics": global_metrics,
    }, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()