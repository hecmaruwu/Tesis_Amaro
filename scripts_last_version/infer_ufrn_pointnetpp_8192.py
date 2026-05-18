#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utils geométricos
# ============================================================

def normalize_cloud(x):
    c = x.mean(dim=1, keepdim=True)
    x = x - c
    r = (x.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    return x / (r + 1e-8)


def knn_indices(query, ref, k):
    d = torch.cdist(query, ref)
    k = min(k, ref.size(1))
    return torch.topk(d, k=k, dim=-1, largest=False).indices


def batched_gather(points, idx):
    B, N, C = points.shape
    _, M, K = idx.shape
    b = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    return points[b, idx, :]


def three_nn_interp(xyz1, xyz2, feats2, k=3):
    idx = knn_indices(xyz1, xyz2, k=min(k, xyz2.size(1)))
    d = torch.cdist(xyz1, xyz2)
    knn_d = torch.gather(d, 2, idx).clamp(min=1e-8)

    w = 1.0 / knn_d
    w = w / w.sum(dim=-1, keepdim=True)

    feats2_perm = feats2.transpose(1, 2).contiguous()
    neigh = batched_gather(feats2_perm, idx)
    out = (w[..., None] * neigh).sum(dim=2)

    return out.transpose(1, 2).contiguous()


class MLP2d(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        layers = []
        c = in_ch
        for oc in channels:
            layers.append(nn.Conv2d(c, oc, 1, bias=False))
            layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.ReLU(True))
            c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP1d(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        layers = []
        c = in_ch
        for oc in channels:
            layers.append(nn.Conv1d(c, oc, 1, bias=False))
            layers.append(nn.BatchNorm1d(oc))
            layers.append(nn.ReLU(True))
            c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_ch, mlp):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = MLP2d(in_ch + 3, mlp)

    def forward(self, xyz, feats):
        # xyz: (B,N,3)
        # feats: (B,C,N) or None
        B, N, _ = xyz.shape
        M = min(self.npoint, N)

        # muestreo determinista simple compatible para inferencia
        idx_center = torch.linspace(0, N - 1, M, device=xyz.device).long()
        idx_center = idx_center[None, :].repeat(B, 1)

        centers = torch.gather(xyz, 1, idx_center[..., None].expand(-1, -1, 3))
        idx_knn = knn_indices(centers, xyz, self.nsample)

        neigh_xyz = batched_gather(xyz, idx_knn)
        rel_xyz = neigh_xyz - centers[:, :, None, :]

        if feats is not None:
            feats_t = feats.transpose(1, 2).contiguous()
            neigh_feats = batched_gather(feats_t, idx_knn)
            grouped = torch.cat([rel_xyz, neigh_feats], dim=-1)
        else:
            grouped = rel_xyz

        grouped = grouped.permute(0, 3, 1, 2).contiguous()
        out = self.mlp(grouped).max(dim=-1)[0]

        return centers, out


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = MLP1d(in_ch, mlp)

    def forward(self, xyz1, xyz2, feats1, feats2):
        # propaga de xyz2 hacia xyz1
        interp = three_nn_interp(xyz1, xyz2, feats2, k=3)

        if feats1 is not None:
            new_feats = torch.cat([feats1, interp], dim=1)
        else:
            new_feats = interp

        return self.mlp(new_feats)


class PointNetPP(nn.Module):
    def __init__(self, num_classes=15, nsample=32):
        super().__init__()

        self.sa1 = SetAbstraction(2048, nsample, 0, [64, 64, 128])
        self.sa2 = SetAbstraction(512, nsample, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(128, nsample, 256, [256, 256, 512])
        self.sa4 = SetAbstraction(32, nsample, 512, [512, 512, 1024])

        self.fp4 = FeaturePropagation(1024 + 512, [256, 256])
        self.fp3 = FeaturePropagation(256 + 256, [256, 256])
        self.fp2 = FeaturePropagation(256 + 128, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128])

        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz):
        # xyz: (B,N,3)
        l0_xyz = xyz
        l0_feats = None

        l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)
        l4_xyz, l4_feats = self.sa4(l3_xyz, l3_feats)

        l3_feats_new = self.fp4(l3_xyz, l4_xyz, l3_feats, l4_feats)
        l2_feats_new = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats_new)
        l1_feats_new = self.fp2(l1_xyz, l2_xyz, l1_feats, l2_feats_new)
        l0_feats_new = self.fp1(l0_xyz, l1_xyz, None, l1_feats_new)

        logits = self.classifier(l0_feats_new).transpose(2, 1)
        return logits


def safe_counts(pred_np):
    counts = {}
    for p in pred_np.tolist():
        p = int(p)
        counts[p] = counts.get(p, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=15)
    ap.add_argument("--d21_internal", type=int, default=8)
    ap.add_argument("--nsample", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.x_npz)["X"].astype(np.float32)
    x = torch.tensor(X, dtype=torch.float32, device=device)

    if args.normalize:
        x = normalize_cloud(x)

    model = PointNetPP(
        num_classes=args.num_classes,
        nsample=args.nsample
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

    pred_np = pred[0].detach().cpu().numpy().astype(np.int32)
    prob_np = prob[0].detach().cpu().numpy().astype(np.float32)

    np.save(out_dir / "pred_labels.npy", np.asarray(pred_np, dtype=np.int32))
    np.save(out_dir / "pred_probs.npy", np.asarray(prob_np, dtype=np.float32))

    counts = safe_counts(pred_np)
    d21_count = int(counts.get(int(args.d21_internal), 0))
    n_points = int(pred_np.shape[0])

    summary = {
        "x_npz": str(args.x_npz),
        "ckpt": str(args.ckpt),
        "device": str(device),
        "input_shape": [int(v) for v in X.shape],
        "num_points": n_points,
        "num_classes": int(args.num_classes),
        "d21_internal": int(args.d21_internal),
        "nsample": int(args.nsample),
        "normalize": bool(args.normalize),
        "pred_class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
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