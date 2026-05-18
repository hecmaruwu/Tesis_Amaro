#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_cloud(x):
    c = x.mean(dim=1, keepdim=True)
    x = x - c
    r = (x.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    return x / (r + 1e-8)

def knn(x, k):
    # x: (B, C, N)
    with torch.no_grad():
        xt = x.transpose(2, 1).contiguous()  # (B,N,C)
        dist = torch.cdist(xt, xt)
        idx = dist.topk(k=k, dim=-1, largest=False)[1]
    return idx


def get_graph_feature(x, k=20):
    # x: (B,C,N)
    B, C, N = x.size()
    idx = knn(x, k=k)  # (B,N,k)

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)

    x_t = x.transpose(2, 1).contiguous()  # (B,N,C)
    feature = x_t.view(B * N, C)[idx, :]
    feature = feature.view(B, N, k, C)

    x_central = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x_central, x_central), dim=3)  # (B,N,k,2C)
    return feature.permute(0, 3, 1, 2).contiguous()  # (B,2C,N,k)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.net(x)


class DGCNNSeg(nn.Module):
    """
    Compatible con checkpoint:
      ec1.net.0.weight (64, 6, 1, 1)
      ec2.net.0.weight (64, 128, 1, 1)
      ec3.net.0.weight (128, 128, 1, 1)
      ec4.net.0.weight (256, 256, 1, 1)
      fuse.0.weight (768, 512, 1)
      head.0.weight (512, 1280, 1)
      head.4.weight (256, 512, 1)
      head.8.weight (15, 256, 1)
    """

    def __init__(self, num_classes=15, k=20, emb_dims=768, dropout=0.5):
        super().__init__()
        self.k = k

        self.ec1 = EdgeConvBlock(6, 64)
        self.ec2 = EdgeConvBlock(128, 64)
        self.ec3 = EdgeConvBlock(128, 128)
        self.ec4 = EdgeConvBlock(256, 256)

        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.head = nn.Sequential(
            nn.Conv1d(emb_dims + 64 + 64 + 128 + 256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),

            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),

            nn.Conv1d(256, num_classes, kernel_size=1)
        )

    def forward(self, xyz):
        # xyz: (B,N,3)
        x = xyz.transpose(2, 1).contiguous()  # (B,3,N)

        x1 = self.ec1(get_graph_feature(x, k=self.k)).max(dim=-1)[0]   # (B,64,N)
        x2 = self.ec2(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]  # (B,64,N)
        x3 = self.ec3(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]  # (B,128,N)
        x4 = self.ec4(get_graph_feature(x3, k=self.k)).max(dim=-1)[0]  # (B,256,N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B,512,N)

        x_global = self.fuse(x_cat)                 # (B,emb,N)
        x_global = torch.max(x_global, dim=2, keepdim=True)[0]
        x_global = x_global.repeat(1, 1, xyz.shape[1])

        feat = torch.cat((x_global, x_cat), dim=1)  # (B,1280,N)
        logits = self.head(feat).transpose(2, 1)    # (B,N,C)
        return logits


def extract_state_dict(ckpt):
    if "model_state" in ckpt:
        return ckpt["model_state"]
    for k in ["model_state_dict", "model", "state_dict"]:
        if k in ckpt:
            return ckpt[k]
    raise RuntimeError(f"No encontré state_dict. Keys: {ckpt.keys()}")


def clean_state_dict(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


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
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--emb_dims", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.x_npz)["X"].astype(np.float32)
    x = torch.tensor(X, dtype=torch.float32, device=device)

    model = DGCNNSeg(
        num_classes=args.num_classes,
        k=args.k,
        emb_dims=args.emb_dims,
        dropout=args.dropout
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = clean_state_dict(extract_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

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
        "k": int(args.k),
        "emb_dims": int(args.emb_dims),
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