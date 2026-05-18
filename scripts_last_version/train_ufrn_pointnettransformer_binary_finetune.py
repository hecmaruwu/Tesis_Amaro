#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_ufrn_pointnettransformer_binary_finetune.py

Fine-tuning binario UFRN d21/resto usando PointNetTransformer preentrenado en Teeth3DS.

Checkpoint esperado:
  embed.net.0.weight                 (128, 3)
  embed.net.3.weight                 (256, 128)
  pos.net.0.weight                   (128, 3)
  pos.net.3.weight                   (256, 128)
  encoder.0..3.layer.*              TransformerEncoderLayer, d_model=256, nhead=8, ff=512
  head.0.weight                      (256, 256)
  head.3.weight                      (15, 256)  -> se reemplaza por (2, 256)

La única capa que debe saltarse en fine-tuning binario es:
  head.3.weight
  head.3.bias

Incluye:
- pretrained y scratch con el mismo script
- selección de best.pt por val_f1 binario d21
- CosineAnnealingLR
- test_metrics.json, history.json, metrics_epoch.csv, plots/
- token_subsample opcional para evitar OOM en self-attention
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class CloudDataset(Dataset):
    def __init__(self, data_dir, split):
        data_dir = Path(data_dir)
        self.X = np.load(data_dir / f"X_{split}.npz", allow_pickle=True)["X"].astype(np.float32)
        self.Y = np.load(data_dir / f"Y_{split}.npz", allow_pickle=True)["Y"].astype(np.int64)
        if self.X.shape[0] != self.Y.shape[0]:
            raise RuntimeError(f"X_{split} e Y_{split} tienen distinto número de muestras.")

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return (
            torch.tensor(np.asarray(self.X[idx], dtype=np.float32), dtype=torch.float32),
            torch.tensor(np.asarray(self.Y[idx], dtype=np.int64), dtype=torch.long),
        )


class LinearMLP(nn.Module):
    """
    MLP con nombres:
      net.0 = Linear(in_dim, hidden_dim)
      net.3 = Linear(hidden_dim, out_dim)
    """
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=256, dropout=0.05, activation="gelu"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"activation no soportada: {activation}")
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.05, activation="gelu", norm_first=False):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=bool(norm_first),
        )

    def forward(self, x):
        return self.layer(x)


def gather_tokens(xyz, token_subsample: int, training: bool, random_tokens: bool):
    B, N, _ = xyz.shape
    if token_subsample is None or int(token_subsample) <= 0 or int(token_subsample) >= N:
        idx = torch.arange(N, device=xyz.device, dtype=torch.long)
        return xyz, idx
    S = int(token_subsample)
    if training and random_tokens:
        idx = torch.randperm(N, device=xyz.device)[:S]
        idx, _ = torch.sort(idx)
    else:
        idx = torch.linspace(0, N - 1, S, device=xyz.device).long()
    token_xyz = xyz[:, idx, :].contiguous()
    return token_xyz, idx


def interpolate_logits_knn(xyz_query, xyz_tokens, logits_tokens, k=3, chunk_size=2048):
    B, N, _ = xyz_query.shape
    _, S, _ = xyz_tokens.shape
    k = min(int(k), int(S))
    chunk_size = max(1, int(chunk_size))
    outs = []
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        q = xyz_query[:, start:end, :]
        d = torch.cdist(q, xyz_tokens)
        knn_d, idx = torch.topk(d, k=k, dim=-1, largest=False)
        knn_d = knn_d.clamp_min(1e-8)
        w = 1.0 / knn_d
        w = w / w.sum(dim=-1, keepdim=True)
        M = q.shape[1]
        batch = torch.arange(B, device=xyz_query.device)[:, None, None].expand(B, M, k)
        neigh_logits = logits_tokens[batch, idx, :]
        out = (w[..., None] * neigh_logits).sum(dim=2)
        outs.append(out)
    return torch.cat(outs, dim=1).contiguous()


class PointNetTransformerBinary(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=256,
        hidden_dim=128,
        depth=4,
        nhead=8,
        dim_feedforward=512,
        dropout=0.05,
        activation="gelu",
        norm_first=False,
        token_subsample=2048,
        token_random=True,
        prop_k=3,
        prop_chunk=2048,
    ):
        super().__init__()
        self.token_subsample = int(token_subsample)
        self.token_random = bool(token_random)
        self.prop_k = int(prop_k)
        self.prop_chunk = int(prop_chunk)
        self.embed = LinearMLP(3, hidden_dim, d_model, dropout, activation)
        self.pos = LinearMLP(3, hidden_dim, d_model, dropout, activation)
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
            for _ in range(depth)
        ])
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"activation no soportada: {activation}")
        self.head = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), int(num_classes)),
        )

    def forward(self, xyz):
        token_xyz, _ = gather_tokens(xyz, self.token_subsample, self.training, self.token_random)
        h = self.embed(token_xyz) + self.pos(token_xyz)
        for block in self.encoder:
            h = block(h)
        logits_tokens = self.head(h)
        if token_xyz.shape[1] == xyz.shape[1]:
            return logits_tokens.contiguous()
        logits = interpolate_logits_knn(xyz, token_xyz, logits_tokens, self.prop_k, self.prop_chunk)
        return logits


def get_state_from_ckpt(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state", "model_state_dict", "model", "state_dict"]:
            if key in ckpt:
                return ckpt[key]
    return ckpt


def load_pretrained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = get_state_from_ckpt(ckpt)
    state = {str(k).replace("module.", ""): v for k, v in state.items()}
    model_state = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped


def counts_from_logits(logits, y):
    pred = logits.argmax(dim=-1)
    p = pred.reshape(-1)
    t = y.reshape(-1)
    tp = ((p == 1) & (t == 1)).sum().item()
    fp = ((p == 1) & (t == 0)).sum().item()
    fn = ((p == 0) & (t == 1)).sum().item()
    tn = ((p == 0) & (t == 0)).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def metrics_from_counts(tp, fp, fn, tn):
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "iou": float(iou),
        "specificity": float(specificity),
    }


def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    model.train(bool(train))
    losses = []
    total_tp = total_fp = total_fn = total_tn = 0
    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits.reshape(-1, 2), Y.reshape(-1))
        if train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        losses.append(float(loss.item()))
        with torch.no_grad():
            tp, fp, fn, tn = counts_from_logits(logits, Y)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
    m = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    m.update({
        "loss": float(np.mean(losses)) if losses else 0.0,
        "tp": int(total_tp), "fp": int(total_fp), "fn": int(total_fn), "tn": int(total_tn),
    })
    return m


def plot_curves(history, out_dir):
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["loss", "f1", "iou", "prec", "rec", "acc", "specificity", "lr"]
    for metric in metrics:
        xs = [r["epoch"] for r in history]
        plt.figure(figsize=(8, 5))
        if metric == "lr":
            ys = [r["lr"] for r in history]
            plt.plot(xs, ys, label="learning_rate")
            plt.ylabel("lr")
        else:
            plt.plot(xs, [r[f"train_{metric}"] for r in history], label=f"train_{metric}")
            plt.plot(xs, [r[f"val_{metric}"] for r in history], label=f"val_{metric}")
            plt.ylabel(metric)
            if metric == "loss":
                best = min(history, key=lambda r: r[f"val_{metric}"])
                label = f"best val loss: ep {best['epoch']}"
            else:
                best = max(history, key=lambda r: r[f"val_{metric}"])
                label = f"best val {metric}: ep {best['epoch']}"
            plt.axvline(best["epoch"], linestyle="--", alpha=0.6, label=label)
        plt.xlabel("Época")
        plt.title(f"PointNetTransformer UFRN binario d21 - {metric}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"curve_{metric}.png", dpi=300)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--from_scratch", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eta_min", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_weight", type=float, default=20.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=1e-5)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--dim_feedforward", type=int, default=512)
    ap.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
    ap.add_argument("--norm_first", action="store_true")
    ap.add_argument("--token_subsample", type=int, default=2048)
    ap.add_argument("--no_token_random", action="store_true")
    ap.add_argument("--prop_k", type=int, default=3)
    ap.add_argument("--prop_chunk", type=int, default=2048)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CloudDataset(args.data_dir, split)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = PointNetTransformerBinary(
        num_classes=2,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        norm_first=args.norm_first,
        token_subsample=args.token_subsample,
        token_random=not bool(args.no_token_random),
        prop_k=args.prop_k,
        prop_chunk=args.prop_chunk,
    ).to(device)

    if args.from_scratch or args.ckpt is None:
        missing, unexpected, skipped = [], [], []
        print("[INFO] Entrenando PointNetTransformer binario desde cero.")
    else:
        missing, unexpected, skipped = load_pretrained(model, args.ckpt)
        print("[INFO] missing:", missing)
        print("[INFO] unexpected:", unexpected)
        print("[INFO] skipped:", skipped)

    weights = torch.tensor([1.0, float(args.pos_weight)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []
    fieldnames = [
        "epoch", "lr",
        "train_loss", "train_acc", "train_prec", "train_rec", "train_f1", "train_iou", "train_specificity",
        "train_tp", "train_fp", "train_fn", "train_tn",
        "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "val_iou", "val_specificity",
        "val_tp", "val_fp", "val_fn", "val_tn",
    ]

    csv_path = out_dir / "metrics_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            lr_now = float(optimizer.param_groups[0]["lr"])
            tr = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True, grad_clip=args.grad_clip)
            va = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False, grad_clip=args.grad_clip)
            row = {"epoch": int(epoch), "lr": lr_now}
            for k in ["loss", "acc", "prec", "rec", "f1", "iou", "specificity", "tp", "fp", "fn", "tn"]:
                row[f"train_{k}"] = tr[k]
                row[f"val_{k}"] = va[k]
            writer.writerow(row)
            f.flush()
            history.append(row)
            improved = va["f1"] > best_val_f1 + args.min_delta
            if improved:
                best_val_f1 = float(va["f1"])
                best_epoch = int(epoch)
                bad_epochs = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "best_val_f1": best_val_f1,
                    "val_metrics": va,
                    "architecture": "PointNetTransformerBinary_dm256_depth4_h8_ff512",
                }, out_dir / "best.pt")
            else:
                bad_epochs += 1
            print(
                f"[{epoch:03d}/{args.epochs}] "
                f"lr={lr_now:.2e} | "
                f"train loss={tr['loss']:.4f} f1={tr['f1']:.4f} iou={tr['iou']:.4f} "
                f"P={tr['prec']:.4f} R={tr['rec']:.4f} | "
                f"val loss={va['loss']:.4f} f1={va['f1']:.4f} iou={va['iou']:.4f} "
                f"P={va['prec']:.4f} R={va['rec']:.4f} | "
                f"best_ep={best_epoch} bad={bad_epochs}/{args.patience}"
            )
            scheduler.step()
            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] Sin mejora en val_f1 por {args.patience} épocas.")
                break

    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "last_epoch": int(history[-1]["epoch"] if history else 0),
        "architecture": "PointNetTransformerBinary_dm256_depth4_h8_ff512",
    }, out_dir / "last.pt")

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir)

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state"])
    test = run_epoch(model, loaders["test"], criterion, optimizer, device, train=False, grad_clip=args.grad_clip)
    save_json(test, out_dir / "test_metrics.json")

    run_meta = {
        "script_name": Path(__file__).name,
        "data_dir": args.data_dir,
        "ckpt": args.ckpt,
        "from_scratch": bool(args.from_scratch),
        "out_dir": str(out_dir),
        "device": str(device),
        "epochs_requested": int(args.epochs),
        "epochs_ran": int(history[-1]["epoch"] if history else 0),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "test": test,
        "missing": [str(x) for x in missing],
        "unexpected": [str(x) for x in unexpected],
        "skipped": [str(x) for x in skipped],
        "scheduler": "CosineAnnealingLR",
        "eta_min": float(args.eta_min),
        "pos_weight": float(args.pos_weight),
        "d_model": int(args.d_model),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "nhead": int(args.nhead),
        "dim_feedforward": int(args.dim_feedforward),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "norm_first": bool(args.norm_first),
        "token_subsample": int(args.token_subsample),
        "token_random": not bool(args.no_token_random),
        "prop_k": int(args.prop_k),
        "prop_chunk": int(args.prop_chunk),
        "architecture": "PointNetTransformerBinary_dm256_depth4_h8_ff512",
    }
    save_json(run_meta, out_dir / "run_meta.json")
    print("[DONE]")
    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()
