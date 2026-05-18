#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CloudDataset(Dataset):
    def __init__(self, data_dir, split):
        data_dir = Path(data_dir)
        self.X = np.load(data_dir / f"X_{split}.npz")["X"].astype(np.float32)
        self.Y = np.load(data_dir / f"Y_{split}.npz")["Y"].astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)


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
        return x + torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
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
        x = xyz.transpose(2, 1)
        T = self.stn(x)
        x = torch.bmm(T, x)

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))

        xg = torch.max(x3, 2, keepdim=True)[0].repeat(1, 1, P)
        x = torch.cat([xg, x2], dim=1)

        x = F.relu(self.fbn1(self.fconv1(x)))
        x = F.relu(self.fbn2(self.fconv2(x)))
        x = self.dropout(x)

        return self.fconv3(x).transpose(2, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt.get("state_dict") or ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}

    filtered, skipped = {}, []
    model_state = model.state_dict()

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped


@torch.no_grad()
def compute_metrics_from_counts(tp, fp, fn, tn):
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    iou = tp / max(tp + fp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, iou=iou, specificity=spec)


@torch.no_grad()
def batch_counts(logits, y):
    pred = logits.argmax(-1)
    p = pred.reshape(-1)
    t = y.reshape(-1)

    tp = ((p == 1) & (t == 1)).sum().item()
    fp = ((p == 1) & (t == 0)).sum().item()
    fn = ((p == 0) & (t == 1)).sum().item()
    tn = ((p == 0) & (t == 0)).sum().item()
    return tp, fp, fn, tn


def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    model.train(train)
    losses = []
    totals = dict(tp=0, fp=0, fn=0, tn=0)

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits.reshape(-1, 2), Y.reshape(-1))

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.item()))

        tp, fp, fn, tn = batch_counts(logits.detach(), Y)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["tn"] += tn

    m = compute_metrics_from_counts(totals["tp"], totals["fp"], totals["fn"], totals["tn"])
    m.update(
        loss=float(np.mean(losses)),
        tp=int(totals["tp"]),
        fp=int(totals["fp"]),
        fn=int(totals["fn"]),
        tn=int(totals["tn"]),
    )
    return m


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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

            if metric != "loss":
                best = max(history, key=lambda r: r[f"val_{metric}"])
                plt.axvline(best["epoch"], linestyle="--", alpha=0.6, label=f"best val {metric}: ep {best['epoch']}")
            else:
                best = min(history, key=lambda r: r[f"val_{metric}"])
                plt.axvline(best["epoch"], linestyle="--", alpha=0.6, label=f"best val loss: ep {best['epoch']}")

        plt.xlabel("Época")
        plt.title(f"Fine-tuning UFRN binario d21 - {metric}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plot_dir / f"curve_{metric}.png", dpi=300)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eta_min", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_weight", type=float, default=20.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min_delta", type=float, default=1e-5)

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loaders = {
        s: DataLoader(
            CloudDataset(args.data_dir, s),
            batch_size=args.batch_size,
            shuffle=(s == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        for s in ["train", "val", "test"]
    }

    model = PointNetSeg(num_classes=2, dropout=args.dropout).to(device)

    missing, unexpected, skipped = load_pretrained(model, args.ckpt)
    print("[INFO] missing:", missing)
    print("[INFO] unexpected:", unexpected)
    print("[INFO] skipped:", skipped)

    weights = torch.tensor([1.0, args.pos_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min
    )

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
            lr_now = optimizer.param_groups[0]["lr"]

            tr = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True, grad_clip=args.grad_clip)
            va = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False, grad_clip=args.grad_clip)

            row = {"epoch": epoch, "lr": lr_now}

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
        "last_epoch": history[-1]["epoch"] if history else 0,
    }, out_dir / "last.pt")

    save_json(history, out_dir / "history.json")
    plot_curves(history, out_dir)

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model_state"])

    test = run_epoch(model, loaders["test"], criterion, optimizer, device, train=False, grad_clip=args.grad_clip)
    save_json(test, out_dir / "test_metrics.json")

    run_meta = {
        "data_dir": args.data_dir,
        "ckpt": args.ckpt,
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
    }

    save_json(run_meta, out_dir / "run_meta.json")
    print("[DONE]")
    print(json.dumps(run_meta, indent=2))


if __name__ == "__main__":
    main()