#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento mejorado de PointNet con BatchNorm y soporte para N variable.

Basado en Qi et al. (2017) con normalización global, dropout, BatchNorm y pesos de clase opcionales.

Dataset esperado:
  data_path/X_train.npz, Y_train.npz, etc.
  data_path/artifacts/class_weights.json (opcional)

Salida:
  out_dir/<tag>/pointnet_bn/
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ====================================================
# Dataset
# ====================================================

class NpzPointCloudDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)["X"].astype(np.float32)
        self.Y = np.load(y_path)["Y"].astype(np.int64)
        assert self.X.shape[0] == self.Y.shape[0]
        # Normalización global (centra y escala a unidad)
        self.X -= self.X.mean(axis=1, keepdims=True)
        r = np.linalg.norm(self.X, axis=2, keepdims=True).max(axis=1, keepdims=True)
        self.X /= (r + 1e-6)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].transpose(1, 0)  # (3, N)
        y = self.Y[idx]
        return x, y

# ====================================================
# Modelo PointNet con BatchNorm estable y N variable
# ====================================================

class PointNetBN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

        self.fc1 = nn.Linear(2048, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, 3, N)
        B, _, N = x.size()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Pooling global
        global_feat = torch.max(x, 2, keepdim=True)[0]  # (B,1024,1)
        global_feat = global_feat.repeat(1, 1, N)       # (B,1024,N)
        x = torch.cat([x, global_feat], 1)              # (B,2048,N)

        # Aplanar y aplicar capas FC por punto
        x = x.transpose(2, 1).contiguous().view(B * N, 2048)

        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = x.view(B, N, -1).transpose(2, 1)  # (B, num_classes, N)
        return x

# ====================================================
# Entrenamiento y evaluación
# ====================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_points = 0.0, 0, 0
    for X, Y in tqdm(loader, desc="Train", leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()
        preds = out.argmax(dim=1)
        total_loss += loss.item() * X.size(0)
        total_correct += (preds == Y).sum().item()
        total_points += Y.numel()
    return total_loss / len(loader.dataset), total_correct / total_points


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_points = 0.0, 0, 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            out = model(X)
            loss = criterion(out, Y)
            preds = out.argmax(dim=1)
            total_loss += loss.item() * X.size(0)
            total_correct += (preds == Y).sum().item()
            total_points += Y.numel()
    return total_loss / len(loader.dataset), total_correct / total_points

# ====================================================
# Main
# ====================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="pointnet_bn_varN")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading data from {data_path}")
    train_ds = NpzPointCloudDataset(data_path / "X_train.npz", data_path / "Y_train.npz")
    val_ds   = NpzPointCloudDataset(data_path / "X_val.npz", data_path / "Y_val.npz")

    num_classes = int(train_ds.Y.max()) + 1
    print(f"[DATA] num_classes={num_classes}")

    # Pesos de clase (opcional)
    weights_path = data_path / "artifacts" / "class_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            cw = json.load(f)
        # Reforzamos que tenga exactamente num_classes entradas
        weights_tensor = torch.ones(num_classes, dtype=torch.float32)
        for k, v in cw.items():
            idx = int(k)
            if idx < num_classes:
                weights_tensor[idx] = float(v)
        weights_tensor = weights_tensor.to(device)
        print(f"[INFO] Loaded class weights for {len(weights_tensor)} classes.")
    else:
        weights_tensor = None

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PointNetBN(num_classes=num_classes, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc, patience, max_patience = 0.0, 0, 35

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"[Ep {epoch:03d}] tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} tr_acc={tr_acc:.4f} va_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), out_dir / "best_pointnet_bn_varN.pt")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"[STOP] Early stopping at epoch {epoch} (best={best_val_acc:.4f})")
                break

    print(f"[DONE] Entrenamiento completado. Mejor val_acc={best_val_acc:.4f}")

if __name__ == "__main__":
    main()
