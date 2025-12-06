#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pointnetpp_improved.py
--------------------------------------------------------------
Implementación académica de PointNet++ mejorado para
segmentación de dientes en nubes de puntos 3D.

Basado en:
  - Encoder-decoder tipo PointNet++
  - SPFE: Single-Point Preliminary Feature Extraction
  - WSLFA: Weighted-Sum Local Feature Aggregation

Compatible con datasets preparados por:
  - augmentation_and_split_paperlike_v3.py
    (X_train.npz, Y_train.npz, etc. + artifacts/)

Autor: Adaptado por ChatGPT (GPT-5.1) para Tesis_Amaro
"""

# ==============================================================
# === Importaciones generales =================================
# ==============================================================

import os, sys, json, csv, time, random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================
# === Utilidades generales =====================================
# ==============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(x, device: torch.device):
    if isinstance(x, (tuple, list)):
        return [to_device(t, device) for t in x]
    return x.to(device, non_blocking=True)


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_cloud(x: torch.Tensor) -> torch.Tensor:
    """
    Normaliza cada nube a esfera unitaria (centro=0, radio máx=1).
    x: (B, P, 3)
    """
    c = x.mean(dim=1, keepdim=True)                               # (B,1,3)
    x = x - c
    r = (x.pow(2).sum(-1).sqrt()).max(dim=1, keepdim=True)[0]      # (B,1)
    r = r.unsqueeze(-1)                                            # (B,1,1)
    return x / (r + 1e-8)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_curves(history: Dict[str, List[float]], out_dir: Path, model_name: str):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["loss", "acc", "f1", "iou", "d21_acc", "d21_f1", "d21_iou"]
    for k in keys:
        plt.figure(figsize=(7, 4))
        for split in ["train", "val"]:
            kk = f"{split}_{k}"
            if kk in history and len(history[kk]) > 0:
                plt.plot(history[kk], label=split)
        plt.xlabel("Época"); plt.ylabel(k.upper())
        plt.title(f"{model_name} – {k.upper()}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{k}.png", dpi=300)
        plt.close()


# ==============================================================
# === Dataset y DataLoader =====================================
# ==============================================================

class CloudDataset(Dataset):
    """
    Dataset de nubes de puntos:
      - Carga X e Y desde archivos .npz con claves 'X' y 'Y'
      - Devuelve X: (P,3), Y: (P,)
    """
    def __init__(self, X_path: Path, Y_path: Path):
        self.X = np.load(X_path)["X"].astype(np.float32)
        self.Y = np.load(Y_path)["Y"].astype(np.int64)
        assert self.X.shape[0] == self.Y.shape[0], \
            f"Dimensión inconsistente entre {X_path.name} y {Y_path.name}"

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx])   # (P,3)
        Y = torch.from_numpy(self.Y[idx])   # (P,)
        return X, Y


def make_loaders(data_dir: Path, batch_size: int = 8, num_workers: int = 4) -> Dict[str, DataLoader]:
    data_dir = Path(data_dir)
    splits = ["train", "val", "test"]
    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        Xp = data_dir / f"X_{split}.npz"
        Yp = data_dir / f"Y_{split}.npz"
        if not (Xp.exists() and Yp.exists()):
            raise FileNotFoundError(f"Faltan archivos para split {split}: {Xp}, {Yp}")
        ds = CloudDataset(Xp, Yp)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return loaders


# ==============================================================
# === Métricas y matrices de confusión =========================
# ==============================================================

@torch.no_grad()
def confusion_matrix(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    logits: (B,P,C), y_true: (B,P)
    C_ij = # gt=i, pred=j
    """
    preds = logits.argmax(dim=-1)
    t = y_true.view(-1)
    p = preds.view(-1)
    valid = (t >= 0) & (t < num_classes)
    t = t[valid]; p = p[valid]
    idx = t * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)
    return cm


def macro_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    cm = cm.float()
    tp = torch.diag(cm)
    gt = cm.sum(1).clamp_min(1e-8)
    pd = cm.sum(0).clamp_min(1e-8)

    acc = (tp.sum() / cm.sum().clamp_min(1e-8)).item()
    prec = torch.mean(tp / pd).item()
    rec = torch.mean(tp / gt).item()
    f1 = torch.mean(2 * tp / (gt + pd).clamp_min(1e-8)).item()
    iou = torch.mean(tp / (gt + pd - tp).clamp_min(1e-8)).item()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "iou": iou}


def d21_metrics(logits: torch.Tensor, y_true: torch.Tensor, cls_id: int) -> Dict[str, float]:
    """
    Métricas específicas para el diente 21 (o la clase indicada).
    """
    preds = logits.argmax(dim=-1).view(-1)
    t = y_true.view(-1)

    tp = ((preds == cls_id) & (t == cls_id)).sum().float()
    fp = ((preds == cls_id) & (t != cls_id)).sum().float()
    fn = ((preds != cls_id) & (t == cls_id)).sum().float()
    tn = ((preds != cls_id) & (t != cls_id)).sum().float()

    acc = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec = (tp / (tp + fn + 1e-8)).item()
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = (tp / (tp + fp + fn + 1e-8)).item()

    return {"d21_acc": acc, "d21_f1": f1, "d21_iou": iou}


# ==============================================================
# === Utilidades geométricas (kNN, gather batched) =============
# ==============================================================

def knn_indices(query: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """
    query: (B, M, 3)
    ref:   (B, N, 3)
    out:   (B, M, k)
    """
    d = torch.cdist(query, ref)  # (B,M,N)
    k = min(k, ref.size(1))
    idx = torch.topk(d, k=k, dim=-1, largest=False).indices
    return idx


def batched_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, M, K)
    out:    (B, M, K, C)
    """
    B, N, C = points.shape
    _, M, K = idx.shape
    b = torch.arange(B, device=points.device)[:, None, None].expand(B, M, K)
    out = points[b, idx, :]  # (B,M,K,C)
    return out


# ==============================================================
# === Bloques de la arquitectura mejorada ======================
# ==============================================================

class SPFE(nn.Module):
    """
    Single-Point Preliminary Feature Extraction (SPFE)
    ------------------------------------------------------------------
    Extrae características iniciales por punto via MLP 1D sobre xyz.

    Entrada:
      xyz: (B, P, 3)
    Salida:
      feats: (B, C, P)
    """
    def __init__(self, in_ch: int = 3, channels: List[int] = [64, 64, 128]):
        super().__init__()
        layers = []
        c = in_ch
        for oc in channels:
            layers += [
                nn.Conv1d(c, oc, 1),
                nn.BatchNorm1d(oc),
                nn.ReLU(True),
            ]
            c = oc
        self.net = nn.Sequential(*layers)
        self.out_ch = c

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B,P,3) -> (B,3,P)
        x = xyz.transpose(2, 1)
        f = self.net(x)  # (B,C,P)
        return f


class WSLFA_SA(nn.Module):
    """
    Weighted-Sum Local Feature Aggregation (WSLFA) + Set Abstraction
    ------------------------------------------------------------------
    - Selecciona centros determinísticamente (equispaciado)
    - Vecinos kNN alrededor de cada centro
    - MLP local (conv2d) sobre vecinos
    - Agregación ponderada (softmax sobre K) en lugar de max-pool

    Entrada:
      xyz:   (B, P, 3)
      feats: (B, C, P) o None

    Salida:
      xyz_down:   (B, M, 3)
      feats_down: (B, C_out, M)
    """
    def __init__(self, nsample: int, in_ch: int, out_ch: int, hidden: int = 128):
        super().__init__()
        self.nsample = nsample

        # MLP local sobre parches: (in_ch + 3) canales
        self.local_mlp = nn.Sequential(
            nn.Conv2d(in_ch + 3, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.Conv2d(hidden, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
        # MLP para obtener pesos por vecino
        self.weight_mlp = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 2, 1),
            nn.ReLU(True),
            nn.Conv2d(out_ch // 2, 1, 1),
        )
        self.out_ch = out_ch

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor]):
        """
        xyz:   (B,P,3)
        feats: (B,C,P) o None
        """
        B, P, _ = xyz.shape
        M = max(1, P // 4)  # número de centros

        # Centros equiespaciados (determinista)
        idx_center = torch.linspace(
            0, P - 1, M, device=xyz.device, dtype=torch.long
        )[None, :].repeat(B, 1)  # (B,M)
        centers = torch.gather(xyz, 1, idx_center[..., None].expand(-1, -1, 3))  # (B,M,3)

        # kNN por centro
        idx_knn = knn_indices(centers, xyz, self.nsample)         # (B,M,K)
        neigh_xyz = batched_gather(xyz, idx_knn)                  # (B,M,K,3)
        local_xyz = neigh_xyz - centers[:, :, None, :]            # (B,M,K,3)
        local_xyz = local_xyz.permute(0, 3, 1, 2).contiguous()    # (B,3,M,K)

        if feats is not None:
            feats_perm = feats.transpose(1, 2).contiguous()       # (B,P,C)
            neigh_f = batched_gather(feats_perm, idx_knn)         # (B,M,K,C)
            neigh_f = neigh_f.permute(0, 3, 1, 2).contiguous()    # (B,C,M,K)
            cat = torch.cat([local_xyz, neigh_f], dim=1)          # (B,3+C,M,K)
        else:
            cat = local_xyz                                       # (B,3,M,K)

        # MLP local
        feat_local = self.local_mlp(cat)                          # (B,C_out,M,K)

        # WSLFA: pesos por vecino (softmax en K)
        w = self.weight_mlp(feat_local)                           # (B,1,M,K)
        w = torch.softmax(w, dim=-1)                              # (B,1,M,K)
        feat_agg = (feat_local * w).sum(dim=-1)                   # (B,C_out,M)

        return centers, feat_agg


class FP_Layer(nn.Module):
    """
    Feature Propagation (FP) con interpolación por vecinos más cercanos.
    """
    def __init__(self, in_ch: int, mlp: List[int]):
        super().__init__()
        layers = []
        c = in_ch
        for oc in mlp:
            layers += [
                nn.Conv1d(c, oc, 1),
                nn.BatchNorm1d(oc),
                nn.ReLU(True),
            ]
            c = oc
        self.net = nn.Sequential(*layers)
        self.out_ch = c

    def forward(self, xyz1, xyz2, feats1, feats2):
        """
        xyz1:   (B,N1,3)  destino
        xyz2:   (B,N2,3)  fuente
        feats1: (B,C1,N1) (puede ser None)
        feats2: (B,C2,N2)
        """
        B, N1, _ = xyz1.shape
        _, C2, N2 = feats2.shape

        # k=3 vecinos
        idx = knn_indices(xyz1, xyz2, k=min(3, N2))         # (B,N1,3)
        d = torch.cdist(xyz1, xyz2)                         # (B,N1,N2)
        knn_d = torch.gather(d, 2, idx).clamp_min(1e-8)    # (B,N1,3)
        w = 1.0 / knn_d
        w = w / w.sum(dim=-1, keepdim=True)                # (B,N1,3)

        f2p = feats2.transpose(1, 2).contiguous()          # (B,N2,C2)
        neigh = batched_gather(f2p, idx)                   # (B,N1,3,C2)
        out = (w[..., None] * neigh).sum(dim=2)            # (B,N1,C2)
        out = out.transpose(1, 2).contiguous()             # (B,C2,N1)

        if feats1 is not None:
            out = torch.cat([out, feats1], dim=1)          # (B,C2+C1,N1)

        return self.net(out)                               # (B,C',N1)


class ImprovedPointNet2Seg(nn.Module):
    """
    PointNet++ mejorado con:
      - SPFE: extracción preliminar por punto
      - WSLFA: agregación local ponderada en SA
      - Encoder-decoder con FP
    """
    def __init__(self, num_classes: int = 10, nsample: int = 32):
        super().__init__()
        # SPFE
        self.spfe = SPFE(in_ch=3, channels=[64, 64, 128])
        c0 = self.spfe.out_ch                                # 128

        # Encoder (dos niveles)
        self.sa1 = WSLFA_SA(nsample=nsample, in_ch=c0,   out_ch=256)
        self.sa2 = WSLFA_SA(nsample=nsample, in_ch=256,  out_ch=512)

        # Decoder (FP)
        self.fp1 = FP_Layer(in_ch=512 + 256, mlp=[256, 256])    # lvl2 -> lvl1
        self.fp2 = FP_Layer(in_ch=256 + c0,  mlp=[256, 128])    # lvl1 -> original

        # Cabeza final de segmentación
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B,P,3)
        B, P, _ = xyz.shape

        # 1) SPFE
        f0 = self.spfe(xyz)                    # (B,C0,P)

        # 2) SA + WSLFA
        xyz1, f1 = self.sa1(xyz, f0)          # (B,M1,3), (B,256,M1)
        xyz2, f2 = self.sa2(xyz1, f1)         # (B,M2,3), (B,512,M2)

        # 3) FP (decoder)
        f1_up = self.fp1(xyz1, xyz2, f1, f2)  # (B,256,M1)
        f0_up = self.fp2(xyz,  xyz1, f0, f1_up)  # (B,128,P)

        # 4) Head
        out = self.head(f0_up).transpose(2, 1)   # (B,P,C)
        return out


# ==============================================================
# === Early Stopping y class_weights ===========================
# ==============================================================

class EarlyStopping:
    def __init__(self, patience: int = 20, delta: float = 1e-4, ckpt_dir: Optional[Path] = None):
        self.patience = patience
        self.delta = delta
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.ckpt_dir:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch},
                    self.ckpt_dir / "best.pt"
                )
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def load_class_weights(artifacts_dir: Path, num_classes: int) -> Optional[torch.Tensor]:
    f = artifacts_dir / "class_weights.json"
    if not f.exists():
        return None
    try:
        w = json.loads(f.read_text())
        if isinstance(w, dict):
            arr = np.zeros((num_classes,), dtype=np.float32)
            for k, v in w.items():
                idx = int(k)
                if 0 <= idx < num_classes:
                    arr[idx] = float(v)
        else:
            arr = np.array(w, dtype=np.float32)
        if arr.shape[0] != num_classes:
            raise ValueError("Tamaño de class_weights.json no coincide con num_classes")
        return torch.tensor(arr, dtype=torch.float32)
    except Exception as e:
        print(f"[WARN] Error cargando class_weights.json: {e}")
        return None


# ==============================================================
# === Bucle de entrenamiento y evaluación ======================
# ==============================================================

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion, device: torch.device,
                    num_classes: int, d21_id: int) -> Tuple[float, Dict[str, float]]:

    model.train()
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21 = {"d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0}
    batches = 0

    for X, Y in loader:
        X, Y = to_device((X, Y), device)
        X = sanitize_tensor(normalize_cloud(sanitize_tensor(X)))  # (B,P,3)

        logits = sanitize_tensor(model(X))                        # (B,P,C)
        Y = torch.clamp(Y, 0, num_classes - 1)

        loss = criterion(logits.transpose(2, 1), Y)               # (B,C,P) vs (B,P)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.detach().cpu())
        cm += confusion_matrix(logits, Y, num_classes)
        d21m = d21_metrics(logits, Y, d21_id)
        for k in d21: d21[k] += d21m[k]
        batches += 1

    macro = macro_from_cm(cm)
    for k in d21: d21[k] /= max(1, batches)
    return loss_sum / max(1, batches), {**macro, **d21}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion,
             device: torch.device, num_classes: int, d21_id: int) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21 = {"d21_acc": 0.0, "d21_f1": 0.0, "d21_iou": 0.0}
    batches = 0

    for X, Y in loader:
        X, Y = to_device((X, Y), device)
        X = sanitize_tensor(normalize_cloud(sanitize_tensor(X)))
        logits = sanitize_tensor(model(X))
        Y = torch.clamp(Y, 0, num_classes - 1)
        loss = criterion(logits.transpose(2, 1), Y)

        loss_sum += float(loss.detach().cpu())
        cm += confusion_matrix(logits, Y, num_classes)
        d21m = d21_metrics(logits, Y, d21_id)
        for k in d21: d21[k] += d21m[k]
        batches += 1

    macro = macro_from_cm(cm)
    for k in d21: d21[k] /= max(1, batches)
    return loss_sum / max(1, batches), {**macro, **d21}


# ==============================================================
# === Logging de historiales ===================================
# ==============================================================

def update_history(history: Dict[str, List[float]], prefix: str, stats: Dict[str, float]):
    for k, v in stats.items():
        history.setdefault(f"{prefix}_{k}", []).append(float(v))


def print_epoch(ep: int, epochs: int, tr_loss: float, va_loss: float,
                va_stats: Dict[str, float]) -> None:
    msg = (
        f"[Ep {ep:03d}/{epochs}] "
        f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
        f"acc={va_stats['acc']:.3f}  f1={va_stats['f1']:.3f}  "
        f"iou={va_stats['iou']:.3f}  d21_f1={va_stats['d21_f1']:.3f}"
    )
    print(msg)


def save_history_csv(history: Dict[str, List[float]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    any_key = next(iter(history.keys()))
    T = len(history[any_key]) if any_key else 0
    keys = sorted(history.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for ep in range(T):
            row = [ep + 1] + [history[k][ep] if ep < len(history[k]) else "" for k in keys]
            w.writerow(row)


def save_run_summary(run_dir: Path, args: Dict[str, Any], num_classes: int,
                     n_params: int, best_val: float, last_epoch: int,
                     te_loss: float, te_stats: Dict[str, float],
                     history: Dict[str, List[float]]):
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json({
        "args": args,
        "num_classes": num_classes,
        "params": int(n_params),
        "best_val_loss": float(best_val),
        "last_epoch": int(last_epoch),
        "test": te_stats,
        "history_keys": list(history.keys())
    }, run_dir / "summary.json")
    save_history_csv(history, run_dir / "history.csv")


# ==============================================================
# === MAIN =====================================================
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PointNet++ mejorado (SPFE + WSLFA)")
    parser.add_argument("--data_dir", required=True,
                        help="Carpeta con X_*.npz / Y_*.npz (+ artifacts/)")
    parser.add_argument("--out_dir", required=True,
                        help="Carpeta base para runs (se crea subcarpeta por timestamp)")
    # Hparams
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--nsample", type=int, default=32,
                        help="Número de vecinos por centro en WSLFA")
    # Loss / labels
    parser.add_argument("--ignore_index", type=int, default=None,
                        help="ID a ignorar en la pérdida (p.ej. fondo)")
    parser.add_argument("--d21_id", type=int, default=21,
                        help="ID (remapeado) del diente 21 para métricas específicas")
    # Scheduler
    parser.add_argument("--use_cosine", action="store_true",
                        help="Usar CosineAnnealingLR (T_max=epochs)")
    # Eval
    parser.add_argument("--eval_best", action="store_true",
                        help="Evaluar test con el mejor checkpoint (best.pt)")

    args = parser.parse_args()

    # Semilla y dispositivo
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # DataLoaders
    loaders = make_loaders(Path(args.data_dir),
                           batch_size=args.batch_size,
                           num_workers=args.num_workers)

    # num_classes desde Y_train
    Ytr = np.load(Path(args.data_dir) / "Y_train.npz")["Y"]
    num_classes = int(np.max(Ytr)) + 1
    print(f"[INFO] num_classes={num_classes}")

    # Modelo
    model = ImprovedPointNet2Seg(num_classes=num_classes,
                                 nsample=args.nsample).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] pointnetpp_improved  params={n_params/1e6:.2f}M")

    # Pérdida + weights
    artifacts = Path(args.data_dir) / "artifacts"
    class_w = load_class_weights(artifacts, num_classes)
    if class_w is not None:
        class_w = class_w.to(device)
        print("[INFO] Usando class_weights.json")

    if args.ignore_index is not None:
        criterion = nn.CrossEntropyLoss(weight=class_w,
                                        ignore_index=args.ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)
                 if args.use_cosine else None)

    # Salida
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"pointnetpp_improved_{stamp}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {run_dir}")

    stopper = EarlyStopping(patience=args.patience, delta=1e-4,
                            ckpt_dir=run_dir / "checkpoints")

    history: Dict[str, List[float]] = {}
    best_val = float("inf")
    last_epoch = 0

    # Entrenamiento
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_stats = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            device, num_classes, args.d21_id
        )
        va_loss, va_stats = evaluate(
            model, loaders["val"], criterion,
            device, num_classes, args.d21_id
        )

        if scheduler is not None:
            scheduler.step()

        update_history(history, "train", {"loss": tr_loss, **tr_stats})
        update_history(history, "val",   {"loss": va_loss, **va_stats})
        print_epoch(ep, args.epochs, tr_loss, va_loss, va_stats)

        improved = stopper(va_loss, model, ep)
        if improved:
            best_val = va_loss
        if stopper.early_stop:
            print("[EARLY] Parada temprana activada.")
            last_epoch = ep
            break

        last_epoch = ep

    # Guardar modelo final
    torch.save({"model": model.state_dict(), "epoch": last_epoch},
               run_dir / "checkpoints" / "final_model.pt")

    # Curvas + resumen (sin test todavía)
    plot_curves(history, run_dir, "pointnetpp_improved")
    save_run_summary(run_dir, vars(args), num_classes, n_params,
                     best_val, last_epoch,
                     te_loss=float("nan"), te_stats={},
                     history=history)

    # Evaluación en test
    ckpt_to_eval = run_dir / "checkpoints" / (
        "best.pt"
        if args.eval_best and (run_dir / "checkpoints" / "best.pt").exists()
        else "final_model.pt"
    )
    print(f"[EVAL] Cargando {ckpt_to_eval.name} para evaluación en test.")
    state = torch.load(ckpt_to_eval, map_location=device)
    model.load_state_dict(state["model"])

    te_loss, te_stats = evaluate(
        model, loaders["test"], criterion,
        device, num_classes, args.d21_id
    )
    print(
        f"[TEST] loss={te_loss:.4f}  acc={te_stats['acc']:.3f}  "
        f"f1={te_stats['f1']:.3f}  iou={te_stats['iou']:.3f}  "
        f"d21_f1={te_stats['d21_f1']:.3f}"
    )

    # Actualizar resumen con test
    save_run_summary(run_dir, vars(args), num_classes, n_params,
                     best_val, last_epoch,
                     te_loss=te_loss, te_stats=te_stats,
                     history=history)

    print(f"[DONE] Resultados guardados en: {run_dir}")


if __name__ == "__main__":
    main()
