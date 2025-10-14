#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inferencia UFRN con PointNet (upper+lower) y evaluación vs CAD (upper_rec_21).
- Lee nubes normalizadas (8192 pts) desde processed_struct/
- Input al modelo: upper_full + lower_full (dos opciones: concat o dos ramas)
- GT: upper_rec_21
- Output esperado del modelo: nube Nx3 (reconstrucción upper con 21 removido)
- Métricas: Chamfer Distance, Hausdorff Distance
- Guarda predicciones .npy (+ opcional .ply) y CSV de métricas
- (Opcional) empaqueta resultados en .zip
"""

import os, sys, json, shutil, zipfile, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

try:
    import trimesh as tm  # opcional para exportar .ply
except Exception:
    tm = None

# ---------------------------
# Utilidades de E/S & métricas
# ---------------------------
def load_cloud(p: Path, n_points: int = 8192) -> np.ndarray:
    """Carga point_cloud.npy, asegura (N,3); si N!=n_points, re-muestrea con reposición."""
    arr = np.load(p)  # (N,3) float32
    if arr.shape[0] != n_points:
        idx = np.random.choice(arr.shape[0], size=n_points, replace=(arr.shape[0] < n_points))
        arr = arr[idx]
    return arr.astype(np.float32)

def save_cloud_ply(pts: np.ndarray, path: Path):
    if tm is None:
        return
    pc = tm.PointCloud(pts)
    pc.export(path)

@torch.no_grad()
def chamfer_distance_np(pcd1: np.ndarray, pcd2: np.ndarray) -> float:
    """Chamfer simétrico (promedios de NN dist^2)."""
    t1 = cKDTree(pcd1); d1, _ = t1.query(pcd2)
    t2 = cKDTree(pcd2); d2, _ = t2.query(pcd1)
    return float(np.mean(d1**2) + np.mean(d2**2))

@torch.no_grad()
def hausdorff_distance_np(pcd1: np.ndarray, pcd2: np.ndarray) -> float:
    """Hausdorff (máximo NN)."""
    t1 = cKDTree(pcd1); d1, _ = t1.query(pcd2)
    t2 = cKDTree(pcd2); d2, _ = t2.query(pcd1)
    return float(max(np.max(d1), np.max(d2)))

# ---------------------------
# Dataset: detecta pacientes válidos
# ---------------------------
def list_patients(struct_root: Path):
    """Devuelve lista de patient_id que tengan upper_full, lower_full y upper_rec_21."""
    U = struct_root / "upper"
    L = struct_root / "lower"
    patients = []
    for x in U.glob("paciente_*_full"):
        pid = x.name.replace("_full", "")
        if (U / f"{pid}_full" / "point_cloud.npy").exists() and \
           (U / f"{pid}_rec_21" / "point_cloud.npy").exists() and \
           (L / f"{pid}_full" / "point_cloud.npy").exists():
            patients.append(pid)
    patients = sorted(set(patients))
    return patients

# ---------------------------
# Modelo: wrapper flexible
# ---------------------------
class TwoBranchConcat(nn.Module):
    """
    En caso de no tener tu clase exacta, este wrapper concatena upper+lower a (N*2,3)
    y pasa por un PointNet simple para reconstrucción (demo). 
    >>> En producción, IMPORTA tu modelo real y úsalo en su lugar.
    """
    def __init__(self, out_points=8192):
        super().__init__()
        self.out_points = out_points
        # Encoder simplificado (placeholder). Reemplaza por tu PointNet real.
        self.enc = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        # Decoder a Nx3 (placeholder). Reemplaza por tu decoder real.
        self.dec = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, self.out_points*3)
        )

    def forward(self, upper: torch.Tensor, lower: torch.Tensor):
        """
        upper: (B, N, 3), lower: (B, N, 3)
        return pred: (B, N, 3)
        """
        B, N, _ = upper.shape
        x = torch.cat([upper, lower], dim=1)        # (B, 2N, 3)
        x = x.transpose(1, 2).contiguous()          # (B, 3, 2N)
        feat = self.enc(x).squeeze(-1)              # (B, 256)
        out = self.dec(feat).view(B, -1, 3)         # (B, N, 3)
        # si N difiere, remuestrea
        if out.shape[1] != N:
            idx = torch.randint(0, out.shape[1], (B, N), device=out.device)
            out = torch.gather(out, 1, idx.unsqueeze(-1).expand(B, N, 3))
        return out

def load_model(args, device):
    """
    Opción A (recomendada): importa tu clase y carga checkpoint
        from your_pkg.model import YourPointNet
        model = YourPointNet(...)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    Opción B (fallback/demo): usa TwoBranchConcat con checkpoint opcional de este wrapper.
    """
    if args.model_import and args.model_class:
        module = __import__(args.model_import, fromlist=[args.model_class])
        ModelClass = getattr(module, args.model_class)
        model = ModelClass(**json.loads(args.model_kwargs))
    else:
        model = TwoBranchConcat(out_points=args.n_points)

    model = model.to(device)
    model.eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        # soporta ckpt como dict con 'state_dict' o directamente state_dict
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[CKPT] Cargado: {args.checkpoint}")
    else:
        if args.checkpoint:
            print(f"[WARN] No se encontró checkpoint: {args.checkpoint} (se usará pesos aleatorios)")
    return model

# ---------------------------
# Inferencia por paciente
# ---------------------------
@torch.no_grad()
def infer_patient(pid: str, struct_root: Path, out_pred_root: Path, model, device, n_points=8192, export_ply=False):
    up_full = load_cloud(struct_root / "upper" / f"{pid}_full" / "point_cloud.npy", n_points)
    lo_full = load_cloud(struct_root / "lower" / f"{pid}_full" / "point_cloud.npy", n_points)
    gt_up21 = load_cloud(struct_root / "upper" / f"{pid}_rec_21" / "point_cloud.npy", n_points)

    # tensors
    u = torch.from_numpy(up_full).unsqueeze(0).to(device)  # (1,N,3)
    l = torch.from_numpy(lo_full).unsqueeze(0).to(device)  # (1,N,3)

    pred = model(u, l).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (N,3)

    # métricas
    cd = chamfer_distance_np(pred, gt_up21)
    hd = hausdorff_distance_np(pred, gt_up21)

    # guardar pred
    pdir = out_pred_root / pid
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / "upper_rec_21_pred.npy", pred)
    # opcional .ply
    if export_ply:
        save_cloud_ply(pred, pdir / "upper_rec_21_pred.ply")

    # copias de GT para inspección rápida (opcionales)
    # np.save(pdir / "upper_rec_21_gt.npy", gt_up21)

    return {"patient_id": pid, "chamfer": cd, "hausdorff": hd}

# ---------------------------
# Empaquetado .zip de predicciones
# ---------------------------
def export_preds_zip(out_pred_root: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_pred_root):
            for f in files:
                abspath = Path(root) / f
                arcname = abspath.relative_to(out_pred_root)
                zf.write(abspath, arcname)
    print(f"[EXPORT] ZIP de predicciones escrito en: {zip_path}")

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True, help="Raíz data/UFRN")
    ap.add_argument("--struct_rel", default="processed_struct", help="Carpeta con nubes normalizadas")
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint", default=None, help="Ruta al .pth/.pt")
    ap.add_argument("--model_import", default=None, help="p.ej. 'models.pointnet_biarch'")
    ap.add_argument("--model_class", default=None, help="p.ej. 'PointNetBiArchRecon'")
    ap.add_argument("--model_kwargs", default="{}", help='JSON con kwargs del modelo')
    ap.add_argument("--out_pred", default="preds_pointnet", help="Carpeta donde se guardan predicciones")
    ap.add_argument("--metrics_csv", default="ufrn_metrics.csv")
    ap.add_argument("--export_zip", default=None, help="(Opcional) ZIP con todas las predicciones")
    ap.add_argument("--export_ply", action="store_true", help="Guarda .ply además de .npy")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    struct_root = Path(args.ufrn_root) / args.struct_rel
    out_pred_root = Path(args.ufrn_root) / args.out_pred
    out_pred_root.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.ufrn_root) / args.metrics_csv

    patients = list_patients(struct_root)
    if len(patients) == 0:
        print("[ERR] No se encontraron pacientes válidos en processed_struct/.")
        sys.exit(1)
    print(f"[INFO] Pacientes para inferencia: {len(patients)}")

    model = load_model(args, device)

    rows = []
    for pid in patients:
        row = infer_patient(pid, struct_root, out_pred_root, model, device,
                            n_points=args.n_points, export_ply=args.export_ply)
        rows.append(row)
        print(f"[OK] {pid}: CD={row['chamfer']:.6f}  HD={row['hausdorff']:.6f}")

    # guardar CSV
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("patient_id")
    df.to_csv(metrics_csv, index=False)
    print(f"[CSV] Métricas escritas en: {metrics_csv}")

    # resumen
    cd_mean = float(df["chamfer"].mean())
    hd_mean = float(df["hausdorff"].mean())
    print(f"[SUMMARY] Chamfer(mean)={cd_mean:.6f}  Hausdorff(mean)={hd_mean:.6f}")

    # ZIP de predicciones
    if args.export_zip:
        export_preds_zip(out_pred_root, Path(args.export_zip))

if __name__ == "__main__":
    main()
