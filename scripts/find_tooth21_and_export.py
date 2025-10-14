#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_tooth21_and_export.py  (con export_zip y CSV)

- Detecta región faltante (diente 21) comparando upper_full vs upper_rec_21.
- Exporta PLY ASCII válidos: *_missing_blue.ply y *_overlay_green.ply.
- Sanea .ply mal nombrados (si eran PNG, renombra a .png; si no es válido, borra).
- Genera summary.json por paciente y summary.csv global.
- Opción --export_zip para comprimir todo el --out_dir al final.

Uso:
  python scripts/find_tooth21_and_export.py \
    --ufrn_root /path/UFRN \
    --struct_rel processed_struct \
    --patient_id all \
    --model_path runs_grid/pointnetpp_improved_8k_s42/final.keras \
    --n_points 8192 \
    --removed_thresh 0.02 \
    --out_dir diag_tooth21 \
    --export_zip /path/ufrn_diag_tooth21.zip
"""

import os, json, argparse, csv, zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf

# ------------------------------------------------------------
# 1) SHIMS de PN++ (usa tus capas reales, sin módulos extra)
# ------------------------------------------------------------
_STRIP_KEYS = ("trainable", "dtype", "name")

try:
    from scripts.train_pointnetpp_improved_d21 import SetAbstraction as _SA_Base
    from scripts.train_pointnetpp_improved_d21 import FeaturePropagation as _FP_Base
except Exception as e:
    raise RuntimeError(
        "No pude importar scripts.train_pointnetpp_improved_d21.\n"
        "Asegúrate de tener PYTHONPATH en la raíz del repo (export PYTHONPATH=\"$PWD:$PYTHONPATH\").\n"
        f"Detalle: {e}"
    )

@tf.keras.utils.register_keras_serializable(package="CustomPN2", name="SetAbstraction")
class SetAbstraction(_SA_Base):
    def __init__(self, *args, **kwargs):
        clean = {k: v for k, v in kwargs.items() if k not in _STRIP_KEYS}
        super().__init__(*args, **clean)
    @classmethod
    def from_config(cls, config):
        clean = {k: v for k, v in config.items() if k not in _STRIP_KEYS}
        return cls(**clean)

@tf.keras.utils.register_keras_serializable(package="CustomPN2", name="FeaturePropagation")
class FeaturePropagation(_FP_Base):
    def __init__(self, *args, **kwargs):
        clean = {k: v for k, v in kwargs.items() if k not in _STRIP_KEYS}
        super().__init__(*args, **clean)
    @classmethod
    def from_config(cls, config):
        clean = {k: v for k, v in config.items() if k not in _STRIP_KEYS}
        return cls(**clean)

# ------------------------------------------------------------
# 2) Utilidades nubes/PLY (sin scipy/matplotlib)
# ------------------------------------------------------------
PNG_SIG = b"\x89PNG\r\n\x1a\n"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_cloud(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    pts = np.load(path).astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Esperaba (N,3) en {path}, obtuve {pts.shape}")
    return pts

def sample_to_n(pts: np.ndarray, n_points: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    if n == n_points:
        return pts
    if n > n_points:
        idx = rng.choice(n, size=n_points, replace=False)
        return pts[idx]
    reps = int(np.ceil(n_points / n))
    tiled = np.tile(pts, (reps, 1))
    idx = rng.choice(tiled.shape[0], size=n_points, replace=False)
    return tiled[idx]

def save_ply(points: np.ndarray, colors: np.ndarray, path: Path):
    ensure_dir(path.parent)
    N = points.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors.astype(np.uint8)):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def chunked_nn_dists(A: np.ndarray, B: np.ndarray, chunk: int = 2048) -> np.ndarray:
    B2 = np.sum(B * B, axis=1)  # (M,)
    d_min = np.full((A.shape[0],), np.inf, dtype=np.float32)
    for i in range(0, A.shape[0], chunk):
        a = A[i:i+chunk]                         # (c,3)
        a2 = np.sum(a*a, axis=1, keepdims=True)  # (c,1)
        G = a @ B.T                              # (c,M)
        d2 = a2 + B2[None, :] - 2.0 * G          # (c,M)
        d2 = np.maximum(d2, 0.0)
        d = np.sqrt(np.min(d2, axis=1))
        d_min[i:i+chunk] = d
    return d_min

def sanitize_legacy_files(out_p: Path):
    for fp in out_p.glob("*.ply"):
        try:
            with open(fp, "rb") as f:
                head = f.read(16)
            if head.startswith(b"ply\n"):  # PLY ASCII correcto
                continue
            if head.startswith(PNG_SIG):
                newp = fp.with_suffix(".png")
                fp.rename(newp)
                print(f"[FIX] {fp.name} era PNG -> {newp.name}")
            else:
                fp.unlink(missing_ok=True)
                print(f"[FIX] {fp.name} no era PLY/PNG válido -> eliminado")
        except Exception as e:
            print(f"[FIX] No pude sanear {fp.name}: {e}")

# ------------------------------------------------------------
# 3) Localización robusta de archivos en processed_struct
# ------------------------------------------------------------
def find_upper_full(struct_root: Path, pid: str) -> Path:
    candidates = [
        struct_root/"upper"/f"{pid}_full"/"point_cloud.npy",
        struct_root/"upper"/f"{pid}_upper_full"/"point_cloud.npy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No encontré upper_full para {pid}. Probé: {candidates}")

def find_upper_rec21(struct_root: Path, pid: str) -> Path:
    candidates = [
        struct_root/"upper"/f"{pid}_upper_rec_21"/"point_cloud.npy",
        struct_root/"upper"/f"{pid}_rec_21"/"point_cloud.npy",
        struct_root/"upper"/f"{pid}_21"/"point_cloud.npy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No encontré upper_rec_21 para {pid}. Probé: {candidates}")

# ------------------------------------------------------------
# 4) Carga de modelo
# ------------------------------------------------------------
def load_model_safe(path: Path):
    custom = {"SetAbstraction": SetAbstraction, "FeaturePropagation": FeaturePropagation}
    return tf.keras.models.load_model(str(path), compile=False, custom_objects=custom)

# ------------------------------------------------------------
# 5) Pipeline por paciente
# ------------------------------------------------------------
def run_for_patient(pid: str, struct_root: Path, model_path: Path,
                    n_points: int, removed_thresh: float, out_dir: Path) -> dict:
    print(f"[INFO] Procesando {pid} …")
    upper_full = find_upper_full(struct_root, pid)
    upper_rec21 = find_upper_rec21(struct_root, pid)

    X_full = sample_to_n(load_cloud(upper_full), n_points)
    X_rec  = sample_to_n(load_cloud(upper_rec21), n_points)

    # Distancia A->B mínima, para decidir “faltante”
    d_full_to_rec = chunked_nn_dists(X_full, X_rec, chunk=2048)
    missing_mask  = (d_full_to_rec > removed_thresh)
    frac_missing  = float(np.mean(missing_mask))

    # (Pred del modelo — opcional para colores extra)
    def normalize(pts):
        c = pts.mean(0, keepdims=True)
        pts = pts - c
        r = np.linalg.norm(pts, axis=1).max()
        return (pts / r).astype(np.float32) if r > 0 else pts.astype(np.float32)

    try:
        model = load_model_safe(model_path)
        _ = model.predict(normalize(X_full)[None, ...], verbose=0)  # “warm-up”
    except Exception as e:
        print(f"[WARN] No pude predecir con el modelo (se omite overlay por probs): {e}")

    # Colores
    gray  = np.array([180,180,180], dtype=np.uint8)
    blue  = np.array([ 30,144,255], dtype=np.uint8)  # faltante
    green = np.array([  0,200, 70], dtype=np.uint8)  # overlay

    col_full = np.tile(gray, (n_points,1))
    col_full[missing_mask] = blue
    col_overlay = col_full.copy()
    col_overlay[missing_mask] = green

    # Guardado
    out_p = out_dir/pid/"ply"
    ensure_dir(out_p)
    save_ply(X_full, col_full,    out_p/"upper_full_missing_blue.ply")
    save_ply(X_full, col_overlay, out_p/"upper_full_overlay_green.ply")
    sanitize_legacy_files(out_p)

    meta = {
        "patient_id": pid,
        "upper_full": str(upper_full),
        "upper_rec_21": str(upper_rec21),
        "n_points": int(n_points),
        "removed_thresh": float(removed_thresh),
        "frac_missing_region": float(frac_missing),
    }
    (out_dir/pid/"summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[DONE] {pid}  missing_frac={frac_missing:.3f}  -> {out_dir/pid}")
    return meta

# ------------------------------------------------------------
# 6) ZIP helper
# ------------------------------------------------------------
def zip_dir(root: Path, zip_path: Path):
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            zf.write(p, p.relative_to(root))
    print(f"[EXPORT] ZIP escrito en: {zip_path}")

# ------------------------------------------------------------
# 7) CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--struct_rel", default="processed_struct")
    ap.add_argument("--patient_id", required=True, help="paciente_XX o 'all'")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--removed_thresh", type=float, default=0.02)
    ap.add_argument("--out_dir", default="diag_tooth21")
    ap.add_argument("--export_zip", default=None, help="Ruta .zip para comprimir --out_dir (opcional)")
    args = ap.parse_args()

    ufrn_root   = Path(args.ufrn_root)
    struct_root = ufrn_root / args.struct_rel
    model_path  = Path(args.model_path)
    out_dir     = Path(args.out_dir)
    ensure_dir(out_dir)

    # Pacientes
    if args.patient_id.lower() != "all":
        pids = [args.patient_id]
    else:
        csv_path = ufrn_root/"ufrn_por_paciente.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"No existe {csv_path}; usa 'all' solo si ya generaste ese resumen.")
        rows = [l.strip().split(",")[0] for l in csv_path.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
        pids = sorted(set(rows))

    print(f"[INFO] Pacientes: {len(pids)}")
    all_rows = []
    for pid in pids:
        try:
            row = run_for_patient(pid, struct_root, model_path, args.n_points, args.removed_thresh, out_dir)
            all_rows.append(row)
        except Exception as e:
            print(f"[WARN] {pid}: {e}")

    # CSV global
    csv_out = out_dir/"summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id","frac_missing_region","n_points","removed_thresh","upper_full","upper_rec_21"])
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "patient_id": r["patient_id"],
                "frac_missing_region": r["frac_missing_region"],
                "n_points": r["n_points"],
                "removed_thresh": r["removed_thresh"],
                "upper_full": r["upper_full"],
                "upper_rec_21": r["upper_rec_21"],
            })
    print(f"[CSV] Resumen en: {csv_out}")

    # ZIP opcional
    if args.export_zip:
        zip_dir(out_dir, Path(args.export_zip))

if __name__ == "__main__":
    main()
