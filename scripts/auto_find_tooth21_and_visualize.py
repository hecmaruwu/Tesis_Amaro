#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_find_tooth21_and_visualize.py

Descubre automáticamente qué índice de clase de tu modelo corresponde al diente 21
comparando upper_full vs upper_rec_21, y genera visualizaciones claras.

• No requiere label_map.json ni SciPy.
• Nearest neighbor por bloques con NumPy.
• Salida: PNGs (si hay matplotlib) y PLYs coloreados (si hay trimesh).

Ejemplo:
python scripts/auto_find_tooth21_and_visualize.py \
  --ufrn_root /home/htaucare/Tesis_dientes_original/data/UFRN \
  --struct_rel processed_struct \
  --patient_id paciente_19 \
  --model_path runs_grid/pointnetpp_improved_8k_s42/final.keras \
  --co_modules "scripts.infer_utils,scripts.train_models,scripts.train_pointnetpp_improved_d21" \
  --co_alias '{"SetAbstraction":"SetAbstraction","FeaturePropagation":"FeaturePropagation"}' \
  --n_points 8192 \
  --out_dir diag_tooth21
"""

import argparse, json, sys, types, importlib, inspect
from pathlib import Path
import numpy as np
import tensorflow as tf

# Opcionales
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

try:
    import trimesh as tm
    _HAS_TM = True
except Exception:
    _HAS_TM = False


# ---------- utilidades ----------
def load_cloud(p: Path) -> np.ndarray:
    a = np.load(p)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"nube inválida {p} shape={a.shape}")
    return a

def normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0, keepdims=True)
    pts = pts - c
    r = np.linalg.norm(pts, axis=1).max()
    return pts / (r if r > 0 else 1.0)

def sample_n(pts: np.ndarray, n: int) -> np.ndarray:
    idx = np.random.choice(pts.shape[0], size=n, replace=(pts.shape[0] < n))
    return pts[idx]

def min_sqdist_chunked(A: np.ndarray, B: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """
    Para cada punto de A, devuelve la distancia mínima (euclídea) a B.
    Implementación por bloques (sin SciPy).
    """
    # Precomputo normas de B para vectorizar
    B2 = (B**2).sum(axis=1, keepdims=True)  # (MB,1)
    out = np.empty((A.shape[0],), dtype=np.float32)
    for s in range(0, A.shape[0], chunk):
        e = min(A.shape[0], s + chunk)
        a = A[s:e]                                    # (m,3)
        a2 = (a**2).sum(axis=1, keepdims=True)        # (m,1)
        # dist^2 = |a|^2 + |b|^2 - 2 a·b  -> (m,MB)
        d2 = a2 + B2.T - 2.0 * (a @ B.T)
        # numéricamente, puede dar negativos muy pequeños
        d2 = np.maximum(d2, 0.0)
        out[s:e] = np.sqrt(d2.min(axis=1))
    return out

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def plot_overlay(out_png: Path,
                 up_full: np.ndarray,
                 removed_mask: np.ndarray,
                 pred_mask21_on_up: np.ndarray):
    if not _HAS_MPL: return False
    up_n = normalize_unit_sphere(up_full)
    fig = plt.figure(figsize=(11, 3.5), dpi=140)

    # 1) upper_full (gris)
    ax1 = fig.add_subplot(1,3,1, projection='3d')
    ax1.scatter(up_n[:,0], up_n[:,1], up_n[:,2], s=1, c="0.65")
    ax1.set_title("Upper FULL (gris)")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

    # 2) región faltante estimada (azul)
    ax2 = fig.add_subplot(1,3,2, projection='3d')
    keep = ~removed_mask
    if keep.any():
        ax2.scatter(up_n[keep,0], up_n[keep,1], up_n[keep,2], s=1, c="0.85", alpha=0.45)
    if removed_mask.any():
        ax2.scatter(up_n[removed_mask,0], up_n[removed_mask,1], up_n[removed_mask,2], s=3, c="tab:blue")
    ax2.set_title("Región faltante (estimada)")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])

    # 3) predicción clase-21 (verde) sobre upper_full
    ax3 = fig.add_subplot(1,3,3, projection='3d')
    other = ~pred_mask21_on_up
    if other.any():
        ax3.scatter(up_n[other,0], up_n[other,1], up_n[other,2], s=1, c="0.85")
    if pred_mask21_on_up.any():
        ax3.scatter(up_n[pred_mask21_on_up,0], up_n[pred_mask21_on_up,1], up_n[pred_mask21_on_up,2], s=3, c="tab:green")
    ax3.set_title("Pred. diente 21 (verde)")
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

    plt.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png)
    plt.close(fig)
    return True

def export_plys(out_dir: Path,
                up_full: np.ndarray,
                removed_mask: np.ndarray,
                pred_mask21_on_up: np.ndarray):
    ensure_dir(out_dir)
    if not _HAS_TM:
        # fallback: guardamos npy
        np.save(out_dir/"upper_full.npy", up_full)
        np.save(out_dir/"removed_mask.npy", removed_mask.astype(np.uint8))
        np.save(out_dir/"pred21_mask.npy", pred_mask21_on_up.astype(np.uint8))
        return

    colors = np.full((up_full.shape[0], 4), [180,180,180,255], dtype=np.uint8)   # gris
    # azul región “removida”
    colors[removed_mask] = [30, 120, 255, 255]
    # verde pred 21
    colors[pred_mask21_on_up] = [40, 200, 80, 255]

    tm.points.PointCloud(up_full, colors=colors).export(out_dir/"upper_full_overlay.ply")


# ---------- carga de modelo con custom_objects ----------
def load_model_custom(path: str, co_modules: list[str], co_alias: dict[str,str]):
    custom = {}

    # evitar fallos por falta de matplotlib al importar módulos que lo usan
    def _inject_dummy(names=("matplotlib","matplotlib.pyplot")):
        for n in names:
            if n not in sys.modules:
                sys.modules[n] = types.ModuleType(n)

    def _collect(modname):
        _inject_dummy()
        m = importlib.import_module(modname)
        return {k:v for k,v in inspect.getmembers(m)
                if (inspect.isclass(v) or inspect.isfunction(v)) and not k.startswith("_")}

    for m in co_modules:
        try:
            custom.update(_collect(m))
            print(f"[INFO] {m}: {sum(1 for _ in custom)} símbolos")
        except Exception as e:
            print(f"[WARN] no pude importar {m} -> {e}")

    # resolver alias
    for alias, real in co_alias.items():
        if real in custom:
            custom[alias] = custom[real]
        else:
            cands = [k for k in custom if real.lower() in k.lower()]
            if len(cands) == 1:
                custom[alias] = custom[cands[0]]

    model = tf.keras.models.load_model(path, compile=False, custom_objects=custom)
    return model


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--struct_rel", default="processed_struct")
    ap.add_argument("--patient_id", required=True)  # ej: paciente_19
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--co_modules", default="")
    ap.add_argument("--co_alias", default="")
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--removed_thresh", type=float, default=0.02,
                    help="Umbral (en espacio normalizado) para marcar puntos ausentes vs rec_21")
    ap.add_argument("--out_dir", default="diag_tooth21")
    args = ap.parse_args()

    # cargar modelo
    co_modules = [x.strip() for x in args.co_modules.split(",") if x.strip()]
    co_alias = json.loads(args.co_alias) if args.co_alias.strip() else {}
    model = load_model_custom(args.model_path, co_modules, co_alias)

    root = Path(args.ufrn_root) / args.struct_rel
    up_full = load_cloud(root/"upper"/f"{args.patient_id}_full"/"point_cloud.npy")
    lo_full = load_cloud(root/"lower"/f"{args.patient_id}_full"/"point_cloud.npy")
    rec21   = load_cloud(root/"upper"/f"{args.patient_id}_rec_21"/"point_cloud.npy")

    # entrada con contexto (upper + lower) → respetar tamaño
    nU = args.n_points // 2
    nL = args.n_points - nU
    up_in = sample_n(up_full, nU)
    lo_in = sample_n(lo_full, nL)
    x = np.concatenate([up_in, lo_in], axis=0)
    # normalización igual que en training
    x = normalize_unit_sphere(x)

    # predicción multiclase
    probs = model.predict(x[None,...], verbose=0)  # (1,P,C)
    preds = np.argmax(probs, axis=-1)[0]          # (P,)

    # sólo en la parte upper de la entrada
    preds_up = preds[:nU]          # (nU,)
    up_used  = x[:nU]              # normalizada, pero basta para máscara relativa

    # Encontrar región “removida”: puntos de up_used que no aparecen en rec_21 (distancia > umbral)
    # Usamos rec_21 en el mismo espacio de normalización: normalizamos también
    rec21_n = normalize_unit_sphere(rec21.copy())
    # ¡Ojo! up_used ya está normalizada por la nube combinada; para robustez normalizamos up_full sola:
    # Pero como solo queremos máscara aproximada, usamos up_used como está.
    d = min_sqdist_chunked(up_used, rec21_n, chunk=4096)   # (nU,)
    removed_mask = d > float(args.removed_thresh)

    # Clase mayoritaria en esa región → índice del "tooth21" en tu modelo
    if removed_mask.any():
        uniq, counts = np.unique(preds_up[removed_mask], return_counts=True)
        maj_idx = int(uniq[np.argmax(counts)])
        print(f"[INFO] Clase mayoritaria en región ausente = {maj_idx} (estimada como diente 21)")
    else:
        maj_idx = int(np.bincount(preds_up).argmax())
        print("[WARN] No se detectó región ausente clara; usando clase mayoritaria global de upper.")

    # máscara de predicción 21 sobre upper
    pred_mask21_on_up = (preds_up == maj_idx)

    # Salidas
    out_dir = Path(args.out_dir) / args.patient_id
    ensure_dir(out_dir)

    # PNGs
    ok = plot_overlay(out_dir/"overlay_tooth21.png", up_full=up_in, 
                      removed_mask=removed_mask, pred_mask21_on_up=pred_mask21_on_up)
    if ok:
        print(f"[OUT] {out_dir/'overlay_tooth21.png'}")
    else:
        print("[NOTE] matplotlib no disponible: se omite PNG.")

    # PLY coloreado
    export_plys(out_dir/"ply", up_full=up_in,
                removed_mask=removed_mask, pred_mask21_on_up=pred_mask21_on_up)
    print(f"[DONE] Guardadas salidas en {out_dir}")

if __name__ == "__main__":
    main()
