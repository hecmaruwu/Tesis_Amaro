#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inferencia UFRN con TensorFlow (segmentación punto a punto) — SIN SciPy,
con soporte robusto de custom_objects (múltiples módulos, alias) y PARCHE
para clases que no aceptan kwargs estándar de Keras (trainable, name, dtype).

- Entrada por paciente desde processed_struct:
    upper/paciente_X_full/point_cloud.npy
    lower/paciente_X_full/point_cloud.npy
    upper/paciente_X_rec_21/point_cloud.npy   (CAD objetivo)
- Opción de usar contexto inferior: se arma (8192,3) como 4096 upper + 4096 lower
  (o mitad/mitad de n_points), de modo que el modelo reciba exactamente P=n_points.
- El modelo devuelve (B,P,C). Se elimina la clase objetivo SOLO en la porción upper.
- Métricas Chamfer/Hausdorff en NumPy por bloques (sin SciPy).
- Guarda pred (.npy, opcional .ply), CSV de métricas y ZIP opcional.

Ejemplo:

PYTHONPATH=. python scripts/infer_ufrn_tf_seg.py \
  --ufrn_root /home/usuario/Tesis_dientes_original/data/UFRN \
  --struct_rel processed_struct \
  --model_path runs_grid/pointnetpp_improved_8k_s42/final.keras \
  --co_modules "scripts.infer_utils,scripts.train_models,scripts.train_pointnetpp_improved_d21" \
  --co_alias '{"SetAbstraction":"SetAbstraction","FeaturePropagation":"FeaturePropagation"}' \
  --co_patch "SetAbstraction,FeaturePropagation" \
  --tooth_class 21 \
  --n_points 8192 \
  --use_lower_context \
  --out_pred preds_pointnetpp_8k_s42 \
  --metrics_csv ufrn_metrics_pointnetpp_8k_s42.csv \
  --export_zip /home/usuario/Tesis_dientes_original/data/UFRN/ufrn_preds_pointnetpp_8k_s42.zip \
  --export_ply \
  --clean
"""

import os
import sys
import json
import argparse
import zipfile
import shutil
import importlib
import inspect
import types
from pathlib import Path
import numpy as np
import tensorflow as tf

# trimesh solo para exportar .ply (opcional)
try:
    import trimesh as tm
except Exception:
    tm = None


# =========================
# Utilidades de E/S
# =========================
def load_cloud(p: Path, n_points: int = 8192) -> np.ndarray:
    """Carga point_cloud.npy; si N!=n_points, remuestrea con reposición."""
    arr = np.load(p)
    if arr.shape[0] != n_points:
        idx = np.random.choice(arr.shape[0], size=n_points, replace=(arr.shape[0] < n_points))
        arr = arr[idx]
    return arr.astype(np.float32)


def save_cloud_ply(pts: np.ndarray, path: Path):
    if tm is None:
        return
    try:
        tm.PointCloud(pts).export(path)
    except Exception:
        pass


# =========================
# Métricas SIN SciPy
# =========================
def _chunked_min_dists(A: np.ndarray, B: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """
    Para cada punto de A calcula la distancia al vecino más cercano en B.
    Implementación vectorizada con bloques (sin SciPy).
    Devuelve vector (len(A),) con distancias (no al cuadrado).
    """
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == 3 and B.shape[1] == 3
    n = A.shape[0]
    mins = np.empty((n,), dtype=np.float32)
    b2 = np.sum(B * B, axis=1, keepdims=True).T  # (1, NB)
    s = 0
    while s < n:
        e = min(s + chunk, n)
        a = A[s:e]                                # (m,3)
        a2 = np.sum(a * a, axis=1, keepdims=True) # (m,1)
        d2 = a2 + b2 - 2.0 * (a @ B.T)            # (m, NB)
        np.maximum(d2, 0.0, out=d2)
        mins[s:e] = np.sqrt(np.min(d2, axis=1))
        s = e
    return mins


def chamfer_distance_np(P: np.ndarray, Q: np.ndarray) -> float:
    """Chamfer simétrico (promedio dist^2 al NN en ambos sentidos)."""
    d1 = _chunked_min_dists(P, Q)
    d2 = _chunked_min_dists(Q, P)
    return float(np.mean(d1**2) + np.mean(d2**2))


def hausdorff_distance_np(P: np.ndarray, Q: np.ndarray) -> float:
    """Hausdorff (máxima distancia al NN en ambos sentidos)."""
    d1 = _chunked_min_dists(P, Q)
    d2 = _chunked_min_dists(Q, P)
    return float(max(float(np.max(d1)), float(np.max(d2))))


# =========================
# Dataset: pacientes válidos
# =========================
def list_patients(struct_root: Path):
    """
    Requiere:
      processed_struct/
        upper/paciente_X_full/point_cloud.npy
        upper/paciente_X_rec_21/point_cloud.npy
        lower/paciente_X_full/point_cloud.npy
    """
    U = struct_root / "upper"
    L = struct_root / "lower"
    pats = []
    for x in U.glob("paciente_*_full"):
        pid = x.name.replace("_full", "")
        if (U / f"{pid}_full" / "point_cloud.npy").exists() and \
           (U / f"{pid}_rec_21" / "point_cloud.npy").exists() and \
           (L / f"{pid}_full" / "point_cloud.npy").exists():
            pats.append(pid)
    return sorted(set(pats))


# =========================
# Export ZIP
# =========================
def export_preds_zip(out_pred_root: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_pred_root):
            for f in files:
                ap = Path(root) / f
                zf.write(ap, ap.relative_to(out_pred_root))
    print(f"[EXPORT] ZIP de predicciones: {zip_path}")


# =========================
# Normalización
# =========================
def normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0, keepdims=True)
    pts = pts - c
    r = np.linalg.norm(pts, axis=1).max()
    return pts / (r if r > 0 else 1.0)


# =========================
# custom_objects (robusto)
# =========================
def _inject_dummy_modules(names=("matplotlib", "matplotlib.pyplot")):
    """
    Inyecta módulos dummy para evitar ImportError al importar tu módulo de capas
    si éste hace 'import matplotlib.pyplot as plt' u otros imports no necesarios.
    """
    for name in names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _collect_objects_from_module(modname: str):
    """Importa módulo y devuelve dict de {nombre: objeto} para clases/funciones públicas."""
    if "." not in sys.path[0]:
        sys.path.insert(0, ".")
    _inject_dummy_modules()
    mod = importlib.import_module(modname)
    objs = {
        name: obj for name, obj in inspect.getmembers(mod)
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not name.startswith("_")
    }
    return objs


def _apply_aliases(obj_pool: dict, alias_map: dict):
    """
    alias_map: {'SetAbstraction':'PointNetSetAbstraction', ...}
    Si rhs no existe literal, intenta resolver por contención de nombre (case-insensitive).
    """
    resolved = {}
    low_names = {k.lower(): k for k in obj_pool.keys()}
    for alias, target in alias_map.items():
        t = target
        if t not in obj_pool:
            key = low_names.get(t.lower())
            if key:
                t = key
            else:
                cand = [k for k in obj_pool if t.lower() in k.lower()]
                if len(cand) == 1:
                    t = cand[0]
                else:
                    print(f"[WARN] Alias '{alias}' no pudo resolverse a '{target}'. Candidates: {cand[:5]}")
                    continue
        resolved[alias] = obj_pool[t]
    return resolved


def _patch_class_kwargs(cls):
    """
    Devuelve una subclase que acepta kwargs extra de Keras y los descarta
    antes de llamar al __init__ original (trainable, name, dtype, ...).
    """
    class Patched(cls):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            for k in ("trainable", "name", "dtype", "dynamic", "autocast", "batch_input_shape"):
                kwargs.pop(k, None)
            super().__init__(*args, **kwargs)
    Patched.__name__ = cls.__name__
    Patched.__qualname__ = cls.__qualname__
    return Patched


def load_model_with_custom_objects(model_path: str,
                                   co_modules: list[str] | None,
                                   co_alias_json: str | None,
                                   co_patch_list: list[str] | None):
    custom_objects = {}

    # 1) Cargar símbolos de los módulos
    if co_modules:
        for m in co_modules:
            m = m.strip()
            if not m:
                continue
            try:
                objs = _collect_objects_from_module(m)
                print(f"[INFO] {m}: registrados {len(objs)} símbolos públicos")
                custom_objects.update(objs)
            except Exception as e:
                print(f"[WARN] No se pudo importar {m}: {e}")

    # 2) Aliases explícitos (nombre_serializado -> nombre_real_en_código)
    if co_alias_json:
        try:
            alias_map = json.loads(co_alias_json)
            alias_objs = _apply_aliases(custom_objects, alias_map)
            custom_objects.update(alias_objs)
            print(f"[INFO] aliases resueltos: {list(alias_objs.keys())}")
        except Exception as e:
            print(f"[WARN] co_alias JSON inválido: {e}")

    # 3) Parchar clases indicadas para aceptar kwargs de Keras
    if co_patch_list:
        for name in co_patch_list:
            key = name.strip()
            if not key:
                continue
            if key in custom_objects and inspect.isclass(custom_objects[key]):
                custom_objects[key] = _patch_class_kwargs(custom_objects[key])
                print(f"[PATCH] {key} (mismo nombre) -> acepta kwargs Keras")
            else:
                # intenta resolver por contención de nombre
                candidates = [k for k, v in custom_objects.items() if inspect.isclass(v) and key.lower() in k.lower()]
                if len(candidates) == 1:
                    real = candidates[0]
                    custom_objects[real] = _patch_class_kwargs(custom_objects[real])
                    custom_objects[key] = custom_objects[real]  # expone alias también
                    print(f"[PATCH] {key} -> {real} (parcheado) acepta kwargs Keras")
                elif len(candidates) > 1:
                    print(f"[WARN] {key}: múltiples candidatos {candidates[:5]} (no se parchea)")
                else:
                    print(f"[WARN] {key}: no encontrado en custom_objects (no se parchea)")

    # 4) Cargar modelo (reintento con alias automático si hace falta)
    try:
        return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        msg = str(e)
        print("[WARN] Primer intento de load_model falló:", msg)
        missing = None
        for key in ["Unknown layer:", "Unknown metric:", "Unknown loss:"]:
            if key in msg:
                missing = msg.split(key, 1)[1].split()[0].strip(", '\"")
                break
        if missing and missing not in custom_objects:
            candidates = [k for k in custom_objects if missing.lower() in k.lower()]
            if len(candidates) == 1:
                print(f"[INFO] mapeando alias '{missing}' -> '{candidates[0]}' y reintentando…")
                custom_objects[missing] = custom_objects[candidates[0]]
                return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        raise


# =========================
# Helpers de muestreo
# =========================
def sample_n(pts: np.ndarray, n: int) -> np.ndarray:
    """Devuelve exactamente n puntos (con reposición si hace falta)."""
    if pts.shape[0] == n:
        return pts
    replace = pts.shape[0] < n
    idx = np.random.choice(pts.shape[0], size=n, replace=replace)
    return pts[idx]


# =========================
# Inferencia por paciente
# =========================
def infer_one(pid: str, struct_root: Path, out_pred_root: Path, model,
              tooth_class: int, n_points=8192, use_lower_context=True, export_ply=False):
    # cargar nubes originales del preproc (cada una ya está a 8192 por defecto)
    up_full_src = load_cloud(struct_root / "upper" / f"{pid}_full"   / "point_cloud.npy", n_points)
    lo_full_src = load_cloud(struct_root / "lower" / f"{pid}_full"   / "point_cloud.npy", n_points)
    gt_up21     = load_cloud(struct_root / "upper" / f"{pid}_rec_21" / "point_cloud.npy", n_points)

    # construir EXACTAMENTE n_points para el input del modelo
    if use_lower_context:
        # mitad upper + mitad lower (redondeo si n_points es impar)
        nU = n_points // 2
        nL = n_points - nU
        up_in = sample_n(up_full_src, nU)
        lo_in = sample_n(lo_full_src, nL)
        x = np.concatenate([up_in, lo_in], axis=0)   # (n_points, 3)
        N = nU  # solo los primeros N corresponden a upper para enmascarar tooth_class
        up_ref_for_mask = up_in
    else:
        # solo upper remuestreado a n_points
        up_in = sample_n(up_full_src, n_points)
        x = up_in
        N = n_points
        up_ref_for_mask = up_in

    # normalizar como en preprocessing
    x_norm = normalize_unit_sphere(x)

    # predicción: (1,P,C) con softmax
    probs = model.predict(x_norm[None, ...], verbose=0)   # (1, P, C)
    preds = np.argmax(probs, axis=-1)[0].astype(np.int32)

    # máscara de clase a eliminar SOLO en la porción upper (primeros N)
    keep_mask_upper = preds[:N] != int(tooth_class)
    up_pred = up_ref_for_mask[keep_mask_upper]

    # robustez si quedaron pocos puntos
    if 0 < up_pred.shape[0] < 256:
        up_pred = sample_n(up_pred, min(1024, up_pred.shape[0]))

    # normalización para métricas (pred vs CAD upper_rec_21)
    up_pred_n = normalize_unit_sphere(up_pred) if up_pred.size else up_pred
    gt_up21_n = normalize_unit_sphere(gt_up21)

    if up_pred_n.size == 0:
        cd = float("inf"); hd = float("inf")
    else:
        cd = chamfer_distance_np(up_pred_n, gt_up21_n)
        hd = hausdorff_distance_np(up_pred_n, gt_up21_n)

    # guardar pred
    pdir = out_pred_root / pid
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / "upper_rec_21_pred.npy", up_pred.astype(np.float32))
    if export_ply:
        save_cloud_ply(up_pred.astype(np.float32), pdir / "upper_rec_21_pred.ply")

    return {"patient_id": pid, "chamfer": cd, "hausdorff": hd, "n_pred": int(up_pred.shape[0])}


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--struct_rel", default="processed_struct")
    ap.add_argument("--model_path", required=True, help="SavedModel dir o archivo .keras/.h5")

    # custom objects
    ap.add_argument("--co_modules", default="",
                    help="módulos custom_objects separados por coma, ej: 'scripts.infer_utils,scripts.pointnetpp_layers'")
    ap.add_argument("--co_alias", default="",
                    help='JSON alias->real, ej: {"SetAbstraction":"PointNetSetAbstraction"}')
    ap.add_argument("--co_patch", default="",
                    help="clases a PARCHEAR para aceptar kwargs Keras, sep. por coma (ej: 'SetAbstraction,FeaturePropagation')")

    # setup inferencia
    ap.add_argument("--tooth_class", type=int, default=21)
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--use_lower_context", action="store_true",
                    help="Concatena lower_full como contexto; solo enmascara en los primeros N puntos (upper)")

    # salidas
    ap.add_argument("--out_pred", default="preds_tf_seg")
    ap.add_argument("--metrics_csv", default="ufrn_metrics_tf_seg.csv")
    ap.add_argument("--export_zip", default=None)
    ap.add_argument("--export_ply", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    struct_root   = Path(args.ufrn_root) / args.struct_rel
    out_pred_root = Path(args.ufrn_root) / args.out_pred
    metrics_csv   = Path(args.ufrn_root) / args.metrics_csv

    if args.clean and out_pred_root.exists():
        shutil.rmtree(out_pred_root)
    out_pred_root.mkdir(parents=True, exist_ok=True)

    co_modules = [m.strip() for m in args.co_modules.split(",") if m.strip()]
    co_alias   = args.co_alias.strip() if args.co_alias.strip() else None
    co_patch   = [x.strip() for x in args.co_patch.split(",") if x.strip()] if args.co_patch else None

    model = load_model_with_custom_objects(args.model_path, co_modules, co_alias, co_patch)
    print(f"[INFO] Modelo cargado: {args.model_path}")

    patients = list_patients(struct_root)
    if not patients:
        print("[ERR] No hay pacientes válidos en processed_struct/"); sys.exit(1)
    print(f"[INFO] Pacientes: {len(patients)}")

    rows = []
    for pid in patients:
        row = infer_one(pid, struct_root, out_pred_root, model,
                        tooth_class=args.tooth_class,
                        n_points=args.n_points,
                        use_lower_context=args.use_lower_context,
                        export_ply=args.export_ply)
        rows.append(row)
        print(f"[OK] {pid}: CD={row['chamfer']:.6f}  HD={row['hausdorff']:.6f}  n_pred={row['n_pred']}")

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("patient_id")
    df.to_csv(metrics_csv, index=False)
    print(f"[CSV] {metrics_csv}")
    print(f"[SUMMARY] Chamfer(mean)={df['chamfer'].mean():.6f}  Hausdorff(mean)={df['hausdorff'].mean():.6f}")

    if args.export_zip:
        export_preds_zip(out_pred_root, Path(args.export_zip))


if __name__ == "__main__":
    main()
