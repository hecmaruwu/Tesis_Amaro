#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crea un split nuevo con:
 - etiquetas remapeadas a índices consecutivos 0..C-1 (artifacts/label_map.json),
 - nubes submuestreadas a N puntos fijos (o N = min puntos si --N auto),
 - (opcional) cap de fondo (label 0 ORIGINAL) a una fracción máxima por nube.

Entrada soportada:
 A) Archivos apilados: X_{train,val,test}.npy/npz (M,N,3) y Y_{...} (M,N)
 B) Archivos por muestra: X_*.npy/npz (N,3) y y_*.npy/npz (N,)

Salida SIEMPRE en .npz con claves "X" / "Y":
  out_split/X_{split}.npz, Y_{split}.npz  + artifacts/label_map.json
"""
import os, json, argparse, shutil, random
from pathlib import Path
import numpy as np

# -------- utils --------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)

def _load_array(path: Path):
    """Carga .npy o .npz (arr_0, X, Y, o primera clave)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            for k in ("X","Y","arr_0"):
                if k in z: return z[k]
            return z[list(z.keys())[0]]
    raise ValueError(f"Extensión no soportada: {path}")

def list_label_files(split_dir: Path):
    """
    Devuelve pares (X_file, Y_file) y el modo.
    Caso A: X_train.* / Y_train.* ... (hasta 3 pares)
    Caso B: por muestra → empareja X_*.npy/npz con y_*.npy/npz
    """
    # Caso A
    stacked = []
    for s in ("train","val","test"):
        xcand = next((split_dir/f"X_{s}{ext}" for ext in (".npz",".npy") if (split_dir/f"X_{s}{ext}").exists()), None)
        ycand = next((split_dir/f"Y_{s}{ext}" for ext in (".npz",".npy") if (split_dir/f"Y_{s}{ext}").exists()), None)
        if xcand is not None and ycand is not None:
            stacked.append((xcand, ycand, s))
    if stacked:
        return [(xf, yf) for xf, yf, _ in stacked], "A", stacked

    # Caso B
    sample_pairs = []
    for yfile in list(split_dir.rglob("y_*.npy")) + list(split_dir.rglob("y_*.npz")):
        xfile_npy = yfile.with_name(yfile.name.replace("y_", "X_")).with_suffix(".npy")
        xfile_npz = yfile.with_name(yfile.name.replace("y_", "X_")).with_suffix(".npz")
        xfile = xfile_npy if xfile_npy.exists() else (xfile_npz if xfile_npz.exists() else None)
        if xfile is not None:
            sample_pairs.append((xfile, yfile))
    if sample_pairs:
        return sample_pairs, "B", None

    return [], "NONE", None

def compute_min_points_for_A(stacked_info):
    mins = []
    for xf, _, _ in stacked_info:
        X = _load_array(xf)
        if X.ndim != 3 or X.shape[2] != 3:
            raise ValueError(f"{xf} no es (M,N,3); shape={X.shape}")
        mins.append(X.shape[1])
    return int(min(mins)) if mins else 0

def compute_min_points_for_B(sample_pairs):
    mins = []
    for xf, _ in sample_pairs:
        X = _load_array(xf)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"{xf} no es (N,3); shape={X.shape}")
        mins.append(X.shape[0])
    return int(min(mins)) if mins else 0

def build_label_map_for_A(stacked_info):
    vals = set()
    for _, yf, _ in stacked_info:
        y = _load_array(yf).astype(np.int64)
        if y.ndim != 2:
            raise ValueError(f"{yf} esperada shape (M,N); got {y.shape}")
        vals.update(np.unique(y).tolist())
    vals = sorted(int(v) for v in vals)
    id2idx = {int(v): i for i, v in enumerate(vals)}
    idx2id = {i: int(v) for i, v in enumerate(vals)}
    return id2idx, idx2id

def build_label_map_for_B(sample_pairs):
    vals = set()
    for _, yf in sample_pairs:
        y = _load_array(yf).astype(np.int64).ravel()
        vals.update(np.unique(y).tolist())
    vals = sorted(int(v) for v in vals)
    id2idx = {int(v): i for i, v in enumerate(vals)}
    idx2id = {i: int(v) for i, v in enumerate(vals)}
    return id2idx, idx2id

def subsample_with_cap(points, labels, N, bg_ids, cap_bg_frac=None, rng=None):
    """Submuestrea exactamente N puntos por muestra. Si cap_bg_frac, limita
    la fracción de etiquetas en bg_ids (ya remapeadas)."""
    rng = rng or np.random.default_rng()
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    n = points.shape[0]
    idx_all = np.arange(n, dtype=np.int64)

    if cap_bg_frac is not None and len(bg_ids) > 0:
        mask_bg = np.isin(labels, np.asarray(list(bg_ids), dtype=np.int64))
        idx_bg  = idx_all[mask_bg]
        idx_fg  = idx_all[~mask_bg]
        max_bg  = int(round(float(cap_bg_frac) * int(N)))
        take_bg = min(max_bg, idx_bg.size)
        rem     = max(0, int(N) - take_bg)
        sel_bg  = rng.choice(idx_bg, size=take_bg, replace=False) if take_bg>0 else np.empty((0,), np.int64)
        pool    = idx_fg if idx_fg.size>0 else idx_bg
        sel_fg  = rng.choice(pool, size=rem, replace=(pool.size < rem))
        sel     = np.concatenate([sel_bg, sel_fg], axis=0)
    else:
        sel = rng.choice(idx_all, size=int(N), replace=(n < int(N)))

    return points[sel], labels[sel]

def save_npz(path: Path, key: str, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: arr})

# -------- pipeline --------
def process_split(in_split: Path, out_split: Path, N_target, cap_bg_frac, seed, bg_original_ids=(0,)):
    seed_all(seed)
    pairs, mode, stacked_info = list_label_files(in_split)
    if mode == "NONE":
        raise SystemExit(f"No se encontraron datos en {in_split}")

    # N objetivo
    if N_target == "auto":
        N = compute_min_points_for_A(stacked_info) if mode == "A" else compute_min_points_for_B(pairs)
        if N <= 0: raise SystemExit("No pude computar el mínimo de puntos.")
        print(f"[N] auto  -> min puntos = {N}")
    else:
        N = int(N_target); print(f"[N] fijo  -> {N}")

    # Label map global
    id2idx, idx2id = (build_label_map_for_A(stacked_info) if mode == "A"
                      else build_label_map_for_B(pairs))
    print(f"[LABELS] únicos={len(id2idx)}  ejemplo: {list(id2idx.items())[:8]}")

    # salida
    out_split.mkdir(parents=True, exist_ok=True)
    (out_split / "artifacts").mkdir(exist_ok=True)
    (out_split/"artifacts/label_map.json").write_text(
        json.dumps({"id2idx": id2idx, "idx2id": idx2id}, indent=2), encoding="utf-8"
    )

    rng = np.random.default_rng(seed)
    total = 0

    if mode == "A":
        for xf, yf, splitname in stacked_info:
            X = _load_array(xf).astype(np.float32)   # (M, N_x, 3)
            Y = _load_array(yf).astype(np.int64)     # (M, N_x)
            if X.ndim != 3 or X.shape[2] != 3 or Y.ndim != 2 or X.shape[:2] != Y.shape:
                raise ValueError(f"Shapes inválidas en {xf}/{yf}: X={X.shape}  Y={Y.shape}")

            M = X.shape[0]
            Xo = np.empty((M, N, 3), dtype=np.float32)
            Yo = np.empty((M, N),    dtype=np.int32)

            bg_ids_remap = [id2idx[i] for i in bg_original_ids if i in id2idx]

            for i in range(M):
                y_m = np.vectorize(id2idx.__getitem__)(Y[i].ravel()).astype(np.int32)
                Xi, Yi = subsample_with_cap(X[i], y_m, N, bg_ids_remap, cap_bg_frac, rng)
                Xo[i] = Xi; Yo[i] = Yi; total += 1
                if total % 50 == 0: print(f"[{total}] {splitname}: {i+1}/{M}")

            save_npz(out_split/f"X_{splitname}.npz", "X", Xo)
            save_npz(out_split/f"Y_{splitname}.npz", "Y", Yo)

        for extra in ("meta.json", "readme.txt"):
            pe = in_split/extra
            if pe.exists(): shutil.copy2(pe, out_split/extra)

    else:
        bg_ids_remap = [id2idx[i] for i in bg_original_ids if i in id2idx]
        X_list, Y_list = [], []
        for xf, yf in pairs:
            X = _load_array(xf).astype(np.float32)        # (N,3)
            Y = _load_array(yf).astype(np.int64).ravel()  # (N,)
            if X.ndim != 2 or X.shape[1] != 3:
                raise ValueError(f"{xf} no es (N,3); shape={X.shape}")
            y_m = np.vectorize(id2idx.__getitem__)(Y).astype(np.int32)
            Xs, ys = subsample_with_cap(X, y_m, N, bg_ids_remap, cap_bg_frac, rng)
            X_list.append(Xs); Y_list.append(ys); total += 1
            if total % 300 == 0: print(f"[{total}] procesadas")

        # Guarda en apilado por conveniencia
        Xo = np.stack(X_list, axis=0).astype(np.float32)  # (M,N,3)
        Yo = np.stack(Y_list, axis=0).astype(np.int32)    # (M,N)
        save_npz(out_split/"X_train.npz", "X", Xo)  # si no tienes splits, puedes duplicar a val/test según necesites
        save_npz(out_split/"Y_train.npz", "Y", Yo)

    print(f"[OK] procesadas {total} {'muestras' if mode=='A' else 'nubes'}.")
    print(f"[OUT] {out_split}")
    print(f"[MAP] {out_split/'artifacts/label_map.json'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_split", required=True)
    ap.add_argument("--out_split", required=True)
    ap.add_argument("--N", default="auto")
    ap.add_argument("--cap_bg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    process_split(Path(a.in_split), Path(a.out_split), a.N, a.cap_bg, a.seed)

if __name__ == "__main__":
    main()
