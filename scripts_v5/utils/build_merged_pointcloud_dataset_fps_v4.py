#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_merged_pointcloud_dataset_fps_v4.py
--------------------------------------------------------------
Fusión de nubes FPS preprocesadas en un dataset global para
entrenamiento de modelos tipo PointNet / PointNet++ / Transformer3D.

Esta versión (v4) es completamente compatible con las salidas del
preprocesamiento extendido que incluye features geométricas:
  - XYZ + normales + curvatura (N,7)
  - o solo XYZ (N,3) si no se calcularon.

Genera splits estratificados train/val/test y guarda:
  X_train.npz, X_val.npz, X_test.npz
  Y_train.npz, Y_val.npz, Y_test.npz
  artifacts/{label_map.json, meta.json}

--------------------------------------------------------------
Autor: Adaptado por ChatGPT (GPT-5)
Para la tesis de H. Taucare (Universidad de Chile)
--------------------------------------------------------------
"""

import argparse
import json
import random
import gc
from pathlib import Path
import numpy as np


# ============================================================
# === Utilidades generales ===================================
# ============================================================

def seed_all(seed: int = 42):
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)


def normalize_xyz_inplace(X: np.ndarray) -> np.ndarray:
    """
    Normaliza solo las 3 primeras columnas (XYZ) de cada nube a esfera unitaria.
    Mantiene el resto de características geométricas (normales, curvatura) intactas.
    """
    X = np.asarray(X, np.float32)
    if X.shape[1] < 3:
        raise ValueError(f"Se esperaban al menos 3 columnas (XYZ), recibido: {X.shape}")
    coords = X[:, :3]
    coords -= coords.mean(axis=0, keepdims=True)
    r = np.linalg.norm(coords, axis=1).max()
    if r > 0:
        coords /= r
    X[:, :3] = coords
    return X


def load_sample(folder: Path):
    """
    Carga una muestra individual de nubes procesadas.
    Prefiere `point_cloud_feats.npy` si existe, o `point_cloud.npy` en su defecto.
    Devuelve:
        X: np.ndarray (N, 3+)
        Y: np.ndarray (N,)
    """
    cand_feats = folder / "point_cloud_feats.npy"
    cand_xyz = folder / "point_cloud.npy"

    if cand_feats.exists():
        X = np.load(cand_feats).astype(np.float32)
    elif cand_xyz.exists():
        X = np.load(cand_xyz).astype(np.float32)
    else:
        raise FileNotFoundError(f"No se encontró ningún point_cloud en {folder}")

    yfile = folder / "labels.npy"
    if yfile.exists():
        Y = np.load(yfile).astype(np.int32)
    else:
        # Si no hay etiquetas, se asignan ceros (permitiendo clustering no supervisado)
        Y = np.zeros((X.shape[0],), np.int32)

    # Normaliza XYZ por muestra
    X = normalize_xyz_inplace(X)
    return X, Y


def build_label_map(all_labels: np.ndarray):
    """
    Construye un mapeo consistente entre IDs originales y
    un rango continuo [0..C-1] para entrenamientos con CE Loss.
    """
    unique = sorted(int(x) for x in np.unique(all_labels))
    id2idx = {int(v): i for i, v in enumerate(unique)}
    idx2id = {i: int(v) for i, v in enumerate(unique)}
    return id2idx, idx2id


def stratified_split(X, Y, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Divide el dataset en splits train/val/test de forma reproducible.
    Estratificación pseudoaleatoria (sin balanceo por clase estricta).
    """
    seed_all(seed)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ============================================================
# === MAIN ===================================================
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fusiona nubes FPS preprocesadas en dataset global (v4)."
    )
    parser.add_argument("--in_root", required=True,
                        help="Ruta raíz de entrada (ej: data/Teeth_3ds/processed_flat/8192)")
    parser.add_argument("--out_dir", required=True,
                        help="Ruta de salida (ej: data/Teeth_3ds/merged_xyz_v4)")
    parser.add_argument("--n_points", type=int, default=8192,
                        help="Número de puntos por muestra (default=8192).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla RNG para reproducibilidad.")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="Proporciones de train, val, test (default 0.8 0.1 0.1).")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Inicialización
    # --------------------------------------------------------
    seed_all(args.seed)
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)

    all_X, all_Y, all_names = [], [], []

    # --------------------------------------------------------
    # Recorre ambas arcadas (upper y lower)
    # --------------------------------------------------------
    for jaw in ("fps_vertices_base_upper", "fps_vertices_base_lower"):
        jaw_dir = in_root / jaw
        if not jaw_dir.exists():
            print(f"[WARN] No existe: {jaw_dir}")
            continue
        for subj in sorted(jaw_dir.iterdir()):
            if not subj.is_dir():
                continue
            try:
                X, Y = load_sample(subj)
                all_X.append(X)
                all_Y.append(Y)
                all_names.append(f"{jaw}_{subj.name}")
            except Exception as e:
                print(f"[WARN] Error cargando {subj}: {e}")
                continue

    if len(all_X) == 0:
        raise SystemExit("[ERROR] No se cargó ninguna muestra válida.")

    # --------------------------------------------------------
    # Stack global y limpieza de memoria
    # --------------------------------------------------------
    X = np.stack(all_X, axis=0).astype(np.float32)
    Y = np.stack(all_Y, axis=0).astype(np.int32)
    del all_X, all_Y
    gc.collect()

    print(f"[OK] Dataset cargado: {X.shape[0]} muestras, "
          f"{X.shape[1]} puntos, {X.shape[2]} canales")

    # --------------------------------------------------------
    # Construcción de mapa de etiquetas (id2idx, idx2id)
    # --------------------------------------------------------
    id2idx, idx2id = build_label_map(Y)
    json.dump({"id2idx": id2idx, "idx2id": idx2id},
              open(out_dir / "artifacts" / "label_map.json", "w"), indent=2)

    # Remapeo de etiquetas a rango continuo
    Y_remap = np.vectorize(id2idx.get)(Y).astype(np.int32)

    # --------------------------------------------------------
    # División train / val / test
    # --------------------------------------------------------
    idx_train, idx_val, idx_test = stratified_split(X, Y_remap, ratios=args.ratios, seed=args.seed)

    def save_split(name, idxs):
        """Guarda un split (X e Y comprimidos en .npz)."""
        np.savez_compressed(out_dir / f"X_{name}.npz", X=X[idxs])
        np.savez_compressed(out_dir / f"Y_{name}.npz", Y=Y_remap[idxs])
        print(f"[SAVE] {name}: {len(idxs)} muestras")

    save_split("train", idx_train)
    save_split("val", idx_val)
    save_split("test", idx_test)

    # --------------------------------------------------------
    # Metadatos finales
    # --------------------------------------------------------
    meta = {
        "n_samples": int(X.shape[0]),
        "n_points": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
        "n_classes": len(id2idx),
        "seed": args.seed,
        "ratios": args.ratios,
        "has_features": bool(X.shape[2] > 3)
    }
    (out_dir / "artifacts" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[DONE] Dataset global guardado en:")
    print(f"       {out_dir}")
    print("Archivos generados:")
    print("  - X_train.npz, Y_train.npz")
    print("  - X_val.npz,   Y_val.npz")
    print("  - X_test.npz,  Y_test.npz")
    print("  - artifacts/label_map.json, artifacts/meta.json\n")


# ============================================================
# === Ejecución directa ======================================
# ============================================================

if __name__ == "__main__":
    main()
