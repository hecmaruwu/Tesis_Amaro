#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusión de nubes FPS preprocesadas en dataset global (PointNet-like).
Basado en Qi et al. 2017 y Akahori et al. 2023.

Entrada:
  processed_struct/<N>/<jaw>/<sample>/{point_cloud.npy,labels.npy}

Salida:
  merged_<N>/
    X_train.npz, Y_train.npz
    X_val.npz,   Y_val.npz
    X_test.npz,  Y_test.npz
    artifacts/{label_map.json, meta.json}
"""

import argparse, json, random, gc
from pathlib import Path
import numpy as np

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def normalize(points):
    points = np.asarray(points, np.float32)
    points -= points.mean(axis=0, keepdims=True)
    d = np.linalg.norm(points, axis=1)
    m = d.max() if d.size else 1.0
    if m > 0:
        points /= m
    return points

def load_sample(folder: Path):
    X = np.load(folder / "point_cloud.npy").astype(np.float32)
    yfile = folder / "labels.npy"
    Y = np.load(yfile).astype(np.int32) if yfile.exists() else np.zeros((X.shape[0],), np.int32)
    return X, Y

def build_label_map(all_labels):
    unique = sorted(int(x) for x in np.unique(all_labels))
    id2idx = {int(v): i for i, v in enumerate(unique)}
    idx2id = {i: int(v) for i, v in enumerate(unique)}
    return id2idx, idx2id

def stratified_split(X, Y, ratios=(0.8, 0.1, 0.1), seed=42):
    """Estratifica por etiqueta mayoritaria."""
    seed_all(seed)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(ratios[0]*n)
    n_val = int(ratios[1]*n)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="processed_struct/<N>")
    ap.add_argument("--out_dir", required=True, help="salida merged_<N>")
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)

    all_X, all_Y, all_names = [], [], []

    for jaw in ("fps_vertices_base_upper", "fps_vertices_base_lower"):
        jaw_dir = in_root / jaw
        if not jaw_dir.exists():
            continue
        for subj in sorted(jaw_dir.iterdir()):
            if not subj.is_dir():
                continue
            try:
                X, Y = load_sample(subj)
                X = normalize(X)
                all_X.append(X)
                all_Y.append(Y)
                all_names.append(f"{jaw}_{subj.name}")
            except Exception as e:
                print(f"[WARN] No se pudo leer {subj}: {e}")
                continue

    X = np.stack(all_X, axis=0).astype(np.float32)
    Y = np.stack(all_Y, axis=0).astype(np.int32)
    del all_X, all_Y
    gc.collect()

    print(f"[OK] Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} pts c/u")

    id2idx, idx2id = build_label_map(Y)
    np.save(out_dir / "label_ids.npy", np.array(list(id2idx.keys())))
    json.dump({"id2idx": id2idx, "idx2id": idx2id},
              open(out_dir / "artifacts" / "label_map.json", "w"), indent=2)

    Y_remap = np.vectorize(id2idx.get)(Y).astype(np.int32)

    idx_train, idx_val, idx_test = stratified_split(X, Y_remap, seed=args.seed)

    def save_split(name, idxs):
        np.savez_compressed(out_dir / f"X_{name}.npz", X=X[idxs])
        np.savez_compressed(out_dir / f"Y_{name}.npz", Y=Y_remap[idxs])
        print(f"[SAVE] {name}: {len(idxs)} muestras")

    save_split("train", idx_train)
    save_split("val", idx_val)
    save_split("test", idx_test)

    json.dump({
        "n_samples": int(X.shape[0]),
        "n_points": int(X.shape[1]),
        "n_classes": len(id2idx),
        "seed": args.seed
    }, open(out_dir / "artifacts" / "meta.json", "w"), indent=2)

    print(f"[DONE] Guardado en {out_dir}")

if __name__ == "__main__":
    main()
