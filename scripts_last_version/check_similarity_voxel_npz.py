#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_similarity_voxel_npz.py

Detecta nubes de puntos "demasiado similares" usando descriptores globales (voxel histogram)
y similitud coseno, incluso si NO son duplicados exactos.

Entrada esperada:
  data_dir/X_train.npz, X_val.npz, X_test.npz   (clave "X" con shape [B,N,3])

Salida:
  - imprime resumen por split y cruces
  - guarda CSV con pares sospechosos (sim >= threshold)
  - guarda top-K pares globales (más similares)

Uso:
python3 check_similarity_voxel_npz.py \
  --data_dir /home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2 \
  --grid 16 --sample_points 2048 --take_train 2280 --take_val 95 --take_test 95 \
  --threshold 0.995 --topk 200 --out_csv suspicious_pairs.csv

Notas:
- grid=16 => descriptor de 4096 dims (16^3). grid=20 => 8000 dims (más fino).
- sample_points controla cuánto submuestrea para construir el descriptor (más rápido).
- threshold muy alto (0.995/0.998) detecta clones casi idénticos.
"""

import os
import argparse
from pathlib import Path
import numpy as np


# ---------------------------
# Utils
# ---------------------------
def normalize_unit_sphere_np(xyz: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # xyz: [N,3]
    c = xyz.mean(axis=0, keepdims=True)
    x = xyz - c
    r = np.linalg.norm(x, axis=1).max()
    r = max(float(r), eps)
    return x / r


def subsample_points(xyz: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    n = xyz.shape[0]
    if m <= 0 or m >= n:
        return xyz
    idx = rng.choice(n, size=m, replace=False)
    return xyz[idx]


def voxel_descriptor(xyz: np.ndarray, grid: int) -> np.ndarray:
    """
    Histograma de ocupación en una grilla grid^3 sobre [-1,1]^3.
    Devuelve vector float32 normalizado L2.
    """
    g = int(grid)
    # clamp a [-1,1] por seguridad
    x = np.clip(xyz, -1.0, 1.0)

    # mapear [-1,1] -> [0,g-1]
    u = (x + 1.0) * 0.5 * (g - 1)
    ijk = np.floor(u + 1e-6).astype(np.int32)  # [N,3]
    ijk = np.clip(ijk, 0, g - 1)

    # linear index
    lin = ijk[:, 0] * (g * g) + ijk[:, 1] * g + ijk[:, 2]  # [N]
    hist = np.bincount(lin, minlength=g * g * g).astype(np.float32)

    # normalizar por conteo total (invariante a N) y L2 (para coseno)
    s = hist.sum()
    if s > 0:
        hist /= s
    nrm = np.linalg.norm(hist)
    if nrm > 0:
        hist /= nrm
    return hist


def load_X(data_dir: Path, split: str) -> np.ndarray:
    p = data_dir / f"X_{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    X = np.load(p, allow_pickle=False)["X"]
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X_{split} shape inesperada: {X.shape} (esperado [B,N,3])")
    return X.astype(np.float32, copy=False)


def build_descs(X: np.ndarray, grid: int, sample_points: int, take: int, seed: int) -> np.ndarray:
    B = X.shape[0]
    take = min(int(take), B) if take and take > 0 else B
    rng = np.random.default_rng(seed)
    D = grid * grid * grid
    out = np.zeros((take, D), dtype=np.float32)
    for i in range(take):
        xyz = X[i]
        xyz = normalize_unit_sphere_np(xyz)
        xyz = subsample_points(xyz, sample_points, rng)
        out[i] = voxel_descriptor(xyz, grid)
    return out


def top_pairs_cosine(A: np.ndarray, B: np.ndarray, topk: int, block: int = 512):
    """
    Devuelve lista de (i,j,sim) con topk máximos sobre A x B.
    Bloquea para no explotar RAM.
    """
    topk = int(topk)
    best = []  # heap manual (lista) porque topk pequeño
    # normalizado L2 => dot = cos
    for i0 in range(0, A.shape[0], block):
        Ai = A[i0:i0+block]  # [bi,D]
        S = Ai @ B.T         # [bi, nb]
        # extraer top local
        bi = S.shape[0]
        for ii in range(bi):
            row = S[ii]
            # topk de esta fila
            if topk <= 0:
                continue
            # argpartition rápido
            kk = min(topk, row.shape[0])
            idx = np.argpartition(row, -kk)[-kk:]
            for j in idx:
                sim = float(row[j])
                best.append((i0 + ii, int(j), sim))
    # ordenar global y recortar
    best.sort(key=lambda x: x[2], reverse=True)
    return best[:topk]


def count_above_threshold(A: np.ndarray, B: np.ndarray, thr: float, block: int = 512) -> int:
    thr = float(thr)
    cnt = 0
    for i0 in range(0, A.shape[0], block):
        Ai = A[i0:i0+block]
        S = Ai @ B.T
        cnt += int((S >= thr).sum())
    return cnt


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("splitA,idxA,splitB,idxB,cos_sim\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)

    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--sample_points", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--take_train", type=int, default=0)
    ap.add_argument("--take_val", type=int, default=0)
    ap.add_argument("--take_test", type=int, default=0)

    ap.add_argument("--threshold", type=float, default=0.995)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--block", type=int, default=256)

    ap.add_argument("--out_csv", type=str, default="suspicious_pairs.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    grid = int(args.grid)
    D = grid**3

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] grid={grid} -> D={D} dims | sample_points={args.sample_points} | threshold={args.threshold}")

    Xtr = load_X(data_dir, "train")
    Xva = load_X(data_dir, "val")
    Xte = load_X(data_dir, "test")

    dtr = build_descs(Xtr, grid, args.sample_points, args.take_train, args.seed + 11)
    dva = build_descs(Xva, grid, args.sample_points, args.take_val, args.seed + 22)
    dte = build_descs(Xte, grid, args.sample_points, args.take_test, args.seed + 33)

    print(f"[DESC] train: {dtr.shape} | val: {dva.shape} | test: {dte.shape}")

    thr = float(args.threshold)
    blk = int(args.block)

    # Internos (ojo: en internos hay diagonal sim=1.0 consigo mismo; lo descontamos)
    def internal_report(name, A):
        if A.shape[0] < 2:
            print(f"[{name}] muy chico")
            return
        # conteo >= thr (incluye diagonal)
        cnt = count_above_threshold(A, A, thr, block=blk)
        diag = A.shape[0]  # sim=1
        print(f"[{name}] pares >=thr (incl diagonal) = {cnt} | sin diagonal = {cnt - diag}")

    internal_report("train", dtr)
    internal_report("val", dva)
    internal_report("test", dte)

    # Cruces
    def cross_report(a_name, A, b_name, B):
        cnt = count_above_threshold(A, B, thr, block=blk)
        print(f"[CROSS] {a_name} ∩ {b_name} : pares >=thr = {cnt}")

    cross_report("train", dtr, "val", dva)
    cross_report("train", dtr, "test", dte)
    cross_report("val", dva, "test", dte)

    # Top pares globales por cada cruce (útil para inspección)
    rows = []

    def add_top(a_name, A, b_name, B, topk):
        pairs = top_pairs_cosine(A, B, topk=topk, block=blk)
        for (i, j, s) in pairs:
            if s >= thr:
                rows.append((a_name, i, b_name, j, f"{s:.6f}"))

    add_top("train", dtr, "val", dva, args.topk)
    add_top("train", dtr, "test", dte, args.topk)
    add_top("val", dva, "test", dte, args.topk)

    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)
    print(f"[OK] CSV sospechosos (sim>=thr) guardado en: {out_csv} | n_rows={len(rows)}")

    # Mensaje final interpretativo
    if len(rows) == 0:
        print("[CONCLUSION] No aparecen pares extremadamente similares bajo este descriptor/umbral. Todo se ve sano.")
    else:
        print("[CONCLUSION] Hay pares MUY similares (ver CSV). Revisa esos índices en tus plots/infer y en index_*.csv.")

if __name__ == "__main__":
    main()
