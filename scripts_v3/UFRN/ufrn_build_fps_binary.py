#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construye dataset binario (upper_full vs upper_rec_21) desde targets_export.
Aplica FPS (Farthest Point Sampling) con distintas resoluciones.
Salidas:
  /data/UFRN/processed_bin/<N>/X.npy, Y.npy
"""

import os, re, sys, shutil
from pathlib import Path
import numpy as np

try:
    import trimesh as tm
except Exception:
    tm = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x


# ---------- Normalización y FPS ----------
def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0, keepdims=True)
    pts = pts - c
    r = np.linalg.norm(pts, axis=1).max()
    if r > 0:
        pts = pts / r
    return pts


def fps(points: np.ndarray, n_samples: int, start_idx: int | None = None) -> np.ndarray:
    """Farthest Point Sampling implementado en NumPy puro."""
    P = np.asarray(points, dtype=np.float32)
    N = P.shape[0]
    n_samples = min(n_samples, N)
    chosen = np.empty(n_samples, dtype=np.int64)
    dists = np.full((N,), np.inf, dtype=np.float32)
    if start_idx is None:
        start_idx = np.random.randint(0, N)
    farthest = start_idx
    for i in range(n_samples):
        chosen[i] = farthest
        diff = P - P[farthest]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dists = np.minimum(dists, dist2)
        farthest = np.argmax(dists)
    return P[chosen]

def load_mesh_safe(path: Path):
    try:
        m = tm.load(path, process=False, force="mesh")
        if isinstance(m, tm.Scene):
            m = m.dump().sum()
        return m
    except Exception as e:
        print(f"[WARN] No se pudo cargar {path}: {e}", file=sys.stderr)
        return None


def sample_points_from_mesh(mesh: tm.Trimesh, n_points: int) -> np.ndarray:
    """Muestreo uniforme sobre superficie, con normalización a esfera unitaria."""
    if mesh is None:
        raise ValueError("Malla inválida.")
    if mesh.faces is not None and len(mesh.faces) > 0:
        pts, _ = tm.sample.sample_surface(mesh, n_points)
    else:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        idx = np.random.choice(len(v), size=n_points, replace=(len(v) < n_points))
        pts = v[idx]
    return normalize_points(pts)


PAT_RE = re.compile(r"(?i)^paciente[_\-\s]*([0-9]{1,3})$")

def get_patients(base: Path):
    pats = [p for p in base.iterdir() if p.is_dir() and PAT_RE.match(p.name)]
    pats.sort(key=lambda p: int(PAT_RE.match(p.name).group(1)))
    return pats

def process_patient(p_dir: Path, n_points: int):
    """
    Procesa un paciente con upper_full.stl y upper_rec_21.stl.
    Devuelve ambas nubes sampleadas y normalizadas.
    """
    stl_dir = p_dir / "stl"
    full_p = stl_dir / "upper_full.stl"
    rec_p = stl_dir / "upper_rec_21.stl"

    if not full_p.exists() or not rec_p.exists():
        return None

    mesh_full = load_mesh_safe(full_p)
    mesh_rec = load_mesh_safe(rec_p)
    if mesh_full is None or mesh_rec is None:
        return None

    pts_full = sample_points_from_mesh(mesh_full, n_points)
    pts_rec = sample_points_from_mesh(mesh_rec, n_points)
    return {"pid": p_dir.name, "pts_full": pts_full, "pts_rec": pts_rec}

def build_binary_dataset(root: Path, n_points_list: list[int]):
    base = root / "targets_export"
    if not base.is_dir():
        raise FileNotFoundError(f"No existe carpeta: {base}")

    patients = get_patients(base)
    print(f"[INFO] Pacientes detectados: {len(patients)}")

    for n_points in n_points_list:
        print(f"\n[BUILD] Resolución {n_points} puntos")
        out_dir = root / "processed_bin" / str(n_points)
        out_dir.mkdir(parents=True, exist_ok=True)
        X_all, Y_all = [], []

        for p in tqdm(patients, ncols=100):
            res = process_patient(p, n_points)
            if res is None:
                continue
            X_all.append(res["pts_full"]); Y_all.append(0)
            X_all.append(res["pts_rec"]);  Y_all.append(1)

        if not X_all:
            print(f"[WARN] No hay datos válidos para {n_points}.")
            continue

        X_all = np.stack(X_all, axis=0)
        Y_all = np.array(Y_all, dtype=np.int64)
        np.save(out_dir / "X.npy", X_all)
        np.save(out_dir / "Y.npy", Y_all)
        print(f"[OK] Guardado dataset binario: {out_dir}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", default="/home/htaucare/Tesis_Amaro/data/UFRN",
                    help="Ruta raíz del dataset UFRN (con targets_export)")
    ap.add_argument("--n_points", nargs="+", type=int,
                    default=[1024, 2048, 4096, 8192],
                    help="Resoluciones FPS a generar")
    args = ap.parse_args()

    build_binary_dataset(Path(args.ufrn_root), args.n_points)
    print("\n✅ Dataset binario FPS generado correctamente.")


if __name__ == "__main__":
    main()
