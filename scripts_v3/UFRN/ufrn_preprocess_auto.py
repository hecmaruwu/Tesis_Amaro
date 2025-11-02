#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesamiento UFRN → nubes normalizadas (8192 pts) + CSVs y resumen.
Detecta:
  - paciente_51_sup.stl
  - paciente_51_upper_full.stl
  - paciente_51_sup_21.stl
  - paciente_51_upper_21.stl
"""

import argparse, os, re, json, shutil, sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np, pandas as pd, trimesh as tm
from tqdm import tqdm

# --- REGEX ampliados ---
PACIENTE_DIR_RE = re.compile(r"(?i)^paciente[\s_\-]*\d+\s*$")
PACIENTE_NUM_RE = re.compile(r"(?i)paciente[\s_\-]*([0-9]{1,3})")

# Arcadas completas
FULL_SUP_FILE_RE = re.compile(r"(?i)(sup|upper)[_\-\s]?(?:full)?\.stl$")
FULL_INF_FILE_RE = re.compile(r"(?i)(inf|lower)[_\-\s]?(?:full)?\.stl$")

# Recortado con diente faltante (ej: sup_21.stl, upper_21.stl)
RECORTADO_MISSING_RE = re.compile(r"(?i)(sup|upper|inf|lower)[_\-\s]?(?:recortado[_\-\s]?)?(1[1-8]|2[1-8]|3[1-8]|4[1-8])\.stl$")

def extract_patient_id_from_path(path: Path) -> str:
    for seg in path.parts:
        if PACIENTE_DIR_RE.match(seg):
            m = PACIENTE_NUM_RE.search(seg)
            if m:
                return f"paciente_{m.group(1)}"
    for seg in path.parts[:-1]:
        m = PACIENTE_NUM_RE.search(seg)
        if m:
            return f"paciente_{m.group(1)}"
    return "paciente_desconocido"

def detect_jaw_from_path_or_name(fpath: Path) -> str:
    low = fpath.name.lower()
    if "_sup" in low or "upper" in low:
        return "upper"
    if "_inf" in low or "lower" in low:
        return "lower"
    for seg in fpath.parts:
        if re.search(r"(?i)(sup|upper)", seg):
            return "upper"
        if re.search(r"(?i)(inf|lower)", seg):
            return "lower"
    return "unknown"

def is_full_arch_file(name: str) -> Tuple[bool, Optional[str]]:
    if "recortado" in name.lower():
        return False, None
    if FULL_SUP_FILE_RE.search(name):
        return True, "upper"
    if FULL_INF_FILE_RE.search(name):
        return True, "lower"
    return False, None

def parse_recortado(name: str) -> Tuple[bool, Optional[str], Optional[int]]:
    m = RECORTADO_MISSING_RE.search(name)
    if m:
        return True, m.group(1).lower(), int(m.group(2))
    return False, None, None

def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0, keepdims=True)
    pts = pts - c
    r = np.linalg.norm(pts, axis=1).max()
    return pts / r if r > 0 else pts

def sample_points_from_mesh(mesh: tm.Trimesh, n: int) -> np.ndarray:
    if mesh.is_empty:
        raise ValueError("Malla vacía.")
    if mesh.faces is not None and len(mesh.faces) > 0:
        pts, _ = tm.sample.sample_surface(mesh, n)
    else:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        idx = np.random.choice(len(v), size=n, replace=(len(v) < n))
        pts = v[idx]
    return normalize_points(pts)

def load_mesh_safe(path: Path):
    try:
        m = tm.load(path, process=False, force="mesh")
        if isinstance(m, tm.Scene):
            m = m.dump().sum()
        return m
    except Exception as e:
        print(f"[WARN] No pude cargar '{path}': {e}", file=sys.stderr)
        return None

def scan_files(auto_root: Path) -> pd.DataFrame:
    base = next((p for p in (auto_root / "extract").iterdir()
                 if "MODELOS" in p.name.upper()), None)
    if not base:
        raise FileNotFoundError(f"No se encontró carpeta MODELOS PESQUISA en {auto_root}")
    print(f"[AUTO] Carpeta MODELOS PESQUISA detectada: {base}")

    rows = []
    for root, _, files in os.walk(base):
        for f in files:
            if not f.lower().endswith(".stl"):
                continue
            fpath = Path(root) / f
            pid = extract_patient_id_from_path(fpath)
            jaw = detect_jaw_from_path_or_name(fpath)
            is_full, jaw2 = is_full_arch_file(f)
            if is_full:
                jaw = jaw if jaw != "unknown" else jaw2
                rows.append({
                    "file": str(fpath),
                    "patient_id": pid,
                    "jaw": jaw,
                    "is_full_arch": True,
                    "is_recortado": False,
                    "missing_tooth": None
                })
                continue
            is_rec, jaw3, miss = parse_recortado(f)
            if is_rec:
                jaw = jaw if jaw != "unknown" else jaw3
                rows.append({
                    "file": str(fpath),
                    "patient_id": pid,
                    "jaw": jaw,
                    "is_full_arch": False,
                    "is_recortado": True,
                    "missing_tooth": miss
                })
                continue
    df = pd.DataFrame(rows)
    print(f"[SCAN] Detectados {len(df)} archivos válidos")
    return df

def process_one(fpath: Path, n_points: int, dst_struct: Path, dst_flat: Path, meta: Dict[str, Any]):
    dst_struct.mkdir(parents=True, exist_ok=True)
    pc_file = dst_struct / "point_cloud.npy"
    if pc_file.exists():
        return
    mesh = load_mesh_safe(fpath)
    if mesh is None:
        return
    pts = sample_points_from_mesh(mesh, n_points)
    np.save(pc_file, pts)
    (dst_struct / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    dst_flat.parent.mkdir(parents=True, exist_ok=True)
    if not dst_flat.exists():
        os.symlink(dst_struct, dst_flat, target_is_directory=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--n_points", type=int, default=8192)
    args = ap.parse_args()

    ufrn_root = Path(args.ufrn_root)
    df = scan_files(ufrn_root)

    out_struct = ufrn_root / "processed_struct" / str(args.n_points)
    out_flat = ufrn_root / "processed_flat" / str(args.n_points)
    out_struct.mkdir(parents=True, exist_ok=True)
    out_flat.mkdir(parents=True, exist_ok=True)

    df.to_csv(ufrn_root / "ufrn_summary.csv", index=False)
    print(f"[CSV] Guardado resumen en: {ufrn_root/'ufrn_summary.csv'}")

    for _, row in tqdm(df.iterrows(), total=len(df), ncols=100, desc="[PROC] Generando nubes"):
        fpath = Path(row["file"])
        pid, jaw = row["patient_id"], row["jaw"]
        tag = "full" if row["is_full_arch"] else (f"rec_{row['missing_tooth']}" if row["missing_tooth"] else "mesh")
        dst_s = out_struct / jaw / f"{pid}_{tag}"
        dst_f = out_flat / jaw / f"{pid}_{tag}"
        meta = {
            "source_file": str(fpath),
            "patient_id": pid,
            "jaw": jaw,
            "is_full_arch": bool(row["is_full_arch"]),
            "is_recortado": bool(row["is_recortado"]),
            "missing_tooth": row["missing_tooth"],
            "n_points": int(args.n_points)
        }
        process_one(fpath, args.n_points, dst_s, dst_f, meta)

    # === RESUMEN AUTOMÁTICO ===
    upper_full = df[df["file"].str.contains(r"(?i)(sup|upper)[_\-\s]?(full)?\.stl") & df["is_full_arch"]]
    upper_21 = df[df["file"].str.contains(r"(?i)(sup|upper)[_\-\s]?21\.stl")]

    patients_full = set(upper_full["patient_id"])
    patients_21 = set(upper_21["patient_id"])

    print("\n=== UFRN SUMMARY ===")
    print(f"Total pacientes: {df['patient_id'].nunique()}")
    print(f"Con upper_full: {len(patients_full)}")
    print(f"Con upper_rec_21: {len(patients_21)}")
    print(f"Pacientes con ambos: {len(patients_full & patients_21)}")

    print("\n✅ Preprocesamiento finalizado correctamente.")
    print(f"Salidas: {out_struct}  y  {out_flat}")

if __name__ == "__main__":
    main()
