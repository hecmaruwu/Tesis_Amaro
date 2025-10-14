#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UFRN audit + (opcional) preprocesado a nubes normalizadas.

Correcciones CLAVE:
- patient_id se toma EXCLUSIVAMENTE de un directorio canónico tipo 'PACIENTE <n>' (regex estricto)
  o de otros directorios con 'paciente_<n>' / 'paciente <n>', pero SIEMPRE de NOMBRES DE CARPETA,
  nunca desde nombres de archivo STL.
- 'ambas arcadas full' exige upper_recortado y lower_recortado del MISMO paciente.
- 'upper con 21 faltante' busca presencia de archivos *_21.stl (u .ply/.obj) del mismo paciente.

Uso:
  Auditoría + CSV:
    python scripts/ufrn_audit_and_preprocess.py \
      --ufrn_root "/home/htaucare/Tesis_dientes_original/data/UFRN" \
      --extract_rel "extract/MODELOS PESQUISA" \
      --summary_csv "ufrn_summary.csv"

  Auditoría + preprocesado a nubes (8192 pts):
    python scripts/ufrn_audit_and_preprocess.py \
      --ufrn_root "/home/htaucare/Tesis_dientes_original/data/UFRN" \
      --extract_rel "extract/MODELOS PESQUISA" \
      --summary_csv "ufrn_summary.csv" \
      --out_struct "processed_struct" \
      --out_flat "processed_flat" \
      --n_points 8192 \
      --do_points
"""

import argparse
import csv
import json
import os
import re
import shutil
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


# ---------- Regex estrictos (SOLO sobre nombres de carpeta) ----------
# Canon: 'PACIENTE 22', 'Paciente_22', 'paciente-22'
CANON_PAT_DIR = re.compile(r"^paciente[ _-]*([0-9]+)$", re.IGNORECASE)

# Dientes en nombre de ARCHIVO: *_21.stl, *_22.ply, etc.
FDI_IN_FILE = re.compile(r"_(\d{2})\.(stl|ply|obj)$", re.IGNORECASE)


def detect_patient_id_from_dirs(path: Path) -> str | None:
    """
    Recorre los padres del path (de hoja a raíz) y devuelve 'paciente_<n>' si
    encuentra un directorio cuyo nombre calza con CANON_PAT_DIR. Nunca mira el archivo.
    """
    for part in reversed(path.parts):
        m = CANON_PAT_DIR.match(part)
        if m:
            return f"paciente_{int(m.group(1))}"
    return None


def detect_jaw_from_path(path: Path) -> str | None:
    """
    Jaw a partir de nombres de carpeta (no archivo).
    - upper si algún segmento contiene 'sup' o 'upper'
    - lower si 'inf' o 'lower'
    """
    for part in path.parts:
        p = part.lower()
        if "upper" in p or "sup" in p:
            return "upper"
        if "lower" in p or "inf" in p:
            return "lower"
    return None


def is_full_arch_file(file_name: str) -> bool:
    """
    Consideramos 'full' si el nombre contiene 'recortado' o algún sufijo _full.*
    """
    fn = file_name.lower()
    return (
        "recortado" in fn
        or fn.endswith("_full.stl")
        or fn.endswith("_full.ply")
        or fn.endswith("_full.obj")
        or "_full." in fn
        or "_full_" in fn
    )


def detect_missing_tooth_from_file(file_name: str) -> str:
    """
    Extrae diente FDI del nombre de archivo si termina con _NN.stl/.ply/.obj.
    (NO se usa para el patient_id).
    """
    m = FDI_IN_FILE.search(file_name)
    return m.group(1) if m else ""


def sample_points_from_mesh(mesh, n_points: int) -> np.ndarray:
    if mesh.faces is not None and len(mesh.faces) > 0:
        pts, _ = tm.sample.sample_surface(mesh, n_points)
    else:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        if v.size == 0:
            raise ValueError("Malla sin vértices")
        idx = np.random.choice(len(v), size=n_points, replace=(len(v) < n_points))
        pts = v[idx]
    return np.asarray(pts, dtype=np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0, keepdims=True)
    pts = points - c
    r = np.linalg.norm(pts, axis=1).max()
    if r > 0:
        pts = pts / r
    return pts


def symlink_or_copy(src_dir: Path, dst_dir: Path, copy_dirs: bool):
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() or dst_dir.is_symlink():
        return
    if copy_dirs:
        shutil.copytree(src_dir, dst_dir)
    else:
        try:
            os.symlink(src_dir, dst_dir, target_is_directory=True)
        except Exception:
            shutil.copytree(src_dir, dst_dir)


# ------------------- Escaneo y CSV -------------------

def scan_files(ufrn_root: Path, extract_rel: str) -> list[dict]:
    base = (ufrn_root / extract_rel)
    if not base.is_dir():
        print(f"[WARN] No existe {base}")
        return []

    rows = []
    exts = (".stl", ".ply", ".obj")
    files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    for f in tqdm(files, desc="[SCAN] STL/PLY/OBJ", unit="file"):
        pid = detect_patient_id_from_dirs(f.parent) or "unknown"
        jaw = detect_jaw_from_path(f.parent) or "unknown"
        is_full = is_full_arch_file(f.name)
        missing_tooth = detect_missing_tooth_from_file(f.name)

        rows.append({
            "file": str(f),
            "patient_id": pid,
            "jaw": jaw,
            "is_full_arch": is_full,
            "is_recortado": ("recortado" in f.name.lower()),
            "missing_tooth": missing_tooth
        })
    return rows


def write_csv(rows: list[dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["file", "patient_id", "jaw", "is_full_arch", "is_recortado", "missing_tooth"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------- Mini auditoría -------------------

def mini_audit(rows: list[dict]):
    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["patient_id"], []).append(r)

    both_full = []
    without_both_full = []
    both_full_upper_missing21 = []

    for pid, items in by_pid.items():
        # ¿tiene upper FULL (recortado) y lower FULL (recortado)?
        has_upper_full = any(it["jaw"] == "upper" and it["is_full_arch"] for it in items)
        has_lower_full = any(it["jaw"] == "lower" and it["is_full_arch"] for it in items)

        if has_upper_full and has_lower_full:
            both_full.append(pid)
            # Upper con 21 faltante: ¿existe algún archivo *_21.* en ese paciente?
            upper_tooth_21 = any(
                (it["jaw"] == "upper" and it["missing_tooth"] == "21")
                for it in items
            )
            if upper_tooth_21:
                both_full_upper_missing21.append(pid)
        else:
            without_both_full.append(pid)

    # IDs reales (excluye 'unknown')
    uniq_pid = sorted(p for p in {r["patient_id"] for r in rows} if p != "unknown")
    unknown_flag = ("unknown" in {r["patient_id"] for r in rows})

    print("\n================ UFRN AUDIT ================")
    print(f"Pacientes totales detectados: {len(uniq_pid)}" + (" (+ unknown)" if unknown_flag else ""))
    print(f"Con ambas arcadas completas: {len(set(both_full))}")
    print(f"Sin ambas arcadas completas: {len(set(without_both_full) - {'unknown'})}")
    print(f"Con ambas completas y upper con 21 faltante: {len(set(both_full_upper_missing21))}")

    def head(lst, n=10): 
        vals = sorted(set([x for x in lst if x != 'unknown']))
        return ", ".join(vals[:n]) + (" ..." if len(vals) > n else "")

    print("\nEjemplos SIN ambas full:", head(without_both_full))
    print("Ejemplos CON ambas full:", head(both_full))
    print("Ejemplos CON ambas full + upper missing 21:", head(both_full_upper_missing21))

    return {
        "patients_total": len(uniq_pid),
        "both_full": sorted(set(p for p in both_full if p != 'unknown')),
        "without_both_full": sorted(set(p for p in without_both_full if p != 'unknown')),
        "both_full_upper_missing21": sorted(set(p for p in both_full_upper_missing21 if p != 'unknown')),
    }


# ------------------- Preprocesado opcional -------------------

def preprocess_points(rows: list[dict], out_struct: Path, out_flat: Path,
                      n_points: int, copy_flat: bool):
    if tm is None:
        print("[WARN] trimesh no disponible; omito muestreo de puntos.")
        return

    out_struct.mkdir(parents=True, exist_ok=True)
    out_flat.mkdir(parents=True, exist_ok=True)

    for r in tqdm(rows, desc="[PREPROC] nubes normalizadas", unit="file"):
        f = Path(r["file"])
        pid = r["patient_id"]; jaw = r["jaw"]
        if pid == "unknown" or jaw not in ("upper", "lower"):
            continue

        # Para no chocar nombres, etiquetamos:
        # - full: *_recortado / *_full.*
        # - toothNN: si missing_tooth está
        tag = "full" if r["is_full_arch"] else (f"tooth{r['missing_tooth']}" if r["missing_tooth"] else "mesh")

        dst_struct = out_struct / jaw / f"{pid}_{tag}"
        pc_file = dst_struct / "point_cloud.npy"
        if pc_file.exists():
            continue

        # Cargar malla
        try:
            mesh = tm.load(f, process=False, force='mesh')
            if isinstance(mesh, tm.Scene):
                mesh = mesh.dump().sum()
        except Exception as e:
            print(f"[ERR] No pude cargar {f}: {e}")
            continue

        try:
            pts = sample_points_from_mesh(mesh, n_points)
            pts = normalize_points(pts)
        except Exception as e:
            print(f"[ERR] Muestreo falló {f}: {e}")
            continue

        dst_struct.mkdir(parents=True, exist_ok=True)
        np.save(pc_file, pts)
        (dst_struct / "meta.json").write_text(json.dumps({
            "source_file": str(f),
            "patient_id": pid,
            "jaw": jaw,
            "is_full_arch": bool(r["is_full_arch"]),
            "is_recortado": bool(r["is_recortado"]),
            "missing_tooth": r["missing_tooth"],
            "n_points": int(n_points),
        }, indent=2), encoding="utf-8")

        dst_flat = out_flat / jaw / f"{pid}_{tag}"
        symlink_or_copy(dst_struct, dst_flat, copy_dirs=copy_flat)


# ------------------- CLI -------------------

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True, help="Raíz UFRN (ej: data/UFRN)")
    ap.add_argument("--extract_rel", default="extract/MODELOS PESQUISA",
                    help="Ruta relativa con los STL/PLY/OBJ desde ufrn_root")
    ap.add_argument("--summary_csv", default="ufrn_summary.csv")
    ap.add_argument("--out_struct", default="processed_struct")
    ap.add_argument("--out_flat", default="processed_flat")
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--do_points", action="store_true", help="Genera nubes normalizadas .npy")
    ap.add_argument("--copy_flat", action="store_true", help="Copia en lugar de symlink")
    return ap


def main():
    args = build_parser().parse_args()

    ufrn_root = Path(args.ufrn_root)
    csv_path = ufrn_root / args.summary_csv
    out_struct = ufrn_root / args.out_struct
    out_flat = ufrn_root / args.out_flat

    rows = scan_files(ufrn_root, args.extract_rel)
    write_csv(rows, csv_path)
    print(f"[CSV] Resumen en: {csv_path}")

    summary = mini_audit(rows)

    if args.do_points:
        preprocess_points(rows, out_struct, out_flat, n_points=args.n_points, copy_flat=args.copy_flat)
        print(f"[OUT] processed_struct: {out_struct}")
        print(f"[OUT] processed_flat  : {out_flat}")

    (ufrn_root / "ufrn_mini_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[JSON] Mini-audit: {ufrn_root/'ufrn_mini_audit.json'}")


if __name__ == "__main__":
    main()
