#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocesamiento UFRN (targets + ZIP combinado):
- Targets por paciente:
    Upper FULL            : *_sup_recortado.stl  OR  *_sup.stl
    Lower FULL            : *_inf_recortado.stl  OR  *_inf.stl
    Upper con 21 removido : *_sup_21.stl (también acepta *_sup_XX_21.stl y *_sup_29_21.stl)
- Salidas:
    * Nubes normalizadas (8192 pts) en processed_struct/ + processed_flat/
    * CSVs: ufrn_summary_targets.csv y (opcional) ufrn_por_paciente.csv
    * ZIP único con STL + nubes por paciente en una estructura estandarizada
"""

import argparse, os, re, json, shutil, sys, zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import trimesh as tm
from tqdm import tqdm

# -------------------------
# Patrones
# -------------------------
PACIENTE_DIR_RE = re.compile(r"(?i)^paciente[\s_\-]*\d+\s*$")
PACIENTE_NUM_RE = re.compile(r"(?i)paciente[\s_\-]*([0-9]{1,3})")

# FULL explícitos (sin "recortado")
FULL_SUP_FILE_RE = re.compile(r"(?i)(?:^|[_\-\s])(sup|upper)(?:[_\-\s]?(?:full)?)\.stl$")
FULL_INF_FILE_RE = re.compile(r"(?i)(?:^|[_\-\s])(inf|lower)(?:[_\-\s]?(?:full)?)\.stl$")

# Recortados:
RECORTADO_GENERIC_RE = re.compile(r"(?i)(?:^|[_\-\s])(sup|upper|inf|lower)[_\-\s]recortado\.stl$")
# importante: permitir un número opcional entre sup/inf y el diente → *_sup_29_21.stl
RECORTADO_MISSING_RE = re.compile(
    r"(?i)(?:^|[_\-\s])(sup|upper|inf|lower)[_\-\s](?:\d{1,3}[_\-\s])?(1[1-8]|2[1-8]|3[1-8]|4[1-8])\.stl$"
)

# -------------------------
# Helpers de ruta/nombre
# -------------------------
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

def _tok2jaw(tok: str) -> Optional[str]:
    t = tok.lower()
    if t in ("sup", "upper"): return "upper"
    if t in ("inf", "lower"): return "lower"
    return None

def detect_jaw_from_path_or_name(fpath: Path) -> str:
    for seg in fpath.parts:
        if re.search(r"(?i)(^|[_\-])(sup|upper)([_\-]|$)", seg): return "upper"
        if re.search(r"(?i)(^|[_\-])(inf|lower)([_\-]|$)", seg): return "lower"
    low = fpath.name.lower()
    if "_sup" in low or "upper" in low: return "upper"
    if "_inf" in low or "lower" in low: return "lower"
    return "unknown"

def is_full_arch_file(name: str) -> Tuple[bool, Optional[str]]:
    if "recortado" in name.lower(): return False, None
    m = FULL_SUP_FILE_RE.search(name)
    if m: return True, "upper"
    m = FULL_INF_FILE_RE.search(name)
    if m: return True, "lower"
    return False, None

def parse_recortado(name: str) -> Tuple[bool, Optional[str], Optional[int]]:
    m = RECORTADO_MISSING_RE.search(name)
    if m:
        return True, _tok2jaw(m.group(1)), int(m.group(2))
    m = RECORTADO_GENERIC_RE.search(name)
    if m:
        return True, _tok2jaw(m.group(1)), None
    if name.lower().endswith("recortado.stl"):
        return True, None, None
    return False, None, None

# -------------------------
# Geometría
# -------------------------
def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0, keepdims=True)
    pts = pts - c
    r = np.linalg.norm(pts, axis=1).max()
    if r > 0: pts = pts / r
    return pts

def sample_points_from_mesh(mesh: tm.Trimesh, n: int) -> np.ndarray:
    if mesh.is_empty: raise ValueError("Malla vacía.")
    if mesh.faces is not None and len(mesh.faces) > 0:
        pts, _ = tm.sample.sample_surface(mesh, n)
    else:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        if v.shape[0] == 0: raise ValueError("Malla sin vértices.")
        idx = np.random.choice(v.shape[0], size=n, replace=(v.shape[0] < n))
        pts = v[idx]
    return normalize_points(pts.astype(np.float32))

def load_mesh_safe(path: Path) -> Optional[tm.Trimesh]:
    try:
        m = tm.load(path, process=False, force="mesh")
        if isinstance(m, tm.Scene): m = m.dump().sum()
        return m
    except Exception as e:
        print(f"[WARN] No pude cargar '{path}': {e}", file=sys.stderr)
        return None

def symlink_or_copy(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink(): return
    if copy:
        shutil.copytree(src, dst)
    else:
        try:
            os.symlink(src, dst, target_is_directory=True)
        except OSError:
            shutil.copytree(src, dst)

# -------------------------
# Escaneo
# -------------------------
def scan_files(extract_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if not f.lower().endswith((".stl", ".ply", ".obj")):
                continue
            fpath = Path(root) / f
            name = fpath.name
            patient_id = extract_patient_id_from_path(fpath)
            jaw = detect_jaw_from_path_or_name(fpath)

            is_rec, jaw2, miss = parse_recortado(name)
            if is_rec:
                if jaw == "unknown" and jaw2: jaw = jaw2
                rows.append({
                    "file": str(fpath),
                    "patient_id": patient_id,
                    "jaw": jaw,
                    "is_full_arch": False,
                    "is_recortado": True,
                    "missing_tooth": miss,
                })
                continue

            is_full, jaw3 = is_full_arch_file(name)
            if is_full:
                if jaw == "unknown" and jaw3: jaw = jaw3
                rows.append({
                    "file": str(fpath),
                    "patient_id": patient_id,
                    "jaw": jaw,
                    "is_full_arch": True,
                    "is_recortado": False,
                    "missing_tooth": None,
                })
                continue

            # Otros no target
            rows.append({
                "file": str(fpath),
                "patient_id": patient_id,
                "jaw": jaw,
                "is_full_arch": False,
                "is_recortado": False,
                "missing_tooth": None,
            })
    return pd.DataFrame(rows)

# -------------------------
# Filtrado a TARGETS exactos
# -------------------------
def filter_to_targets_ufrn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Targets:
      - FULL_UPPER  : full upper OR recortado upper genérico (missing_tooth NaN)
      - FULL_LOWER  : full lower OR recortado lower genérico (missing_tooth NaN)
      - REC_UPPER_21: recortado upper con missing_tooth == 21
    """
    if df.empty: return df.copy()

    df = df.copy()
    df["kind"] = None

    # Upper FULL
    m_upper_full = ((df["is_full_arch"]) & (df["jaw"] == "upper")) | \
                   ((df["is_recortado"]) & (df["jaw"] == "upper") & (df["missing_tooth"].isna()))
    df.loc[m_upper_full, "kind"] = "FULL_UPPER"

    # Lower FULL
    m_lower_full = ((df["is_full_arch"]) & (df["jaw"] == "lower")) | \
                   ((df["is_recortado"]) & (df["jaw"] == "lower") & (df["missing_tooth"].isna()))
    df.loc[m_lower_full, "kind"] = df.loc[m_lower_full, "kind"].fillna("FULL_LOWER")

    # Upper 21
    m_upper_21 = (df["is_recortado"]) & (df["jaw"] == "upper") & (df["missing_tooth"] == 21)
    df.loc[m_upper_21, "kind"] = "REC_UPPER_21"

    keep = df[df["kind"].notna()].copy()
    if keep.empty: return keep

    keep.sort_values(["patient_id", "kind", "file"], inplace=True)
    keep = keep.groupby(["patient_id", "kind"], as_index=False).first()
    return keep

# -------------------------
# Procesamiento → nubes
# -------------------------
def _process_one(fpath: Path, n_points: int, dst_struct: Path, dst_flat: Path,
                 meta: Dict[str, Any], copy_flat: bool):
    dst_struct.mkdir(parents=True, exist_ok=True)
    pc_file = dst_struct / "point_cloud.npy"
    meta_file = dst_struct / "meta.json"
    if pc_file.exists() and meta_file.exists(): return
    mesh = load_mesh_safe(fpath)
    if mesh is None: return
    pts = sample_points_from_mesh(mesh, n_points)
    np.save(pc_file, pts)
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    symlink_or_copy(dst_struct, dst_flat, copy=copy_flat)

def process_df(df: pd.DataFrame, out_struct: Path, out_flat: Path, n_points: int, copy_flat: bool):
    if df.empty:
        print("[WARN] No hay archivos a procesar.")
        return
    print(f"[PROC] Generando nubes para {len(df)} archivos (targets/selección actual)...")
    for _, row in tqdm(df.iterrows(), total=len(df), ncols=100):
        fpath = Path(row["file"])
        pid   = row["patient_id"]
        jaw   = row["jaw"]
        kind  = row.get("kind")

        if kind == "FULL_UPPER":
            dst_s = out_struct / "upper" / f"{pid}_full"
            dst_f = out_flat   / "upper" / f"{pid}_full"
            mode  = "SURF_SAMPLE_FULL_UPPER"
        elif kind == "FULL_LOWER":
            dst_s = out_struct / "lower" / f"{pid}_full"
            dst_f = out_flat   / "lower" / f"{pid}_full"
            mode  = "SURF_SAMPLE_FULL_LOWER"
        elif kind == "REC_UPPER_21":
            dst_s = out_struct / "upper" / f"{pid}_rec_21"
            dst_f = out_flat   / "upper" / f"{pid}_rec_21"
            mode  = "SURF_SAMPLE_REC_UPPER_21"
        else:
            dst_s = out_struct / jaw / pid
            dst_f = out_flat   / jaw / pid
            mode  = "SURF_SAMPLE_OTHER"

        meta = {"source_file": str(fpath), "patient_id": pid, "jaw": jaw,
                "n_points": int(n_points), "mode": mode}
        _process_one(fpath, n_points, dst_s, dst_f, meta, copy_flat)

# -------------------------
# Export ZIP (STL + nubes)
# -------------------------
def export_targets_zip_combined(df_targets: pd.DataFrame, out_struct: Path,
                                export_root: Path, zip_path: Path):
    """
    Estructura exportada:
      export_root/
        paciente_XX/
          stl/
            upper_full.stl
            lower_full.stl
            upper_rec_21.stl
          clouds/
            upper_full/
              point_cloud.npy
              meta.json
            lower_full/
              point_cloud.npy
              meta.json
            upper_rec_21/
              point_cloud.npy
              meta.json
    """
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    # Copiar STL estandarizados
    for pid, g in df_targets.groupby("patient_id"):
        pdir = export_root / pid
        (pdir / "stl").mkdir(parents=True, exist_ok=True)
        (pdir / "clouds").mkdir(parents=True, exist_ok=True)

        for _, r in g.iterrows():
            src = Path(r["file"])
            kind = r["kind"]
            if kind == "FULL_UPPER":
                dst = pdir / "stl" / "upper_full.stl"
                clouds_src = out_struct / "upper" / f"{pid}_full"
                clouds_dst = pdir / "clouds" / "upper_full"
            elif kind == "FULL_LOWER":
                dst = pdir / "stl" / "lower_full.stl"
                clouds_src = out_struct / "lower" / f"{pid}_full"
                clouds_dst = pdir / "clouds" / "lower_full"
            elif kind == "REC_UPPER_21":
                dst = pdir / "stl" / "upper_rec_21.stl"
                clouds_src = out_struct / "upper" / f"{pid}_rec_21"
                clouds_dst = pdir / "clouds" / "upper_rec_21"
            else:
                continue

            # STL
            shutil.copy2(src, dst)

            # Nubes (si existen)
            if clouds_src.exists():
                clouds_dst.mkdir(parents=True, exist_ok=True)
                for f in ["point_cloud.npy", "meta.json"]:
                    s = clouds_src / f
                    if s.exists():
                        shutil.copy2(s, clouds_dst / f)

    # Empaquetar ZIP
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(export_root):
            for f in files:
                abspath = Path(root) / f
                arcname = abspath.relative_to(export_root)
                zf.write(abspath, arcname)
    print(f"[EXPORT] ZIP combinado escrito en: {zip_path}")
    print(f"[EXPORT] Carpeta temporal con contenido: {export_root}")

# -------------------------
# Auditoría mínima
# -------------------------
def audit_and_print(df: pd.DataFrame, *, targets_only: bool = False):
    header = "UFRN AUDIT (TARGETS)" if targets_only else "UFRN AUDIT"
    pats = sorted(df["patient_id"].unique().tolist()) if not df.empty else []
    print(f"\n================ {header} ================")
    print(f"Pacientes totales detectados: {len(pats)}")
    if df.empty: return
    if "kind" in df.columns:
        by = df.groupby("kind")["patient_id"].nunique().to_dict()
        print(f"Con upper FULL     : {by.get('FULL_UPPER',0)}")
        print(f"Con lower FULL     : {by.get('FULL_LOWER',0)}")
        print(f"Con upper REC 21   : {by.get('REC_UPPER_21',0)}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--extract_rel", default="extract/MODELOS PESQUISA")
    ap.add_argument("--out_struct", default="processed_struct")
    ap.add_argument("--out_flat", default="processed_flat")
    ap.add_argument("--summary_csv", default="ufrn_summary.csv")
    ap.add_argument("--per_paciente_csv", default=None)
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--copy_flat", action="store_true")
    ap.add_argument("--only_targets", action="store_true",
                    help="Procesa únicamente {upper FULL, lower FULL, upper REC 21}")
    ap.add_argument("--export_zip", default=None,
                    help="Ruta del ZIP combinado (STL + nubes), ej: data/UFRN/ufrn_targets_full.zip")
    ap.add_argument("--export_root", default="targets_export",
                    help="Carpeta temporal de export")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    ufrn_root = Path(args.ufrn_root)
    extract_dir = ufrn_root / args.extract_rel
    out_struct = ufrn_root / args.out_struct
    out_flat = ufrn_root / args.out_flat
    summary_csv = ufrn_root / args.summary_csv
    per_paciente_csv = Path(args.per_paciente_csv) if args.per_paciente_csv else None
    export_root = ufrn_root / args.export_root
    zip_path = Path(args.export_zip) if args.export_zip else None

    if not extract_dir.is_dir():
        print(f"[ERR] No existe carpeta: {extract_dir}", file=sys.stderr)
        sys.exit(1)

    if args.clean:
        print("[CLEAN] Eliminando salidas previas…")
        for p in [out_struct, out_flat, export_root]:
            if p.exists(): shutil.rmtree(p)
        for p in [summary_csv, zip_path]:
            if p and p.exists(): p.unlink()
        if per_paciente_csv and per_paciente_csv.exists(): per_paciente_csv.unlink()

    # 1) Escaneo
    df_all = scan_files(extract_dir)

    # 2) Filtrado (targets o todo)
    if args.only_targets:
        df = filter_to_targets_ufrn(df_all)
        summary_targets = summary_csv.with_name(summary_csv.stem + "_targets.csv")
        summary_targets.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_targets, index=False)
        print(f"[CSV] Resumen TARGETS en: {summary_targets}")
    else:
        df = df_all
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_csv, index=False)
        print(f"[CSV] Resumen en: {summary_csv}")

    # 3) Procesar nubes
    process_df(df, out_struct, out_flat, args.n_points, copy_flat=args.copy_flat)

    # 4) Agregado por paciente (opcional)
    if per_paciente_csv:
        agg = df.copy()
        if "kind" not in agg.columns:
            agg["kind"] = "OTHER"
        piv = (agg.pivot_table(index="patient_id", columns="kind", values="file",
                               aggfunc="count", fill_value=0)
                 .reset_index()
                 .sort_values("patient_id"))
        per_paciente_csv.parent.mkdir(parents=True, exist_ok=True)
        piv.to_csv(per_paciente_csv, index=False)
        print(f"[CSV] Agregado por paciente en: {per_paciente_csv}")

    # 5) Export ZIP combinado (STL + nubes)
    if args.export_zip:
        if "kind" not in df.columns and args.only_targets is False:
            print("[WARN] --export_zip recomendado con --only_targets para exportar solo los 3 objetivos.")
        export_targets_zip_combined(df, out_struct, export_root, zip_path)

    # 6) Auditoría breve
    audit_and_print(df, targets_only=args.only_targets)

if __name__ == "__main__":
    main()
