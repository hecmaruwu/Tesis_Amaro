#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UFRN audit + preprocesado automático adaptado a:
  /home/htaucare/Tesis_Amaro/data/UFRN

✔ Busca automáticamente la subcarpeta que contenga “MODELOS PESQUISA”
✔ Detecta pacientes, arcadas (upper/lower), mallas full y dientes ausentes
✔ Genera nubes normalizadas (8192 pts por defecto)
"""

import argparse, csv, json, os, re, shutil
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

# ---------- Regex de detección ----------
CANON_PAT_DIR = re.compile(r"^paciente[ _-]*([0-9]+)$", re.IGNORECASE)
FDI_IN_FILE = re.compile(r"_(\d{2})\.(stl|ply|obj)$", re.IGNORECASE)

def detect_patient_id_from_dirs(path: Path) -> str | None:
    for part in reversed(path.parts):
        m = CANON_PAT_DIR.match(part)
        if m:
            return f"paciente_{int(m.group(1))}"
    return None

def detect_jaw_from_path(path: Path) -> str | None:
    for part in path.parts:
        p = part.lower()
        if "upper" in p or "sup" in p:
            return "upper"
        if "lower" in p or "inf" in p:
            return "lower"
    return None

def is_full_arch_file(fn: str) -> bool:
    fn = fn.lower()
    return any(k in fn for k in ["recortado", "_full", "full_"])

def detect_missing_tooth_from_file(fn: str) -> str:
    m = FDI_IN_FILE.search(fn)
    return m.group(1) if m else ""

def find_extract_root(ufrn_root: Path) -> Path | None:
    """
    Busca recursivamente una carpeta que contenga 'MODELOS' y 'PESQUISA'.
    Devuelve la primera coincidencia encontrada.
    """
    for sub in ufrn_root.rglob("*"):
        if sub.is_dir() and re.search(r"modelos.*pesquisa", sub.name, re.IGNORECASE):
            print(f"[AUTO] Carpeta MODELOS PESQUISA detectada: {sub}")
            return sub
    print("[WARN] No se encontró carpeta tipo 'MODELOS PESQUISA'.")
    return None

def scan_files(ufrn_root: Path, extract_root: Path) -> list[dict]:
    rows = []
    if not extract_root or not extract_root.is_dir():
        print(f"[ERR] No existe {extract_root}")
        return rows
    files = [p for p in extract_root.rglob("*") if p.suffix.lower() in (".stl",".ply",".obj")]
    for f in tqdm(files, desc="[SCAN] STL/PLY/OBJ", unit="file"):
        pid = detect_patient_id_from_dirs(f.parent) or "unknown"
        jaw = detect_jaw_from_path(f.parent) or "unknown"
        rows.append({
            "file": str(f),
            "patient_id": pid,
            "jaw": jaw,
            "is_full_arch": is_full_arch_file(f.name),
            "is_recortado": ("recortado" in f.name.lower()),
            "missing_tooth": detect_missing_tooth_from_file(f.name)
        })
    return rows

def mini_audit(rows: list[dict]):
    by_pid = {}
    for r in rows: by_pid.setdefault(r["patient_id"], []).append(r)
    both_full, missing21 = [], []
    for pid, it in by_pid.items():
        has_u = any(x["jaw"]=="upper" and x["is_full_arch"] for x in it)
        has_l = any(x["jaw"]=="lower" and x["is_full_arch"] for x in it)
        if has_u and has_l:
            both_full.append(pid)
            if any(x["jaw"]=="upper" and x["missing_tooth"]=="21" for x in it):
                missing21.append(pid)
    uniq = sorted(p for p in {r["patient_id"] for r in rows} if p!="unknown")
    print(f"\n=== UFRN AUDIT ===\nTotal pacientes: {len(uniq)}")
    print(f"Ambas arcadas full: {len(both_full)}")
    print(f"Ambas full + upper con 21 faltante: {len(missing21)}")
    return {"total":len(uniq),"both_full":both_full,"missing21":missing21}

def sample_points_from_mesh(mesh, n_points: int) -> np.ndarray:
    if mesh.faces is not None and len(mesh.faces)>0:
        pts,_=tm.sample.sample_surface(mesh,n_points)
    else:
        v=np.asarray(mesh.vertices,dtype=np.float32)
        idx=np.random.choice(len(v),size=n_points,replace=(len(v)<n_points))
        pts=v[idx]
    return np.asarray(pts,dtype=np.float32)

def normalize_points(pts: np.ndarray)->np.ndarray:
    c=pts.mean(axis=0,keepdims=True)
    pts-=c
    r=np.linalg.norm(pts,axis=1).max()
    if r>0: pts/=r
    return pts

def preprocess_points(rows, out_struct, out_flat, n_points=8192):
    if tm is None:
        print("[WARN] trimesh no disponible.")
        return
    out_struct.mkdir(parents=True,exist_ok=True)
    out_flat.mkdir(parents=True,exist_ok=True)

    for r in tqdm(rows,desc="[PREPROC]",unit="file"):
        f=Path(r["file"]); pid=r["patient_id"]; jaw=r["jaw"]
        if pid=="unknown" or jaw not in("upper","lower"): continue
        tag="full" if r["is_full_arch"] else (f"tooth{r['missing_tooth']}" if r["missing_tooth"] else "mesh")
        dst=out_struct/jaw/f"{pid}_{tag}"
        pc_file=dst/"point_cloud.npy"
        if pc_file.exists(): continue

        try:
            mesh=tm.load(f,process=False,force='mesh')
            if isinstance(mesh,tm.Scene): mesh=mesh.dump().sum()
        except Exception as e:
            print(f"[ERR] Carga fallida {f}: {e}"); continue

        try:
            pts=sample_points_from_mesh(mesh,n_points)
            pts=normalize_points(pts)
        except Exception as e:
            print(f"[ERR] Muestreo fallido {f}: {e}"); continue

        dst.mkdir(parents=True,exist_ok=True)
        np.save(pc_file,pts)
        meta={"source_file":str(f),"patient_id":pid,"jaw":jaw,
              "is_full_arch":r["is_full_arch"],"missing_tooth":r["missing_tooth"]}
        (dst/"meta.json").write_text(json.dumps(meta,indent=2))
        # copia plana
        flat=out_flat/jaw/f"{pid}_{tag}"
        flat.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(dst,flat,dirs_exist_ok=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ufrn_root",default="/home/htaucare/Tesis_Amaro/data/UFRN",
                        help="Raíz base de datos UFRN")
    parser.add_argument("--summary_csv",default="ufrn_summary.csv")
    parser.add_argument("--out_struct",default="processed_struct")
    parser.add_argument("--out_flat",default="processed_flat")
    parser.add_argument("--n_points",type=int,default=8192)
    parser.add_argument("--do_points",action="store_true")
    args=parser.parse_args()

    root=Path(args.ufrn_root)
    extract=find_extract_root(root)
    rows=scan_files(root,extract)
    csv_path=root/args.summary_csv
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()) if rows else ["file","patient_id","jaw"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[CSV] Resumen en {csv_path}")
    audit=mini_audit(rows)

    if args.do_points:
        preprocess_points(rows,root/args.out_struct,root/args.out_flat,args.n_points)
        print(f"[OUT] processed_struct: {root/args.out_struct}")
        print(f"[OUT] processed_flat  : {root/args.out_flat}")

    (root/"ufrn_mini_audit.json").write_text(json.dumps(audit,indent=2))
    print(f"[JSON] Mini audit: {root/'ufrn_mini_audit.json'}")

if __name__=="__main__":
    main()
