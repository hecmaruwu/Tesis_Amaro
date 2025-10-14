#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, shutil, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Carpeta del experimento (puede ser la padre o run_single)")
    ap.add_argument("--export_root", default="./figures", help="Carpeta raíz donde guardar todo centralizado")
    ap.add_argument("--tag", default=None, help="Nombre corto para la carpeta de salida; por defecto se deriva del path")
    ap.add_argument("--include_metrics", action="store_true", help="Copia también infer_test_metrics_*.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    export_root = Path(args.export_root).resolve()

    # Detectar subcarpeta de trabajo real (si existe run_single, úsala)
    real_dir = run_dir
    if (run_dir / "run_single").is_dir():
        real_dir = run_dir / "run_single"

    # Buscar directorios vis_* (vis_best, vis_final, etc.)
    vis_dirs = []
    for d in ["vis_best", "vis_final"]:
        if (real_dir / d).is_dir():
            vis_dirs.append(real_dir / d)

    # Si no hay vis_* directo, busca a 1 nivel (por si el script cambia nombres)
    if not vis_dirs:
        for p in real_dir.glob("vis_*"):
            if p.is_dir():
                vis_dirs.append(p)

    if not vis_dirs:
        raise SystemExit(f"No se encontraron carpetas vis_* en {real_dir}")

    # Derivar tag si no se pasa
    tag = args.tag
    if tag is None:
        # usa los últimos 3 componentes del path como nombre base, p.ej: P8192_WCE/P8192_WCE_lr5e-4/run_single
        parts = real_dir.parts
        tag = "_".join(parts[-3:]) if len(parts) >= 3 else parts[-1]

    # Estructura de destino: figures/inference/<tag>/(vis_best|vis_final)
    out_base = export_root / "inference" / tag
    out_base.mkdir(parents=True, exist_ok=True)

    copied = 0
    for vd in vis_dirs:
        target = out_base / vd.name
        target.mkdir(parents=True, exist_ok=True)
        for png in vd.rglob("*.png"):
            rel = png.relative_to(vd)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, dest)
            copied += 1

    # Copiar métricas si se solicita
    if args.include_metrics:
        for jf in real_dir.glob("infer_test_metrics_*.json"):
            shutil.copy2(jf, out_base / jf.name)

    print(f"[OK] Copiadas {copied} imágenes a: {out_base}")
    if args.include_metrics:
        print(f"[OK] Métricas JSON copiadas a: {out_base}")

if __name__ == "__main__":
    main()
