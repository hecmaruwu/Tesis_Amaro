#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_metrics_UFRN_inference.py
---------------------------------
Diagnóstico de inferencia UFRN:
- Lee CSV de métricas (Chamfer/Hausdorff).
- Imprime estadísticas, histos y TOP peores pacientes.
- Genera gráficos 3D de la malla segmentada (pred) vs CAD (gt upper_rec_21).
  * Si matplotlib no está disponible, exporta PLYs coloreados para inspección externa.

Requisitos blandos:
- matplotlib (opcional; si falta, se saltan PNGs pero se exportan PLYs).
- trimesh (opcional; solo para PLYs, si falta, guarda solo NPY de colores).

Ejecemplo:
python scripts/analyze_metrics_UFRN_inference.py \
  --csv /home/htaucare/Tesis_dientes_original/data/UFRN/ufrn_metrics_pointnetpp_8k_s42.csv \
  --ufrn_root /home/htaucare/Tesis_dientes_original/data/UFRN \
  --struct_rel processed_struct \
  --pred_rel preds_pointnetpp_8k_s42 \
  --out_dir ufrn_diag \
  --max_plots 15
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# matplotlib es opcional (por tus issues de GLIBCXX)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (necesario para proyección 3d)
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

# trimesh es opcional para exportar PLYs coloreados
try:
    import trimesh as tm
    _HAS_TM = True
except Exception:
    _HAS_TM = False


def load_cloud(path: Path) -> np.ndarray:
    """Carga un .npy de puntos (P,3)."""
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Nube inválida {path}: shape {arr.shape}")
    return arr


def normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    """Normalización para visualización consistente."""
    c = pts.mean(axis=0, keepdims=True)
    pts = pts - c
    r = np.linalg.norm(pts, axis=1).max()
    return pts / (r if r > 0 else 1.0)


def scatter3(ax, pts: np.ndarray, s=1, color="k", alpha=0.9, title=None):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=color, alpha=alpha, depthshade=False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    if title:
        ax.set_title(title)


def save_overlay_png(out_png: Path, pred: np.ndarray, gt: np.ndarray, title: str):
    """Guarda figura con GT (gris) y Pred (rojo) normalizados."""
    if not _HAS_MPL:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pred_n = normalize_unit_sphere(pred)
    gt_n = normalize_unit_sphere(gt)

    fig = plt.figure(figsize=(10, 3.2), dpi=150)
    # 3 vistas: GT, Pred, Overlay
    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        if i == 1:
            scatter3(ax, gt_n, s=1, color="0.5", alpha=0.9, title="GT upper_rec_21")
        elif i == 2:
            scatter3(ax, pred_n, s=1, color="tab:red", alpha=0.9, title="Predicción (upper rec 21)")
        else:
            scatter3(ax, gt_n, s=1, color="0.6", alpha=0.6, title="Overlay (GT gris + Pred rojo)")
            ax.scatter(pred_n[:, 0], pred_n[:, 1], pred_n[:, 2], s=1, c="tab:red", alpha=0.9, depthshade=False)
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def export_overlay_ply(out_dir: Path, pid: str, pred: np.ndarray, gt: np.ndarray):
    """
    Exporta PLYs coloreados: GT gris, Pred rojo.
    - Si hay trimesh: guarda 2 PLYs individuales y 1 combinado.
    - Si no hay trimesh: guarda .npy con colores (fallback).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if _HAS_TM:
        # GT gris
        gt_colors = np.full((gt.shape[0], 4), [180, 180, 180, 255], dtype=np.uint8)
        pred_colors = np.full((pred.shape[0], 4), [220, 30, 30, 255], dtype=np.uint8)

        gt_pc = tm.PointCloud(gt, colors=gt_colors)
        pr_pc = tm.PointCloud(pred, colors=pred_colors)

        gt_pc.export(out_dir / f"{pid}_gt_rec21.ply")
        pr_pc.export(out_dir / f"{pid}_pred_rec21.ply")

        # Combinado
        comb = np.vstack([gt, pred])
        comb_col = np.vstack([gt_colors, pred_colors])
        tm.PointCloud(comb, colors=comb_col).export(out_dir / f"{pid}_overlay_gt_pred.ply")
    else:
        # Fallback: NPY con colores (para uso posterior si hay herramientas).
        np.save(out_dir / f"{pid}_gt_rec21.npy", gt)
        np.save(out_dir / f"{pid}_pred_rec21.npy", pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Ruta al CSV de métricas (ej: ufrn_metrics_pointnetpp_8k_s42.csv)")
    ap.add_argument("--ufrn_root", required=True, help="Carpeta base de UFRN (ej: /home/..../data/UFRN)")
    ap.add_argument("--struct_rel", default="processed_struct", help="Relativo a ufrn_root (default: processed_struct)")
    ap.add_argument("--pred_rel", default="preds_tf_seg", help="Relativo a ufrn_root (default: preds_tf_seg)")
    ap.add_argument("--out_dir", default="ufrn_diag", help="Carpeta de salida (PNG/PLY/JSON)")
    ap.add_argument("--max_plots", type=int, default=12, help="Máximo de pacientes para graficar/overlay")
    ap.add_argument("--only_worst", action="store_true", help="Graficar solo los peores (por Hausdorff)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # Asegurar columnas con nombres esperados
    # Permitimos variantes de nombre de columnas
    cols = {c.lower(): c for c in df.columns}
    chamfer_col = cols.get("chamfer", "Chamfer")
    haus_col    = cols.get("hausdorff", "Hausdorff")
    pid_col     = cols.get("patient_id", "patient_id")

    print("[INFO] CSV:", args.csv, "shape:", df.shape)
    print(df.head(3))

    # Stats
    print("\n[STATS] Chamfer")
    print(df[chamfer_col].describe())
    print("\n[STATS] Hausdorff")
    print(df[haus_col].describe())

    # Histos (si hay matplotlib)
    if _HAS_MPL:
        fig1 = plt.figure(figsize=(5, 3.5), dpi=130)
        df[chamfer_col].hist(bins=20)
        plt.title("Distribución Chamfer")
        plt.xlabel("Chamfer"); plt.ylabel("Frecuencia")
        fig1.tight_layout()
        fig1.savefig(out_dir / "hist_chamfer.png")
        plt.close(fig1)

        fig2 = plt.figure(figsize=(5, 3.5), dpi=130)
        df[haus_col].hist(bins=20)
        plt.title("Distribución Hausdorff")
        plt.xlabel("Hausdorff"); plt.ylabel("Frecuencia")
        fig2.tight_layout()
        fig2.savefig(out_dir / "hist_hausdorff.png")
        plt.close(fig2)
        print(f"[PLOT] Histogramas: {out_dir/'hist_chamfer.png'}, {out_dir/'hist_hausdorff.png'}")
    else:
        print("[WARN] matplotlib no disponible: se omiten histogramas PNG.")

    # Top peores (para inspección)
    worst_cd = df.sort_values(chamfer_col, ascending=False).head(10)[[pid_col, chamfer_col]]
    worst_hd = df.sort_values(haus_col,    ascending=False).head(10)[[pid_col, haus_col]]
    print("\n[TOP 10] Peor Chamfer:\n", worst_cd.to_string(index=False))
    print("\n[TOP 10] Peor Hausdorff:\n", worst_hd.to_string(index=False))

    # Guardar resumen JSON
    summary = {
        "stats": {
            "Chamfer": df[chamfer_col].describe().to_dict(),
            "Hausdorff": df[haus_col].describe().to_dict()
        },
        "worst_chamfer_top10": worst_cd.to_dict(orient="records"),
        "worst_hausdorff_top10": worst_hd.to_dict(orient="records"),
        "matplotlib_available": _HAS_MPL,
        "trimesh_available": _HAS_TM
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[JSON] Resumen: {out_dir/'summary.json'}")

    # Gráficos de inferencia: pred vs GT
    ufrn_root   = Path(args.ufrn_root)
    struct_root = ufrn_root / args.struct_rel
    pred_root   = ufrn_root / args.pred_rel

    # Selección de pacientes a graficar
    if args.only_worst:
        patient_list = df.sort_values(haus_col, ascending=False)[pid_col].tolist()
    else:
        # por defecto: orden como aparece en CSV
        patient_list = df[pid_col].tolist()

    plotted = 0
    for pid in patient_list:
        if plotted >= int(args.max_plots):
            break
        # Rutas esperadas
        pred_npy = pred_root / pid / "upper_rec_21_pred.npy"
        gt_npy   = struct_root / "upper" / f"{pid}_rec_21" / "point_cloud.npy"

        if not pred_npy.exists() or not gt_npy.exists():
            print(f"[SKIP] {pid}: falta pred o gt -> {pred_npy.name}, {gt_npy.name}")
            continue

        try:
            pred = load_cloud(pred_npy)
            gt   = load_cloud(gt_npy)

            # PNG overlay
            ok_png = False
            if _HAS_MPL:
                png_path = out_dir / "figs_overlay" / f"{pid}_overlay.png"
                ok_png = save_overlay_png(png_path, pred, gt, title=f"{pid} (pred vs gt)")

            # PLY overlay (colores) si trimesh está disponible; si no, fallback a npy
            overlay_dir = out_dir / "ply_overlay"
            export_overlay_ply(overlay_dir, pid, pred, gt)

            if ok_png:
                print(f"[OK] {pid}: overlay PNG -> {png_path}")
            else:
                print(f"[OK] {pid}: overlay PLY/NPY -> {overlay_dir}")

            plotted += 1
        except Exception as e:
            print(f"[WARN] {pid}: error generando visualización -> {e}")

    print(f"[DONE] Visualizaciones generadas: {plotted}/{len(patient_list)} (máximo {args.max_plots})")
    if not _HAS_MPL:
        print("[NOTE] Instala matplotlib si quieres PNGs dentro del servidor; "
              "por ahora se exportaron PLYs/NPYs para inspección externa.")


if __name__ == "__main__":
    main()
