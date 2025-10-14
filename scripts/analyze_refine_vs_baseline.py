#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_refine_vs_baseline.py

Compara métricas "baseline" vs. "refinadas" (DBSCAN/HDBSCAN) y genera
gráficos con Plotly + visuales 3D por paciente.

Requiere: numpy, pandas, plotly
Opcionales para 3D por paciente:
- processed_struct con point_cloud.npy
- refine_tooth21_* con --export_best_ply (para mostrar el clúster refinado)

Ejemplo:
  python scripts/analyze_refine_vs_baseline.py \
    --ufrn_root /path/UFRN \
    --refine_dir refine_tooth21_hdbscan \
    --baseline_csv /path/UFRN/ufrn_metrics_pointnetpp_8k_s42.csv \
    --struct_rel processed_struct \
    --out_dir analysis_refine_report \
    --max_patients_3d 12
"""

import os, argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Utilidades
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_refine(refine_dir: Path):
    grid = pd.read_csv(refine_dir/"grid_metrics.csv")
    best = pd.read_csv(refine_dir/"best_by_patient.csv")
    # normaliza NaNs/strings vacías
    for col in ["eps","min_samples","min_cluster_size","hdb_min_samples"]:
        if col in best.columns:
            best[col] = best[col].replace("", np.nan)
    return grid, best

def read_baseline(baseline_csv: Path):
    """
    Espera columnas tipo: patient_id, Chamfer, Hausdorff, ...
    Ajusta aquí si tus nombres cambian.
    """
    if not baseline_csv or not baseline_csv.is_file():
        return None
    df = pd.read_csv(baseline_csv)
    # Normaliza col id
    if "patient_id" not in df.columns:
        # intenta inferir
        for c in df.columns:
            if re.match(r"^paciente_\d+$", str(df[c].iloc[0])):
                df = df.rename(columns={c: "patient_id"})
                break
    return df

def save_plot(fig, path: Path):
    ensure_dir(path.parent)
    fig.write_html(str(path), include_plotlyjs="cdn")

# ---------------------------
# Carga nubes / PLY ASCII
# ---------------------------
def load_npy_cloud(p: Path) -> np.ndarray:
    pts = np.load(p).astype(np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 3
    return pts

def parse_ply_ascii(ply_path: Path) -> np.ndarray:
    """
    Lee PLY ASCII con cabecera simple y devuelve Nx3.
    Ignora color si existe.
    """
    with open(ply_path, "r", encoding="utf-8") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY sin end_header")
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        # propiedades
        props = [h for h in header if h.startswith("property ")]
        has_color = any("uchar red" in h for h in props)
        # cargar puntos
        pts = []
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            pts.append((x, y, z))
        return np.array(pts, dtype=np.float32)

# ---------------------------
# Locate files in processed_struct
# ---------------------------
def find_upper_full(struct_root: Path, pid: str) -> Path | None:
    candidates = [
        struct_root/"upper"/f"{pid}_full"/"point_cloud.npy",
        struct_root/"upper"/f"{pid}_upper_full"/"point_cloud.npy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None

def find_best_ply(refine_dir: Path, pid: str) -> Path | None:
    # generado por refine_tooth21_* si usaste --export_best_ply
    p = refine_dir/"ply"/pid/f"{pid}_tooth21_best.ply"
    return p if p.is_file() else None

# ---------------------------
# 3D por paciente
# ---------------------------
def plot_patient_3d(pid: str, struct_root: Path, refine_dir: Path, out_dir: Path,
                    sample_full: int = 8192, color_full="#B0B0B0", color_best="#E74C3C"):
    npy = find_upper_full(struct_root, pid)
    ply = find_best_ply(refine_dir, pid)
    if npy is None:
        return False
    X = load_npy_cloud(npy)
    if X.shape[0] > sample_full:
        idx = np.random.default_rng(42).choice(X.shape[0], sample_full, replace=False)
        X = X[idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X[:,0], y=X[:,1], z=X[:,2],
        mode='markers',
        marker=dict(size=2, color=color_full, opacity=0.4),
        name="upper_full"
    ))
    if ply is not None:
        P = parse_ply_ascii(ply)
        fig.add_trace(go.Scatter3d(
            x=P[:,0], y=P[:,1], z=P[:,2],
            mode='markers',
            marker=dict(size=3, color=color_best, opacity=0.9),
            name="best_cluster (21)"
        ))
    fig.update_layout(
        title=f"{pid} — upper_full + best_cluster",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        height=700
    )
    save_plot(fig, out_dir/f"figs/{pid}_3d.html")
    return True

# ---------------------------
# Heatmaps de hiperparámetros
# ---------------------------
def heatmap_dbscan(grid: pd.DataFrame, out_dir: Path):
    g = grid[grid["method"]=="dbscan"].copy()
    if g.empty: return
    # Cast
    g["eps"] = pd.to_numeric(g["eps"], errors="coerce")
    g["min_samples"] = pd.to_numeric(g["min_samples"], errors="coerce")
    # Métrica promedio por (eps, min_samples)
    pivot = g.pivot_table(index="eps", columns="min_samples", values="f1", aggfunc="mean")
    fig = px.imshow(pivot, text_auto=".2f", aspect="auto",
                    title="DBSCAN — F1 medio por (eps, min_samples)")
    save_plot(fig, out_dir/"figs/dbscan_heatmap_f1.html")

def heatmap_hdbscan(grid: pd.DataFrame, out_dir: Path):
    g = grid[grid["method"]=="hdbscan"].copy()
    if g.empty: return
    g["min_cluster_size"] = pd.to_numeric(g["min_cluster_size"], errors="coerce")
    g["hdb_min_samples"]  = pd.to_numeric(g["hdb_min_samples"], errors="coerce")
    pivot = g.pivot_table(index="min_cluster_size", columns="hdb_min_samples", values="f1", aggfunc="mean")
    fig = px.imshow(pivot, text_auto=".2f", aspect="auto",
                    title="HDBSCAN — F1 medio por (min_cluster_size, min_samples)")
    save_plot(fig, out_dir/"figs/hdbscan_heatmap_f1.html")

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--refine_dir", required=True, help="Carpeta con grid_metrics.csv y best_by_patient.csv")
    ap.add_argument("--baseline_csv", default=None, help="CSV de baseline (ej: ufrn_metrics_pointnetpp_8k_s42.csv)")
    ap.add_argument("--struct_rel", default="processed_struct")
    ap.add_argument("--out_dir", default="analysis_refine_report")
    ap.add_argument("--max_patients_3d", type=int, default=12)
    args = ap.parse_args()

    ufrn_root = Path(args.ufrn_root)
    refine_dir = Path(args.refine_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir/"figs")

    # Cargar refinado
    grid, best = read_refine(refine_dir)
    # Stats refinado
    ref_stats = {
        "refine_mean_chamfer": float(best["chamfer"].mean()),
        "refine_mean_f1": float(best["f1"].mean()),
        "refine_n": int(best.shape[0]),
    }

    # Cargar baseline (opcional)
    base = read_baseline(Path(args.baseline_csv)) if args.baseline_csv else None
    if base is not None:
        # Normalizar nombres de columnas (ajusta si tus CSVs usan otros)
        # Buscamos 'Chamfer' y/o 'F1'
        for c in base.columns:
            if c.lower() == "chamfer":
                base = base.rename(columns={c:"Chamfer"})
            if c.lower() == "f1" or c.lower()=="f1_macro":
                base = base.rename(columns={c:"F1"})
        # Merge por paciente
        merged = pd.merge(best, base, on="patient_id", how="left", suffixes=("_refine", "_base"))
        # Deltas (negativo en Chamfer es bueno, positivo en F1 es bueno)
        if "Chamfer" in merged.columns:
            merged["delta_chamfer"] = merged["Chamfer"] - merged["chamfer"]
        else:
            merged["delta_chamfer"] = np.nan
        if "F1" in merged.columns:
            merged["delta_f1"] = merged["f1"] - merged["F1"]
        else:
            merged["delta_f1"] = np.nan

        # CSVs
        merged.to_csv(out_dir/"compare_by_patient.csv", index=False)
        cmp_summary = {
            **ref_stats,
            "baseline_mean_chamfer": float(merged["Chamfer"].mean()) if "Chamfer" in merged.columns else np.nan,
            "baseline_mean_f1": float(merged["F1"].mean()) if "F1" in merged.columns else np.nan,
            "delta_chamfer_mean": float(merged["delta_chamfer"].mean(skipna=True)),
            "delta_f1_mean": float(merged["delta_f1"].mean(skipna=True)),
        }
        pd.DataFrame([cmp_summary]).to_csv(out_dir/"compare_summary.csv", index=False)

        # Plots
        if "Chamfer" in merged.columns:
            fig = px.histogram(merged, x="Chamfer", nbins=30, title="Baseline Chamfer")
            save_plot(fig, out_dir/"figs/hist_chamfer_baseline.html")
        fig = px.histogram(merged, x="chamfer", nbins=30, title="Refined Chamfer")
        save_plot(fig, out_dir/"figs/hist_chamfer_refined.html")

        if "F1" in merged.columns:
            fig = px.histogram(merged, x="F1", nbins=30, title="Baseline F1")
            save_plot(fig, out_dir/"figs/hist_f1_baseline.html")
        fig = px.histogram(merged, x="f1", nbins=30, title="Refined F1")
        save_plot(fig, out_dir/"figs/hist_f1_refined.html")

        if "delta_chamfer" in merged.columns:
            fig = px.histogram(merged, x="delta_chamfer", nbins=30, title="Δ Chamfer (baseline - refined)")
            save_plot(fig, out_dir/"figs/hist_delta_chamfer.html")
        if "delta_f1" in merged.columns:
            fig = px.histogram(merged, x="delta_f1", nbins=30, title="Δ F1 (refined - baseline)")
            save_plot(fig, out_dir/"figs/hist_delta_f1.html")

        # Scatter P-R (refined)
        fig = px.scatter(merged, x="precision", y="recall", color="method",
                         title="Refined Precision vs Recall",
                         hover_data=["patient_id","eps","min_samples","min_cluster_size","hdb_min_samples"])
        save_plot(fig, out_dir/"figs/scatter_pr_refined.html")
    else:
        # Sin baseline: igual guardamos stats y plots del refinado
        best.to_csv(out_dir/"refined_best_by_patient.csv", index=False)
        pd.DataFrame([ref_stats]).to_csv(out_dir/"refined_summary.csv", index=False)
        fig = px.histogram(best, x="chamfer", nbins=30, title="Refined Chamfer")
        save_plot(fig, out_dir/"figs/hist_chamfer_refined.html")
        fig = px.histogram(best, x="f1", nbins=30, title="Refined F1")
        save_plot(fig, out_dir/"figs/hist_f1_refined.html")
        fig = px.scatter(best, x="precision", y="recall", color="method",
                         title="Refined Precision vs Recall",
                         hover_data=["patient_id","eps","min_samples","min_cluster_size","hdb_min_samples"])
        save_plot(fig, out_dir/"figs/scatter_pr_refined.html")

    # Heatmaps hiperparámetros
    heatmap_dbscan(grid, out_dir)
    heatmap_hdbscan(grid, out_dir)

    # Visuales 3D por paciente (subset)
    struct_root = ufrn_root / args.struct_rel
    # tomar primeros N pacientes (ordenados por peor chamfer, por ejemplo)
    pids = best.sort_values("chamfer", ascending=False)["patient_id"].tolist()
    picked = pids[:int(args.max_patients_3d)]
    made = 0
    for pid in picked:
        ok = plot_patient_3d(pid, struct_root, refine_dir, out_dir)
        if ok: made += 1

    # Índice simple
    index_md = ["# Reporte refinamiento (DBSCAN/HDBSCAN)"]
    if (out_dir/"compare_summary.csv").is_file():
        index_md.append("- [compare_summary.csv](compare_summary.csv)")
    if (out_dir/"compare_by_patient.csv").is_file():
        index_md.append("- [compare_by_patient.csv](compare_by_patient.csv)")
    index_md += [
        "- Figuras:",
        "  - [hist_chamfer_refined](figs/hist_chamfer_refined.html)",
        "  - [hist_f1_refined](figs/hist_f1_refined.html)",
        "  - [scatter_pr_refined](figs/scatter_pr_refined.html)",
        "  - [dbscan_heatmap_f1](figs/dbscan_heatmap_f1.html)",
        "  - [hdbscan_heatmap_f1](figs/hdbscan_heatmap_f1.html)",
        f"- 3D por paciente (generados: {made}): carpeta [figs](figs/)"
    ]
    (out_dir/"INDEX.md").write_text("\n".join(index_md), encoding="utf-8")
    print(f"[DONE] Reporte en: {out_dir}")

if __name__ == "__main__":
    main()
