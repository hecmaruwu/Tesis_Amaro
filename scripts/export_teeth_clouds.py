#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np, tensorflow as tf
from pathlib import Path
import trimesh as tm

# --- Clustering opcional ---
try:
    from sklearn.cluster import DBSCAN
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import hdbscan as HDB
    HAVE_HDB = True
except Exception:
    HAVE_HDB = False

# === Paleta fija por etiqueta (RGB 0–255) ===
COLOR_MAP = {
    0:  (230, 190, 138),  # tono piel / tan (CAMBIO)
    11: (  0,   0, 255),  # blue
    12: (  0, 128,   0),  # green
    13: (255, 165,   0),  # orange
    14: (128,   0, 128),  # purple
    15: (  0, 255, 255),  # cyan
    16: (255,   0, 255),  # magenta
    17: (255, 255,   0),  # yellow
    18: (165,  42,  42),  # brown
    21: (  0, 191, 255),  # DeepSkyBlue (CAMBIO)
    22: (  0,   0, 128),  # navy
    23: (  0, 128, 128),  # teal
    24: (238, 130, 238),  # violet
    25: (250, 128, 114),  # salmon
    26: (255, 215,   0),  # gold
    27: (173, 216, 230),  # lightblue
    28: (255, 127,  80),  # coral
    31: (128, 128,   0),  # olive
    32: (192, 192, 192),  # silver
    33: (128, 128, 128),  # gray
    34: (  0,   0,   0),  # black
    35: (139,   0,   0),  # darkred
    36: (  0, 100,   0),  # darkgreen
    37: (  0,   0, 139),  # darkblue
    38: (148,   0, 211),  # darkviolet
    41: (205, 133,  63),  # peru
    42: (210, 105,  30),  # chocolate
    43: (199,  21, 133),  # mediumvioletred
    44: (135, 206, 250),  # lightskyblue
    45: (255, 182, 193),  # lightpink
    46: (221, 160, 221),  # plum
    47: (240, 230, 140),  # khaki
    48: (176, 224, 230),  # powderblue
}

def color_for_label(l: int) -> np.ndarray:
    """Devuelve color RGB uint8 para la etiqueta l."""
    if int(l) in COLOR_MAP:
        return np.array(COLOR_MAP[int(l)], dtype=np.uint8)
    # Fallback determinista para etiquetas desconocidas
    rng = np.random.default_rng(int(l) * 12345 + 7)
    col = rng.integers(50, 235, size=3)
    return col.astype(np.uint8)

def load_best_or_final(run_dir: Path, which: str):
    run_dir = Path(run_dir)
    if (run_dir / "run_single").is_dir():
        run_dir = run_dir / "run_single"
    cand = run_dir / ("checkpoints/best" if which == "best" else "final_model")
    if not cand.exists():
        raise FileNotFoundError(f"No existe el modelo en {cand}")
    print(f"[LOAD] {which} -> {cand}")
    # Carga sin métricas custom
    return tf.keras.models.load_model(str(cand), compile=False)

def save_ply(points_xyz: np.ndarray, rgb: np.ndarray, out_path: Path):
    """Exporta nube como PLY con colores (uint8)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    pc = tm.points.PointCloud(points_xyz, colors=rgb)
    pc.export(out_path)

def cluster_split(xyz: np.ndarray,
                  method: str = "none",
                  dbscan_eps: float = 0.01,
                  dbscan_min_samples: int = 50,
                  hdb_min_cluster_size: int = 50,
                  hdb_min_samples: int | None = None):
    """
    Devuelve lista de (comp_id, pts_comp). Si method='none', 1 componente.
    """
    if xyz.shape[0] == 0:
        return []
    if method == "none":
        return [(None, xyz)]
    if method == "dbscan":
        if not HAVE_SK:
            print("[WARN] DBSCAN no disponible (scikit-learn no instalado). Sin clustering.")
            return [(None, xyz)]
        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(xyz)
        comp_ids = np.unique(labels[labels >= 0])
        if comp_ids.size == 0:
            return [(None, xyz)]
        return [(int(cid), xyz[labels == cid]) for cid in comp_ids]
    if method == "hdbscan":
        if not HAVE_HDB:
            print("[WARN] HDBSCAN no disponible (pip install hdbscan). Sin clustering.")
            return [(None, xyz)]
        clusterer = HDB.HDBSCAN(min_cluster_size=hdb_min_cluster_size,
                                min_samples=hdb_min_samples)
        labels = clusterer.fit_predict(xyz)
        comp_ids = np.unique(labels[labels >= 0])
        if comp_ids.size == 0:
            return [(None, xyz)]
        return [(int(cid), xyz[labels == cid]) for cid in comp_ids]
    # fallback
    return [(None, xyz)]

def export_one_set(xyz: np.ndarray, lab_vec: np.ndarray, out_dir: Path,
                   skip_label0: bool, method: str,
                   dbscan_eps: float, dbscan_min_samples: int,
                   hdb_min_cluster_size: int, hdb_min_samples: int | None,
                   prefix: str):
    """
    Exporta por clase (y opcionalmente por componente) a out_dir/<prefix>_by_class/.
    """
    uniq = np.unique(lab_vec)
    for l in uniq:
        if skip_label0 and int(l) == 0:
            continue
        mask = (lab_vec == l)
        pts = xyz[mask]
        if pts.shape[0] == 0:
            continue
        comps = cluster_split(
            pts, method=method,
            dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
            hdb_min_cluster_size=hdb_min_cluster_size, hdb_min_samples=hdb_min_samples
        )
        if len(comps) == 0:
            continue
        for cid, pts_c in comps:
            rgb = np.repeat(color_for_label(int(l))[None, :], pts_c.shape[0], axis=0)
            sub = out_dir / f"{prefix}_by_class"
            name = f"{prefix}_label{int(l)}" + (f"_c{cid}" if cid is not None else "") + ".ply"
            save_ply(pts_c, rgb, sub / name)

def export_full_color(xyz: np.ndarray, lab_vec: np.ndarray, out_path: Path, skip_label0: bool):
    rgb = np.zeros((xyz.shape[0], 3), np.uint8)
    for l in np.unique(lab_vec):
        if skip_label0 and int(l) == 0:
            continue
        rgb[lab_vec == l] = color_for_label(int(l))
    save_ply(xyz, rgb, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/exp8192_wce (o carpeta padre)")
    ap.add_argument("--data_dir", required=True, help="data/3dteethseg/splits/8192_seed42_pairs")
    ap.add_argument("--which_model", default="best", choices=["best","final"])
    ap.add_argument("--out_dir", required=True, help="carpeta destino (e.g., figures/teeth_clouds)")
    ap.add_argument("--samples", type=int, default=6, help="cuántas muestras del test exportar")
    ap.add_argument("--skip_label0", action="store_true", help="ignorar clase 0 (fondo)")

    # Clustering
    ap.add_argument("--cluster", default="none", choices=["none", "dbscan", "hdbscan"],
                    help="método de clustering por clase")
    ap.add_argument("--dbscan_eps", type=float, default=0.01, help="eps DBSCAN (espacio normalizado)")
    ap.add_argument("--dbscan_min_samples", type=int, default=50, help="min_samples DBSCAN")
    ap.add_argument("--hdb_min_cluster_size", type=int, default=50, help="min_cluster_size HDBSCAN")
    ap.add_argument("--hdb_min_samples", type=int, default=-1,
                    help="min_samples HDBSCAN (usa -1 para dejar None)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Datos
    Xte = np.load(data_dir / "X_test.npz")["X"].astype("float32")   # (N, P, 3)
    Yte = np.load(data_dir / "Y_test.npz")["Y"].astype("int32")     # (N, P)

    # Modelo
    model = load_best_or_final(run_dir, args.which_model)
    M = min(args.samples, Xte.shape[0])
    pred = model.predict(Xte[:M], verbose=0)                        # (M, P, C)
    yhat = pred.argmax(-1).astype("int32")                          # (M, P)

    # Param hdbscan
    hdb_min_samples = None if args.hdb_min_samples is None or args.hdb_min_samples < 0 else int(args.hdb_min_samples)

    # Export por muestra
    for i in range(M):
        xyz = Xte[i]
        gt  = Yte[i]
        ph  = yhat[i]

        smp_dir = out_root / f"sample_{i:03d}"
        smp_dir.mkdir(parents=True, exist_ok=True)

        # Por clase (GT / PRED)
        export_one_set(
            xyz, gt, smp_dir, args.skip_label0, args.cluster,
            args.dbscan_eps, args.dbscan_min_samples,
            args.hdb_min_cluster_size, hdb_min_samples, prefix="gt"
        )
        export_one_set(
            xyz, ph, smp_dir, args.skip_label0, args.cluster,
            args.dbscan_eps, args.dbscan_min_samples,
            args.hdb_min_cluster_size, hdb_min_samples, prefix="pred"
        )

        # Full coloreado
        export_full_color(xyz, gt,  smp_dir / "full_gt.ply",   args.skip_label0)
        export_full_color(xyz, ph,  smp_dir / "full_pred.ply", args.skip_label0)

        print(f"[OK] sample {i}: clases_pred={np.unique(ph).tolist()} clases_gt={np.unique(gt).tolist()}")

    print(f"\n✅ Export listo en: {out_root}")
    print("   - Para cada muestra: full_gt.ply, full_pred.ply y carpetas gt_by_class/ pred_by_class/")
    if args.cluster != "none":
        print(f"   - Clustering: {args.cluster} (ver *_c#.ply por componentes)")

if __name__ == "__main__":
    main()
