#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MODELS = {
    "PointNet": "/home/htaucare/Tesis_Amaro/outputs/pointnet_classic/grid_pro_runner_v1/exp03_bs16_lr3e4_do05_bg003_amp/inference",
    "PointNetPP": "/home/htaucare/Tesis_Amaro/outputs/pointnetpp/grid_pro_runner_v2/exp04_bs8_lr3e4_r010_020_040_ns32/inference",
    "DGCNN": "/home/htaucare/Tesis_Amaro/outputs/dgcnn/grid_pro_runner_v2_gpu1/exp06_bs8_lr2e4_k20_emb768_bg003/inference",
    "PointNetTransformer": "/home/htaucare/Tesis_Amaro/outputs/pointnettransformer/grid_pro_runner_v2_gpu0/exp13_bs4_lr2e4_dm256_dep4_h8_ff512_do005/inference",
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)




def load_npy_numeric(path: Path, dtype=None):
    """
    Carga .npy de forma robusta.

    Motivo:
    Algunas exportaciones antiguas pueden quedar como dtype=object.
    Como estos archivos fueron generados por nuestro propio pipeline local,
    se permite allow_pickle=True y luego se fuerza conversión numérica.
    """
    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape == ():
            arr = arr.item()
        else:
            try:
                arr = np.asarray(arr.tolist())
            except Exception:
                arr = np.asarray(arr)

    arr = np.asarray(arr)

    # Si queda como arreglo de objetos con un solo elemento que contiene otro array
    if arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())

    if dtype is not None:
        arr = arr.astype(dtype, copy=False)

    return arr


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path: Path, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def binary_metrics(pred, gt, cls, bg=0, include_bg=False):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    if not include_bg:
        mask = gt != bg
        pred = pred[mask]
        gt = gt[mask]

    if gt.size == 0:
        return 0.0, 0.0, 0.0

    t_pos = gt == cls
    p_pos = pred == cls

    tp = np.logical_and(p_pos, t_pos).sum()
    fp = np.logical_and(p_pos, ~t_pos).sum()
    fn = np.logical_and(~p_pos, t_pos).sum()
    tn = np.logical_and(~p_pos, ~t_pos).sum()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return float(acc), float(f1), float(iou)


def macro_metrics_no_bg(pred, gt, num_classes=15, bg=0):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    mask = gt != bg
    pred = pred[mask]
    gt = gt[mask]

    if gt.size == 0:
        return 0.0, 0.0

    f1s = []
    ious = []

    for c in range(num_classes):
        if c == bg:
            continue

        tp = np.logical_and(pred == c, gt == c).sum()
        fp = np.logical_and(pred == c, gt != c).sum()
        fn = np.logical_and(pred != c, gt == c).sum()

        denom = tp + fp + fn
        if denom == 0:
            continue

        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        f1s.append(f1)
        ious.append(iou)

    if len(f1s) == 0:
        return 0.0, 0.0

    return float(np.mean(f1s)), float(np.mean(ious))


def compute_case_metrics(gt, pred, num_classes=15, bg=0, d21=8):
    gt = gt.astype(np.int64).reshape(-1)
    pred = pred.astype(np.int64).reshape(-1)

    acc_all = float((gt == pred).mean())

    mask = gt != bg
    acc_no_bg = float((gt[mask] == pred[mask]).mean()) if mask.any() else 0.0

    f1_macro, iou_macro = macro_metrics_no_bg(
        pred=pred,
        gt=gt,
        num_classes=num_classes,
        bg=bg,
    )

    d21_acc, d21_f1, d21_iou = binary_metrics(
        pred=pred,
        gt=gt,
        cls=d21,
        bg=bg,
        include_bg=False,
    )

    d21_bin_acc_all, d21_bin_f1_all, d21_bin_iou_all = binary_metrics(
        pred=pred,
        gt=gt,
        cls=d21,
        bg=bg,
        include_bg=True,
    )

    score_composite = (
        0.50 * d21_f1 +
        0.25 * f1_macro +
        0.25 * iou_macro
    )

    return {
        "acc_all": acc_all,
        "acc_no_bg": acc_no_bg,
        "f1_macro": f1_macro,
        "iou_macro": iou_macro,
        "d21_acc": d21_acc,
        "d21_f1": d21_f1,
        "d21_iou": d21_iou,
        "d21_bin_acc_all": d21_bin_acc_all,
        "d21_bin_f1_all": d21_bin_f1_all,
        "d21_bin_iou_all": d21_bin_iou_all,
        "score_composite": float(score_composite),
    }


def load_model_cases(model_name, inference_dir: Path, num_classes=15, bg=0, d21=8):
    manifest_path = inference_dir / "inference_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe manifest: {manifest_path}")

    manifest = read_csv(manifest_path)

    rows = []
    for r in manifest:
        if "gt_npy" not in r or "pred_npy" not in r:
            raise KeyError(
                f"El manifest de {model_name} no tiene columnas gt_npy/pred_npy. "
                f"Debe usar la versión nueva con exportación .npy."
            )

        gt_path = inference_dir / r["gt_npy"]
        pred_path = inference_dir / r["pred_npy"]
        xyz_path = inference_dir / r["xyz_npy"] if r.get("xyz_npy") else ""

        if not gt_path.exists() or not pred_path.exists():
            raise FileNotFoundError(f"Falta .npy en {model_name}: {gt_path} / {pred_path}")

        gt = load_npy_numeric(gt_path, dtype=np.int64)
        pred = load_npy_numeric(pred_path, dtype=np.int64)

        metrics = compute_case_metrics(
            gt=gt,
            pred=pred,
            num_classes=num_classes,
            bg=bg,
            d21=d21,
        )

        out = {
            "model": model_name,
            "row_i": int(r["row_i"]),
            "tag": r.get("tag", ""),
            "sample_name": r.get("sample_name", ""),
            "jaw": r.get("jaw", ""),
            "path": r.get("path", ""),
            "inference_dir": str(inference_dir),
            "png_all": r.get("png_all", ""),
            "png_errors": r.get("png_errors", ""),
            "png_d21": r.get("png_d21", ""),
            "xyz_npy": r.get("xyz_npy", ""),
            "gt_npy": r.get("gt_npy", ""),
            "pred_npy": r.get("pred_npy", ""),
        }
        out.update(metrics)
        rows.append(out)

    return rows


def select_cases_for_group(rows, metric):
    vals = np.array([float(r[metric]) for r in rows], dtype=np.float64)

    mean_val = float(vals.mean())
    median_val = float(np.median(vals))

    best_i = int(np.argmax(vals))
    worst_i = int(np.argmin(vals))
    mean_i = int(np.argmin(np.abs(vals - mean_val)))
    median_i = int(np.argmin(np.abs(vals - median_val)))

    selected = []

    for label, idx, target in [
        ("best", best_i, float(vals[best_i])),
        ("worst", worst_i, float(vals[worst_i])),
        ("closest_to_mean", mean_i, mean_val),
        ("closest_to_median", median_i, median_val),
    ]:
        r = dict(rows[idx])
        r["selection_type"] = label
        r["rank_metric"] = metric
        r["rank_metric_value"] = float(r[metric])
        r["target_value"] = float(target)
        r["group_mean"] = mean_val
        r["group_median"] = median_val
        selected.append(r)

    return selected


def build_global_rows(all_rows, metric):
    by_row = defaultdict(list)

    for r in all_rows:
        key = int(r["row_i"])
        by_row[key].append(r)

    global_rows = []
    for row_i, items in sorted(by_row.items()):
        vals = np.array([float(x[metric]) for x in items], dtype=np.float64)

        first = items[0]
        global_rows.append({
            "scope": "GLOBAL_MEAN_ACROSS_MODELS",
            "row_i": int(row_i),
            "sample_name": first.get("sample_name", ""),
            "jaw": first.get("jaw", ""),
            "path": first.get("path", ""),
            "model_count": len(items),
            f"{metric}_mean_across_models": float(vals.mean()),
            f"{metric}_std_across_models": float(vals.std(ddof=0)),
            "score_composite": float(np.mean([float(x["score_composite"]) for x in items])),
            "d21_f1": float(np.mean([float(x["d21_f1"]) for x in items])),
            "d21_iou": float(np.mean([float(x["d21_iou"]) for x in items])),
            "f1_macro": float(np.mean([float(x["f1_macro"]) for x in items])),
            "iou_macro": float(np.mean([float(x["iou_macro"]) for x in items])),
            "acc_no_bg": float(np.mean([float(x["acc_no_bg"]) for x in items])),
            "models_available": ",".join([x["model"] for x in items]),
        })

    return global_rows


def gaussian_kde_np(values, grid):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return np.zeros_like(grid)

    std = float(values.std(ddof=1))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)

    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma <= 1e-12:
        sigma = std if std > 1e-12 else 1.0

    bw = 0.9 * sigma * (values.size ** (-1 / 5))
    bw = max(float(bw), 1e-6)

    z = (grid[:, None] - values[None, :]) / bw
    dens = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens


def plot_distribution_raw(rows, metric, out_png: Path):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(float(r[metric]))

    plt.figure(figsize=(9, 5), dpi=220)

    all_vals = np.array([float(r[metric]) for r in rows], dtype=np.float64)
    xmin = max(0.0, float(np.percentile(all_vals, 1)) - 0.02)
    xmax = min(1.0, float(np.percentile(all_vals, 99)) + 0.02)

    if xmax <= xmin:
        xmin, xmax = float(all_vals.min()), float(all_vals.max())

    grid = np.linspace(xmin, xmax, 300)

    for model, vals in by_model.items():
        vals = np.array(vals, dtype=np.float64)
        plt.hist(vals, bins=12, density=True, alpha=0.18)
        dens = gaussian_kde_np(vals, grid)
        plt.plot(grid, dens, linewidth=2, label=f"{model} KDE")

    plt.xlabel(metric)
    plt.ylabel("Densidad")
    plt.title(f"Distribución por paciente — {metric} (escala original)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_distribution_zscore(rows, metric, out_png: Path):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(float(r[metric]))

    plt.figure(figsize=(9, 5), dpi=220)

    grid = np.linspace(-3.5, 3.5, 400)

    for model, vals in by_model.items():
        vals = np.array(vals, dtype=np.float64)
        mu = vals.mean()
        sd = vals.std(ddof=0)
        if sd <= 1e-12:
            z = vals * 0.0
        else:
            z = (vals - mu) / sd

        plt.hist(z, bins=12, density=True, alpha=0.16)
        dens = gaussian_kde_np(z, grid)
        plt.plot(grid, dens, linewidth=2, label=f"{model} KDE z-score")

    normal = np.exp(-0.5 * grid * grid) / np.sqrt(2 * np.pi)
    plt.plot(grid, normal, linestyle="--", linewidth=2, label="Normal estándar referencial")

    plt.xlabel(f"{metric} estandarizado por modelo: z = (x - media) / sd")
    plt.ylabel("Densidad")
    plt.title(f"Pseudo-campana comparativa — {metric} con transformación lineal z-score")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_boxplot(rows, metric, out_png: Path):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(float(r[metric]))

    labels = list(by_model.keys())
    data = [by_model[k] for k in labels]

    plt.figure(figsize=(9, 5), dpi=220)
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(metric)
    plt.title(f"Boxplot por modelo — {metric}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()




# ============================================================
# EXTRA PLOTS Y TRANSFORMACIONES NUMÉRICAS
# ============================================================
def _metric_values_by_model(rows, metric):
    by_model = defaultdict(list)
    for r in rows:
        if metric in r:
            try:
                v = float(r[metric])
                if np.isfinite(v):
                    by_model[r["model"]].append(v)
            except Exception:
                pass
    return {k: np.asarray(v, dtype=np.float64) for k, v in by_model.items() if len(v) > 0}


def _normal_score_transform(values):
    """
    Rank-based inverse normal transform usando solo librería estándar.
    Convierte los rangos empíricos a cuantiles de una N(0,1).
    """
    from statistics import NormalDist
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n <= 1:
        return np.zeros_like(values)

    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)

    probs = (ranks - 0.5) / n
    nd = NormalDist()
    return np.asarray([nd.inv_cdf(float(pp)) for pp in probs], dtype=np.float64)


def transform_values(values, transform="raw", eps=1e-5):
    """
    Transformaciones para métricas acotadas en [0,1].
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return x

    t = str(transform).lower()

    if t == "raw":
        return x

    if t == "zscore":
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=0))
        if sd <= 1e-12:
            return x * 0.0
        return (x - mu) / sd

    if t == "robust_zscore":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            return x * 0.0
        return (x - med) / scale

    if t == "minmax":
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo <= 1e-12:
            return x * 0.0
        return (x - lo) / (hi - lo)

    if t == "log1p":
        return np.log1p(np.clip(x, 0.0, None))

    if t == "logit":
        xc = np.clip(x, eps, 1.0 - eps)
        return np.log(xc / (1.0 - xc))

    if t == "error":
        return 1.0 - x

    if t == "neglog10_error":
        err = np.clip(1.0 - x, eps, None)
        return -np.log10(err)

    if t == "arcsin_sqrt":
        xc = np.clip(x, 0.0, 1.0)
        return np.arcsin(np.sqrt(xc))

    if t == "normal_score":
        return _normal_score_transform(x)

    raise ValueError(f"Transformación desconocida: {transform}")


def _pretty_transform_label(metric, transform):
    transform = str(transform)
    labels = {
        "raw": f"{metric}",
        "zscore": f"{metric} z-score",
        "robust_zscore": f"{metric} robust z-score",
        "minmax": f"{metric} min-max",
        "log1p": f"log(1 + {metric})",
        "logit": f"logit({metric})",
        "error": f"1 - {metric}",
        "neglog10_error": f"-log10(1 - {metric} + eps)",
        "arcsin_sqrt": f"arcsin(sqrt({metric}))",
        "normal_score": f"{metric} normal score",
    }
    return labels.get(transform, f"{metric} ({transform})")


def plot_distribution_transform(rows, metric, transform, out_png: Path):
    by_model = _metric_values_by_model(rows, metric)

    plt.figure(figsize=(9, 5), dpi=220)

    transformed_all = []
    transformed_by_model = {}

    for model, vals in by_model.items():
        tv = transform_values(vals, transform=transform)
        if tv.size == 0:
            continue
        transformed_by_model[model] = tv
        transformed_all.append(tv)

    if not transformed_all:
        plt.close()
        return

    all_vals = np.concatenate(transformed_all)
    all_vals = all_vals[np.isfinite(all_vals)]

    if all_vals.size < 2:
        plt.close()
        return

    xmin = float(np.percentile(all_vals, 1))
    xmax = float(np.percentile(all_vals, 99))

    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmin = float(np.min(all_vals))
        xmax = float(np.max(all_vals))

    pad = 0.08 * max(1e-8, xmax - xmin)
    xmin -= pad
    xmax += pad

    grid = np.linspace(xmin, xmax, 350)

    for model, vals in transformed_by_model.items():
        plt.hist(vals, bins=14, density=True, alpha=0.16)
        dens = gaussian_kde_np(vals, grid)
        plt.plot(grid, dens, linewidth=2, label=f"{model}")

    plt.xlabel(_pretty_transform_label(metric, transform))
    plt.ylabel("Densidad")
    plt.title(f"Distribución por paciente — {metric} | transformación: {transform}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_boxplot_transform(rows, metric, transform, out_png: Path):
    by_model = _metric_values_by_model(rows, metric)

    labels = []
    data = []

    for model, vals in by_model.items():
        tv = transform_values(vals, transform=transform)
        if tv.size > 0:
            labels.append(model)
            data.append(tv)

    if not data:
        return

    plt.figure(figsize=(9, 5), dpi=220)
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(_pretty_transform_label(metric, transform))
    plt.title(f"Boxplot — {metric} | transformación: {transform}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_violin_transform(rows, metric, transform, out_png: Path):
    by_model = _metric_values_by_model(rows, metric)

    labels = []
    data = []

    for model, vals in by_model.items():
        tv = transform_values(vals, transform=transform)
        if tv.size > 0:
            labels.append(model)
            data.append(tv)

    if not data:
        return

    plt.figure(figsize=(9, 5), dpi=220)
    plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=20)
    plt.ylabel(_pretty_transform_label(metric, transform))
    plt.title(f"Violin plot — {metric} | transformación: {transform}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_strip_transform(rows, metric, transform, out_png: Path):
    by_model = _metric_values_by_model(rows, metric)
    rng = np.random.default_rng(42)

    labels = []
    data = []

    for model, vals in by_model.items():
        tv = transform_values(vals, transform=transform)
        if tv.size > 0:
            labels.append(model)
            data.append(tv)

    if not data:
        return

    plt.figure(figsize=(9, 5), dpi=220)

    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0.0, 0.045, size=vals.size)
        x = np.full(vals.size, i, dtype=np.float64) + jitter
        plt.scatter(x, vals, s=13, alpha=0.55)

        med = float(np.median(vals))
        mean = float(np.mean(vals))
        plt.plot([i - 0.22, i + 0.22], [med, med], linewidth=2)
        plt.scatter([i], [mean], marker="^", s=55)

    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=20)
    plt.ylabel(_pretty_transform_label(metric, transform))
    plt.title(f"Strip plot por paciente — {metric} | transformación: {transform}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_raincloud_transform(rows, metric, transform, out_png: Path):
    """
    Gráfico combinado: violin + boxplot + puntos.
    No es un raincloud perfecto, pero funciona como figura resumen compacta.
    """
    by_model = _metric_values_by_model(rows, metric)
    rng = np.random.default_rng(42)

    labels = []
    data = []

    for model, vals in by_model.items():
        tv = transform_values(vals, transform=transform)
        if tv.size > 0:
            labels.append(model)
            data.append(tv)

    if not data:
        return

    plt.figure(figsize=(10, 5), dpi=220)

    parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.22)

    plt.boxplot(
        data,
        positions=np.arange(1, len(data) + 1),
        widths=0.18,
        showmeans=True,
        patch_artist=False,
    )

    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(0.18, 0.035, size=vals.size)
        x = np.full(vals.size, i, dtype=np.float64) + jitter
        plt.scatter(x, vals, s=10, alpha=0.45)

    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=20)
    plt.ylabel(_pretty_transform_label(metric, transform))
    plt.title(f"Raincloud compacto — {metric} | transformación: {transform}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_ecdf_raw(rows, metric, out_png: Path):
    by_model = _metric_values_by_model(rows, metric)

    plt.figure(figsize=(9, 5), dpi=220)

    for model, vals in by_model.items():
        vals = np.sort(vals[np.isfinite(vals)])
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        plt.step(vals, y, where="post", linewidth=2, label=model)

    plt.xlabel(metric)
    plt.ylabel("F(x)")
    plt.title(f"ECDF por paciente — {metric} escala original")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_metric_correlation_heatmap(rows, metrics, out_png: Path):
    valid_metrics = [m for m in metrics if m in rows[0]]
    if len(valid_metrics) < 2:
        return

    mat = []
    for r in rows:
        vals = []
        ok = True
        for m in valid_metrics:
            try:
                v = float(r[m])
                if not np.isfinite(v):
                    ok = False
                    break
                vals.append(v)
            except Exception:
                ok = False
                break
        if ok:
            mat.append(vals)

    if len(mat) < 3:
        return

    X = np.asarray(mat, dtype=np.float64)
    corr = np.corrcoef(X, rowvar=False)

    plt.figure(figsize=(9, 8), dpi=220)
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(valid_metrics)), valid_metrics, rotation=45, ha="right")
    plt.yticks(np.arange(len(valid_metrics)), valid_metrics)

    for i in range(len(valid_metrics)):
        for j in range(len(valid_metrics)):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.title("Correlación entre métricas por paciente")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()




def plot_violin_grid_transform(rows, metrics, transform, out_png: Path):
    """
    Genera una grilla de violin plots:
    - una fila/columna por métrica
    - cada subplot compara los modelos
    - útil como figura resumen para tesis
    """
    valid_metrics = [m for m in metrics if len(_metric_values_by_model(rows, m)) > 0]
    if len(valid_metrics) == 0:
        return

    n = len(valid_metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), dpi=220)
    axes = np.asarray(axes).reshape(-1)

    for ax_i, metric in enumerate(valid_metrics):
        ax = axes[ax_i]
        by_model = _metric_values_by_model(rows, metric)

        labels = []
        data = []

        for model, vals in by_model.items():
            tv = transform_values(vals, transform=transform)
            if tv.size > 0:
                labels.append(model)
                data.append(tv)

        if len(data) == 0:
            ax.axis("off")
            continue

        ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel(_pretty_transform_label(metric, transform), fontsize=8)
        ax.grid(alpha=0.25)

    for j in range(len(valid_metrics), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Violin plots por métrica y modelo | transformación: {transform}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def copy_selected_assets(selected_rows, out_dir: Path):
    dst_root = ensure_dir(out_dir / "selected_assets")

    for r in selected_rows:
        model = r.get("model", r.get("scope", "global"))
        sel = r["selection_type"]
        row_i = r["row_i"]
        sample = r.get("sample_name", "")
        tag = r.get("tag", f"row{row_i}")

        safe_model = str(model).replace("/", "_")
        safe_sel = str(sel).replace("/", "_")

        dst_dir = ensure_dir(dst_root / safe_model / safe_sel)

        inf_dir = Path(r["inference_dir"]) if r.get("inference_dir") else None
        if inf_dir is None:
            continue

        for col in ["png_all", "png_errors", "png_d21", "xyz_npy", "gt_npy", "pred_npy"]:
            rel = r.get(col, "")
            if not rel:
                continue

            src = inf_dir / rel
            if src.exists():
                dst_name = f"{safe_model}_{safe_sel}_row{row_i}_{Path(rel).name}"
                shutil.copy2(src, dst_dir / dst_name)


def copy_global_assets(global_selected, all_rows, out_dir: Path):
    dst_root = ensure_dir(out_dir / "selected_assets" / "GLOBAL_MEAN_ACROSS_MODELS")

    by_row = defaultdict(list)
    for r in all_rows:
        by_row[int(r["row_i"])].append(r)

    for g in global_selected:
        sel = g["selection_type"]
        row_i = int(g["row_i"])
        dst_dir = ensure_dir(dst_root / sel / f"row_{row_i}")

        for r in by_row[row_i]:
            inf_dir = Path(r["inference_dir"])
            model = r["model"]

            for col in ["png_all", "png_errors", "png_d21", "xyz_npy", "gt_npy", "pred_npy"]:
                rel = r.get(col, "")
                if not rel:
                    continue

                src = inf_dir / rel
                if src.exists():
                    dst_name = f"{model}_{col}_{Path(rel).name}"
                    shutil.copy2(src, dst_dir / dst_name)




# ============================================================
# COPIA DE MALLAS / ARCHIVOS FUENTE PARA CASOS GLOBALES
# ============================================================
def _safe_name(x):
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")


def discover_mesh_candidates(sample_name="", jaw="", source_path=""):
    """
    Busca archivos de malla / nube / estructura relacionados con el caso.
    Prioridad:
    1) source_path exacto si existe
    2) si source_path es directorio, busca adentro
    3) busca por sample_name en raíces comunes del proyecto
    """
    exts = {".ply", ".stl", ".obj", ".off", ".npz"}
    candidates = []

    def add_if_mesh(fp):
        fp = Path(fp)
        if fp.exists() and fp.is_file() and fp.suffix.lower() in exts:
            candidates.append(fp)

    def add_from_dir(dp):
        dp = Path(dp)
        if not dp.exists() or not dp.is_dir():
            return
        for f in dp.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                candidates.append(f)

    # 1) source_path directo
    if source_path:
        sp = Path(source_path)
        if sp.exists():
            if sp.is_file() and sp.suffix.lower() in exts:
                add_if_mesh(sp)
            elif sp.is_dir():
                add_from_dir(sp)

    # 2) búsqueda por sample_name
    roots = [
        Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds"),
        Path("/home/htaucare/Tesis_Amaro/data/UFRN"),
        Path("/home/htaucare/Tesis_Amaro"),
    ]

    patterns = []
    if sample_name:
        patterns += [
            f"*{sample_name}*",
        ]
    if sample_name and jaw:
        patterns += [
            f"*{sample_name}*{jaw}*",
            f"*{jaw}*{sample_name}*",
        ]

    seen = set(str(x) for x in candidates)

    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for f in root.rglob(pat):
                if f.is_file() and f.suffix.lower() in exts:
                    sf = str(f)
                    if sf not in seen:
                        candidates.append(f)
                        seen.add(sf)

    # Ordena por rutas más cortas primero
    candidates = sorted(set(candidates), key=lambda x: (len(str(x)), str(x)))
    return candidates


def copy_global_meshes(global_selected, out_dir: Path):
    """
    Copia mallas/archivos fuente SOLO para los casos globales:
    best, worst, closest_to_mean, closest_to_median
    """
    mesh_root = ensure_dir(out_dir / "selected_global_meshes")

    for row in global_selected:
        sel = row.get("selection_type", "unknown")
        sample_name = row.get("sample_name", "")
        jaw = row.get("jaw", "")
        source_path = row.get("path", "")
        row_i = row.get("row_i", "")

        dst_dir = ensure_dir(mesh_root / _safe_name(sel))

        candidates = discover_mesh_candidates(
            sample_name=sample_name,
            jaw=jaw,
            source_path=source_path,
        )

        copied = []
        for src in candidates:
            try:
                dst_name = f"{_safe_name(sel)}__row{row_i}__{src.name}"
                dst = dst_dir / dst_name
                shutil.copy2(src, dst)
                copied.append(str(dst))
            except Exception:
                pass

        meta = {
            "selection_type": sel,
            "row_i": row_i,
            "sample_name": sample_name,
            "jaw": jaw,
            "source_path": source_path,
            "score_composite": row.get("score_composite"),
            "d21_f1": row.get("d21_f1"),
            "d21_iou": row.get("d21_iou"),
            "f1_macro": row.get("f1_macro"),
            "iou_macro": row.get("iou_macro"),
            "models_available": row.get("models_available", ""),
            "n_candidates_found": len(candidates),
            "copied_files": copied,
        }
        save_json(meta, dst_dir / "metadata.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--rank_metric", type=str, default="score_composite")
    ap.add_argument("--num_classes", type=int, default=15)
    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument("--d21", type=int, default=8)
    ap.add_argument(
        "--metric_plots",
        type=str,
        default="score_composite,acc_all,acc_no_bg,f1_macro,iou_macro,d21_acc,d21_f1,d21_iou,d21_bin_acc_all,d21_bin_f1_all,d21_bin_iou_all",
        help="Métricas separadas por coma para graficar distribuciones."
    )
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    plots_dir = ensure_dir(out_dir / "plots")

    all_rows = []

    for model_name, inf_dir in DEFAULT_MODELS.items():
        print(f"[LOAD] {model_name}: {inf_dir}", flush=True)
        rows = load_model_cases(
            model_name=model_name,
            inference_dir=Path(inf_dir),
            num_classes=args.num_classes,
            bg=args.bg,
            d21=args.d21,
        )
        all_rows.extend(rows)

    if len(all_rows) == 0:
        raise RuntimeError("No se cargaron casos.")

    if args.rank_metric not in all_rows[0]:
        raise KeyError(f"rank_metric inválida: {args.rank_metric}")

    # Guardar métricas por caso y modelo
    write_csv(all_rows, out_dir / "per_case_metrics_all_models.csv")

    # Resumen por modelo
    summary_rows = []
    for model in sorted(set(r["model"] for r in all_rows)):
        rows = [r for r in all_rows if r["model"] == model]
        vals = np.array([float(r[args.rank_metric]) for r in rows], dtype=np.float64)

        summary_rows.append({
            "model": model,
            "n_cases": len(rows),
            "rank_metric": args.rank_metric,
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
        })

    write_csv(summary_rows, out_dir / "summary_by_model.csv")

    # Selección por modelo
    selected_by_model = []
    for model in sorted(set(r["model"] for r in all_rows)):
        rows = [r for r in all_rows if r["model"] == model]
        selected_by_model.extend(select_cases_for_group(rows, args.rank_metric))

    write_csv(selected_by_model, out_dir / "selected_cases_by_model.csv")

    # Selección global: promedio del score por paciente entre modelos
    global_rows = build_global_rows(all_rows, args.rank_metric)
    write_csv(global_rows, out_dir / "per_case_metrics_global_mean_across_models.csv")

    global_selected = select_cases_for_group(global_rows, args.rank_metric)
    for r in global_selected:
        r["scope"] = "GLOBAL_MEAN_ACROSS_MODELS"

    write_csv(global_selected, out_dir / "selected_cases_global.csv")

    # Copiar assets seleccionados
    copy_selected_assets(selected_by_model, out_dir)
    copy_global_assets(global_selected, all_rows, out_dir)

    # Copiar mallas / archivos fuente SOLO para los casos globales
    copy_global_meshes(global_selected, out_dir)

    # Plots extendidos
    metric_plots = [m.strip() for m in args.metric_plots.split(",") if m.strip()]

    transforms = [
        "raw",
        "zscore",
        "robust_zscore",
        "minmax",
        "log1p",
        "logit",
        "error",
        "neglog10_error",
        "arcsin_sqrt",
        "normal_score",
    ]

    for metric in metric_plots:
        if metric not in all_rows[0]:
            print(f"[WARN] Métrica no existe y se omite: {metric}", flush=True)
            continue

        print(f"[PLOT] Generando gráficos para métrica: {metric}", flush=True)

        # Plots antiguos conservados
        plot_distribution_raw(
            rows=all_rows,
            metric=metric,
            out_png=plots_dir / f"distribution_raw_{metric}.png",
        )

        plot_distribution_zscore(
            rows=all_rows,
            metric=metric,
            out_png=plots_dir / f"distribution_zscore_{metric}.png",
        )

        plot_boxplot(
            rows=all_rows,
            metric=metric,
            out_png=plots_dir / f"boxplot_{metric}.png",
        )

        # Plots nuevos en escala original
        plot_ecdf_raw(
            rows=all_rows,
            metric=metric,
            out_png=plots_dir / f"ecdf_raw_{metric}.png",
        )

        # Plots nuevos con transformaciones
        for tr in transforms:
            safe_tr = tr.replace("/", "_")

            plot_distribution_transform(
                rows=all_rows,
                metric=metric,
                transform=tr,
                out_png=plots_dir / f"distribution_transform_{safe_tr}_{metric}.png",
            )

            plot_boxplot_transform(
                rows=all_rows,
                metric=metric,
                transform=tr,
                out_png=plots_dir / f"boxplot_transform_{safe_tr}_{metric}.png",
            )

            plot_violin_transform(
                rows=all_rows,
                metric=metric,
                transform=tr,
                out_png=plots_dir / f"violin_transform_{safe_tr}_{metric}.png",
            )

            # Los strip plots y raincloud generan muchos PNG;
            # se dejan para las transformaciones más interpretables.
            if tr in ("raw", "zscore", "robust_zscore", "logit", "error", "neglog10_error", "normal_score"):
                plot_strip_transform(
                    rows=all_rows,
                    metric=metric,
                    transform=tr,
                    out_png=plots_dir / f"strip_transform_{safe_tr}_{metric}.png",
                )

                plot_raincloud_transform(
                    rows=all_rows,
                    metric=metric,
                    transform=tr,
                    out_png=plots_dir / f"raincloud_transform_{safe_tr}_{metric}.png",
                )

    plot_metric_correlation_heatmap(
        rows=all_rows,
        metrics=metric_plots,
        out_png=plots_dir / "correlation_heatmap_metrics.png",
    )

    # Grillas resumen de violin plots para todas las métricas
    for tr in ["raw", "zscore", "robust_zscore", "logit", "error", "neglog10_error", "normal_score"]:
        plot_violin_grid_transform(
            rows=all_rows,
            metrics=metric_plots,
            transform=tr,
            out_png=plots_dir / f"violin_grid_transform_{tr}_all_metrics.png",
        )

    save_json(
        {
            "rank_metric": args.rank_metric,
            "num_classes": args.num_classes,
            "bg": args.bg,
            "d21": args.d21,
            "n_total_rows": len(all_rows),
            "n_models": len(DEFAULT_MODELS),
            "models": DEFAULT_MODELS,
            "description": (
                "Se seleccionan mejor, peor, más cercano al promedio y más cercano a la mediana "
                "por modelo y también globalmente promediando los modelos por row_i. "
                "Las distribuciones z-score usan transformación lineal z=(x-media)/sd para comparar formas."
            ),
        },
        out_dir / "run_meta_selection.json",
    )

    print("\n[DONE] Selección terminada.", flush=True)
    print(f"[OUT] {out_dir}", flush=True)
    print(f"[CSV] {out_dir / 'selected_cases_by_model.csv'}", flush=True)
    print(f"[CSV] {out_dir / 'selected_cases_global.csv'}", flush=True)
    print(f"[PLOTS] {plots_dir}", flush=True)


if __name__ == "__main__":
    main()
