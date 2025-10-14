#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    r = np.linalg.norm(x, axis=1, keepdims=True).max()
    return x / (r + 1e-6)

def load_label_map(artifacts_dir: Path):
    j = json.loads((artifacts_dir/"label_map.json").read_text(encoding="utf-8"))
    id2idx = {int(k): int(v) for k,v in j["id2idx"].items()}
    idx2id = {int(k): int(v) for k,v in j["idx2id"].items()}
    return id2idx, idx2id

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return cm

def per_class_from_cm(cm):
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)
    pr = cm.sum(axis=0).astype(np.float64)
    prec = np.divide(tp, pr, out=np.zeros_like(tp), where=pr>0)
    rec  = np.divide(tp, gt, out=np.zeros_like(tp), where=gt>0)
    f1   = np.divide(2*prec*rec, (prec+rec), out=np.zeros_like(tp), where=(prec+rec)>0)
    iou  = np.divide(tp, (gt+pr-tp), out=np.zeros_like(tp), where=(gt+pr-tp)>0)
    sup  = gt.astype(np.int64)
    return prec, rec, f1, iou, sup

def row_normalize(cm):
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    out = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums>0.0)
    return out

def plot_cm(cm, class_names, out_png, title, norm=False):
    fig = plt.figure(figsize=(10,8), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto", vmin=0.0, vmax=(1.0 if norm else None))
    ax.set_title(title)
    ax.set_xlabel("Pred"); ax.set_ylabel("GT")
    ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def top_confusions(cm, class_names, k=10):
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i,j] > 0:
                pairs.append((int(cm2[i,j]), i, j))
    pairs.sort(reverse=True)
    rows = []
    for n, (cnt, i, j) in enumerate(pairs[:k], 1):
        rows.append({
            "rank": n,
            "gt_class": class_names[i],
            "pred_class": class_names[j],
            "count": int(cnt)
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="carpeta que contiene run_single/")
    ap.add_argument("--data_dir", required=True, help="carpeta con X_test.npz / Y_test.npz y artifacts/")
    ap.add_argument("--which_model", default="best", choices=["best","final"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--bg_original_id", type=int, default=0, help="ID original del background (por defecto 0).")
    args = ap.parse_args()

    run_single = Path(args.run_dir)/"run_single"
    mdl_path = run_single/("checkpoints/best" if args.which_model=="best" else "final_model")
    model = keras.models.load_model(mdl_path, compile=False)

    data_dir = Path(args.data_dir)
    Xte = np.load(data_dir/"X_test.npz")["X"]
    Yte = np.load(data_dir/"Y_test.npz")["Y"]
    N, P, _ = Xte.shape

    # mapas de etiquetas para detectar índice remapeado del fondo
    id2idx, idx2id = load_label_map(data_dir/"artifacts")
    if args.bg_original_id in id2idx:
        bg_idx = int(id2idx[args.bg_original_id])
    else:
        # fallback: si no existe en el mapa, asumimos 0
        bg_idx = 0
    print(f"[INFO] idx remapeado de fondo (id={args.bg_original_id}) -> {bg_idx}")

    # lista de clases de dientes (excluye fondo)
    all_idx = sorted(idx2id.keys())
    tooth_idx = [i for i in all_idx if i != bg_idx]
    # nombres bonitos: usa id original (por ejemplo, 11,12,...) si aplica
    class_names = [str(idx2id[i]) for i in tooth_idx]

    # normalizar por muestra igual que en train
    Xn = np.empty_like(Xte, dtype=np.float32)
    for i in range(N): Xn[i] = normalize_cloud_np(Xte[i])

    probs = model.predict(Xn, batch_size=16, verbose=1)
    ypred = probs.argmax(axis=-1)

    # CM completa en el espacio remapeado 0..C-1
    C = probs.shape[-1]
    cm_full = confusion_matrix(Yte, ypred, C)

    # Filtrar a dientes: quitamos fila/col del fondo
    cm_teeth = cm_full[np.ix_(tooth_idx, tooth_idx)]

    # Métricas por diente
    prec, rec, f1, iou, sup = per_class_from_cm(cm_teeth)

    # Salida
    out_base = Path(args.out_dir) if args.out_dir else (run_single/"eval_teeth_only")
    out_base.mkdir(parents=True, exist_ok=True)

    # CSV por diente
    with open(out_base/"per_tooth_metrics.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tooth_id","precision","recall","f1","iou","support"])
        for name, p, r, f, u in zip(class_names, prec, rec, f1, iou, sup):
            w.writerow([name, float(p), float(r), float(f), float(i), int(u)])

    # Promedios macro sólo dientes
    overall = {
        "precision_macro_teeth": float(np.mean(prec)),
        "recall_macro_teeth": float(np.mean(rec)),
        "f1_macro_teeth": float(np.mean(f1)),
        "miou_macro_teeth": float(np.mean(iou)),
        "num_tooth_classes": int(len(class_names)),
        "N_samples": int(N),
        "P_points": int(P),
        "bg_index": int(bg_idx)
    }
    (out_base/"overall_teeth.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    # Guardar y plotear CM cruda (dientes) y normalizada por fila
    np.save(out_base/"confusion_matrix_teeth.npy", cm_teeth)
    plot_cm(cm_teeth, class_names, out_base/"cm_teeth_raw.png", "Confusion Matrix (teeth only)", norm=False)
    cm_row = row_normalize(cm_teeth)
    plot_cm(cm_row, class_names, out_base/"cm_teeth_row_norm.png", "Row-normalized CM (teeth only)", norm=True)

    # Top confusiones entre dientes (sin diagonales)
    top = top_confusions(cm_teeth, class_names, k=10)
    with open(out_base/"top10_confusions.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank","gt_class","pred_class","count"])
        w.writeheader(); w.writerows(top)

    print("✓ Guardado en:", out_base)

if __name__ == "__main__":
    main()
