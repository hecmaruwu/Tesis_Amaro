#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse
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

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.reshape(-1), y_pred.reshape(-1)):
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
    return prec, rec, f1, iou

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="carpeta que contiene run_single/")
    ap.add_argument("--data_dir", required=True, help="carpeta con X_test.npz / Y_test.npz")
    ap.add_argument("--which_model", default="best", choices=["best","final"])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    run_single = Path(args.run_dir)/"run_single"
    mdl_path = run_single/("checkpoints/best" if args.which_model=="best" else "final_model")
    model = keras.models.load_model(mdl_path, compile=False)

    Xte = np.load(Path(args.data_dir)/"X_test.npz")["X"]
    Yte = np.load(Path(args.data_dir)/"Y_test.npz")["Y"]
    N, P, _ = Xte.shape
    ncls = int(max(Yte.max(), 0) + 1)

    Xn = np.empty_like(Xte, dtype=np.float32)
    for i in range(N):
        Xn[i] = normalize_cloud_np(Xte[i])

    probs = model.predict(Xn, batch_size=16, verbose=1)
    ypred = probs.argmax(axis=-1)

    cm = confusion_matrix(Yte, ypred, ncls)
    prec, rec, f1, iou = per_class_from_cm(cm)

    out_base = Path(args.out_dir) if args.out_dir else (run_single/"eval_test")
    out_base.mkdir(parents=True, exist_ok=True)

    # CSV por-clase
    import csv
    with open(out_base/"per_class_metrics.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id","precision","recall","f1","iou","support"])
        sup = cm.sum(axis=1)
        for c in range(ncls):
            w.writerow([c, float(prec[c]), float(rec[c]), float(f1[c]), float(iou[c]), int(sup[c])])

    # Promedios
    overall = {
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
        "miou_macro": float(np.mean(iou)),
        "num_classes": int(ncls),
        "N_samples": int(N),
        "P_points": int(P),
    }
    (out_base/"overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    # Guardar y plotear CM
    np.save(out_base/"confusion_matrix.npy", cm)
    fig = plt.figure(figsize=(8,6), dpi=160)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    ax.set_title("Confusion Matrix (test)"); ax.set_xlabel("Pred"); ax.set_ylabel("GT")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_base/"confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    print("✓ Guardado en:", out_base)

if __name__ == "__main__":
    main()
