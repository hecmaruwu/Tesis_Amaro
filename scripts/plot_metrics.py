#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse
from pathlib import Path
import matplotlib.pyplot as plt

def pct(arr):
    return [x*100.0 for x in arr]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Carpeta del run que contiene history.json y test_metrics.json")
    ap.add_argument("--out", default=None, help="PNG de salida (si se omite, muestra en pantalla)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    hist_path = run / "history.json"
    if not hist_path.exists():
        raise SystemExit(f"No existe {hist_path}")

    with open(hist_path, "r") as f:
        H = json.load(f)

    keys = list(H.keys())
    # métricas esperadas
    loss_tr = H.get("loss", [])
    loss_va = H.get("val_loss", [])
    acc_tr  = H.get("accuracy", [])
    acc_va  = H.get("val_accuracy", [])

    # opcionales
    prec_tr = H.get("prec_macro", H.get("precision_macro", []))
    prec_va = H.get("val_prec_macro", H.get("val_precision_macro", []))
    rec_tr  = H.get("rec_macro", H.get("recall_macro", []))
    rec_va  = H.get("val_rec_macro", H.get("val_recall_macro", []))
    f1_tr   = H.get("f1_macro", [])
    f1_va   = H.get("val_f1_macro", [])
    miou_tr = H.get("miou", [])
    miou_va = H.get("val_miou", [])

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2,2,1)
    ax1.plot(loss_tr, label="train")
    ax1.plot(loss_va, label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

    ax2 = plt.subplot(2,2,2)
    ax2.plot(pct(acc_tr), label="train")
    ax2.plot(pct(acc_va), label="val")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("%"); ax2.set_ylim(0, 100); ax2.legend()

    ax3 = plt.subplot(2,2,3)
    if f1_tr:
        ax3.plot(pct(prec_tr), label="P-train")
        ax3.plot(pct(prec_va), label="P-val")
        ax3.plot(pct(rec_tr),  label="R-train")
        ax3.plot(pct(rec_va),  label="R-val")
        ax3.plot(pct(f1_tr),   label="F1-train", linestyle="--")
        ax3.plot(pct(f1_va),   label="F1-val",   linestyle="--")
        ax3.set_title("PRF1 macro (%)")
        ax3.set_ylim(0, 100)
    else:
        ax3.text(0.5, 0.5, "Sin PRF1; corre con --metrics_macro", ha="center")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("%"); ax3.legend()

    ax4 = plt.subplot(2,2,4)
    if miou_tr:
        ax4.plot(pct(miou_tr), label="train")
        ax4.plot(pct(miou_va), label="val")
        ax4.set_title("mIoU (%)"); ax4.set_ylim(0, 100)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "Sin mIoU; corre con --metrics_macro", ha="center")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("%")

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=160)
        print(f"[OK] Guardado gráfico en {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
