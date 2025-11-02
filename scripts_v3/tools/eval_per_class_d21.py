#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_per_class_d21.py
Evalúa por-clase (incluyendo d21) un checkpoint de un run ya entrenado.
Reutiliza las utilidades de train_models_paperlike_loss_balanceada.py
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# IMPORTA utilidades del trainer (asegúrate de tener PYTHONPATH apuntando a scripts_v3)
from train_models_paperlike_loss_balanceada import (
    build_model, make_loaders, sanitize_tensor, normalize_cloud,
    confusion_matrix, d21_metrics
)

@torch.no_grad()
def eval_split_per_class(model, loader, device, num_classes, d21_id):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, device=device)
    d21_aggr = {"d21_acc":0.0,"d21_f1":0.0,"d21_iou":0.0}
    batches = 0
    ce = nn.CrossEntropyLoss(ignore_index=0).to(device)

    loss_sum = 0.0
    for X, Y in loader:
        X = sanitize_tensor(normalize_cloud(X)).to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True).clamp(0, num_classes-1)
        logits = sanitize_tensor(model(X))
        loss_sum += float(ce(logits.transpose(2,1), Y).detach().cpu())
        cm += confusion_matrix(logits, Y, num_classes)
        dm = d21_metrics(logits, Y, d21_id)
        for k in d21_aggr: d21_aggr[k] += dm[k]
        batches += 1

    # macro
    cmf = cm.float()
    tp = torch.diag(cmf)
    gt = cmf.sum(1).clamp_min(1e-8)
    pd = cmf.sum(0).clamp_min(1e-8)

    # por-clase
    prec_c = (tp / pd).cpu().numpy()
    rec_c  = (tp / gt).cpu().numpy()
    f1_c   = (2*tp / (gt+pd)).cpu().numpy()
    iou_c  = (tp / (gt+pd-tp).clamp_min(1e-8)).cpu().numpy()
    sup_c  = gt.cpu().numpy()  # soporte (puntos GT por clase)

    # promedios macro
    acc = (tp.sum() / cmf.sum().clamp_min(1e-8)).item()
    macro_f1 = float(np.nanmean(f1_c))
    macro_iou = float(np.nanmean(iou_c))
    loss = loss_sum / max(1,batches)
    for k in d21_aggr: d21_aggr[k] /= max(1,batches)

    return {
        "loss": loss, "acc": acc, "macro_f1": macro_f1, "macro_iou": macro_iou,
        "cm": cm.cpu().numpy(), "prec_c": prec_c, "rec_c": rec_c,
        "f1_c": f1_c, "iou_c": iou_c, "sup_c": sup_c, "d21": d21_aggr
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_dir",  required=True, help="Carpeta del run (con checkpoints/best.pt o final_model.pt)")
    ap.add_argument("--model",    required=True, choices=["pointnet","pointnetpp","dilatedtoothsegnet","transformer3d","toothformer"])
    ap.add_argument("--split",    default="val", choices=["val","test"])
    ap.add_argument("--d21_id",   type=int, default=1)
    ap.add_argument("--ckpt",     default="best.pt", choices=["best.pt","final_model.pt"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    run_dir  = Path(args.run_dir)
    ckpt_p   = run_dir / "checkpoints" / args.ckpt
    if not ckpt_p.exists():
        raise FileNotFoundError(f"No existe checkpoint: {ckpt_p}")

    # detectar clases
    Ytr = np.load(data_dir / "Y_train.npz")["Y"]
    num_classes = int(np.max(Ytr)) + 1
    print(f"[INFO] num_classes={num_classes} | d21_id={args.d21_id}")

    # loaders
    loaders = make_loaders(data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    loader = loaders[args.split]

    # modelo
    model = build_model(args.model, num_classes=num_classes).to(device)
    state = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(state["model"])
    print(f"[LOAD] {ckpt_p.name} -> epoch={state.get('epoch','?')}")

    # eval
    out = eval_split_per_class(model, loader, device, num_classes, args.d21_id)

    # imprimir d21 y macro
    d = out["d21"]
    print(f"\n[{args.split.upper()}] loss={out['loss']:.4f}  acc={out['acc']:.3f}  "
          f"macroF1={out['macro_f1']:.3f}  macroIoU={out['macro_iou']:.3f}")
    print(f"   d21 -> acc={d['d21_acc']:.3f}  f1={d['d21_f1']:.3f}  iou={d['d21_iou']:.3f}")

    # guardar CSV por-clase
    import csv
    csvp = run_dir / f"per_class_metrics_{args.split}.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id","support","precision","recall","f1","iou"])
        for cid in range(num_classes):
            w.writerow([
                cid, int(out["sup_c"][cid]),
                f"{out['prec_c'][cid]:.6f}",
                f"{out['rec_c'][cid]:.6f}",
                f"{out['f1_c'][cid]:.6f}",
                f"{out['iou_c'][cid]:.6f}",
            ])
    print(f"[WRITE] {csvp}")

if __name__ == "__main__":
    main()
