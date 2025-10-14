#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models_aug.py
-------------------
Entrenador alternativo que aplica augmentations en tf.data
No toca el flujo de train_models.py original.
"""

import os, argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

import scripts.augment_utils as aug
from scripts.train_models import (
    setup_devices, make_datasets, PrecisionMacro, RecallMacro,
    F1Macro, SparseMeanIoU, DentalPointNet, get_optimizer,
    save_model_summary, save_history
)

def make_datasets_with_aug(data_path: str, batch_size: int, seed: int,
                           augment=False, aug_params=None):
    (Xtr,Ytr), (Xva,Yva), (Xte,Yte), info = make_datasets(data_path, batch_size, seed)

    if not augment:
        return Xtr, Xva, Xte, info

    def _map_fn(x, y):
        x = aug.augment(
            x,
            rotate_deg=aug_params.get("rotate_deg", 15),
            jitter_sigma=aug_params.get("jitter_sigma", 0.005),
            jitter_clip=aug_params.get("jitter_clip", 0.02),
            scale_min=aug_params.get("scale_min", 0.95),
            scale_max=aug_params.get("scale_max", 1.05),
            dropout_rate=aug_params.get("dropout_rate", 0.05),
        )
        return x, y

    Xtr = Xtr.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return Xtr, Xva, Xte, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--cuda", default="1")
    ap.add_argument("--metrics_macro", action="store_true")

    # Augment flags
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--aug_rotate_deg", type=float, default=15.0)
    ap.add_argument("--aug_jitter_sigma", type=float, default=0.005)
    ap.add_argument("--aug_jitter_clip", type=float, default=0.02)
    ap.add_argument("--aug_scale_min", type=float, default=0.95)
    ap.add_argument("--aug_scale_max", type=float, default=1.05)
    ap.add_argument("--aug_dropout", type=float, default=0.05)

    args = ap.parse_args()

    # salida
    out_base = Path(args.out_dir) / args.tag
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / "run_single"
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy = setup_devices(args.cuda, multi_gpu=False)

    aug_params = {
        "rotate_deg": args.aug_rotate_deg,
        "jitter_sigma": args.aug_jitter_sigma,
        "jitter_clip": args.aug_jitter_clip,
        "scale_min": args.aug_scale_min,
        "scale_max": args.aug_scale_max,
        "dropout_rate": args.aug_dropout,
    }

    ds_tr, ds_va, ds_te, dinfo = make_datasets_with_aug(
        args.data_path, args.batch_size, args.seed,
        augment=args.augment, aug_params=aug_params
    )
    num_classes = dinfo["num_classes"]

    def compile_model():
        model = DentalPointNet(
            num_classes=num_classes,
            base_channels=args.base_channels,
            dropout=args.dropout,
            activation=args.activation
        )
        metrics = ['accuracy']
        if args.metrics_macro:
            metrics += [
                PrecisionMacro(num_classes=num_classes, name="prec_macro"),
                RecallMacro(num_classes=num_classes, name="rec_macro"),
                F1Macro(num_classes=num_classes, name="f1_macro"),
                SparseMeanIoU(num_classes=num_classes, name="miou"),
            ]
        model.compile(optimizer=get_optimizer(args),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=metrics)
        return model

    if strategy:
        with strategy.scope():
            model = compile_model()
    else:
        model = compile_model()

    save_model_summary(model, run_dir)

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=str(run_dir/"checkpoints/best"),
                                        save_best_only=True, monitor='val_loss',
                                        save_format='tf'),
        keras.callbacks.CSVLogger(str(run_dir/"train_log.csv")),
    ]

    history = model.fit(ds_tr, validation_data=ds_va,
                        epochs=args.epochs, verbose=1,
                        callbacks=callbacks)
    save_history(history, run_dir)

    final_dir = run_dir/"final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_dir, save_format='tf')

if __name__ == "__main__":
    main()
