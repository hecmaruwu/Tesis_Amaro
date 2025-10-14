#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenador DGCNN (segmentation) compatible con los splits .npz.

Uso:
  python train_dgcnn.py \
    --data_path data/3dteethseg/splits/8192_seed42_pairs \
    --out_dir runs_grid/dgcnn_8192_pairs_s42 \
    --tag run_single \
    --epochs 150 --batch_size 8 --lr 1e-3 --metrics_macro --seed 42
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def set_global_seed(seed:int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(seed); tf.random.set_seed(seed)

def load_npz_splits(data_path: Path):
    p = Path(data_path)
    Xtr = np.load(p/"X_train.npz")["X"]
    Ytr = np.load(p/"Y_train.npz")["Y"]
    Xva = np.load(p/"X_val.npz")["X"]
    Yva = np.load(p/"Y_val.npz")["Y"]
    Xte = np.load(p/"X_test.npz")["X"]
    Yte = np.load(p/"Y_test.npz")["Y"]
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    print(f"[DATA] Xtr{Xtr.shape} Xva{Xva.shape} Xte{Xte.shape}  classes={num_classes}")
    return (Xtr,Ytr),(Xva,Yva),(Xte,Yte),{"num_classes":num_classes}

@tf.function
def _normalize_cloud(x):
    x = tf.cast(x, tf.float32)
    mean = tf.reduce_mean(x, axis=-2, keepdims=True)
    x = x - mean
    r = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
    r = tf.reduce_max(r, axis=-2, keepdims=True)
    return x / (r + 1e-6)

def _augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, scale_low=0.9, scale_high=1.1):
    theta = tf.random.uniform([], -rot_deg, rot_deg, dtype=tf.float32) * (np.pi/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c,-s,0.],[s,c,0.],[0.,0.,1.]], axis=0)
    x = tf.matmul(x, R)
    s = tf.random.uniform([], scale_low, scale_high, dtype=tf.float32)
    x = x * s
    noise = tf.random.normal(tf.shape(x), 0.0, jitter_std, dtype=tf.float32)
    x = x + noise
    return x, y

def make_datasets(data_path, batch, seed, do_augment, rot_deg, jitter_std, scale_low, scale_high):
    (Xtr,Ytr),(Xva,Yva),(Xte,Yte),info = load_npz_splits(data_path)
    def _ds(X,Y,shuffle=False,augment=False):
        ds = tf.data.Dataset.from_tensor_slices((X,Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(4096,X.shape[0]), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda x,y: (_normalize_cloud(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x,y: _augment_cloud(x,y,rot_deg,jitter_std,scale_low,scale_high),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return (_ds(Xtr,Ytr,True,do_augment),
            _ds(Xva,Yva,False,False),
            _ds(Xte,Yte,False,False),
            info)

# ======= Métricas =======
class _CMM(keras.metrics.Metric):
    def __init__(self, num_classes, name="cm", **kw):
        super().__init__(name=name, **kw)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(shape=(self.num_classes,self.num_classes),
                                  initializer="zeros", dtype=tf.float32, name="cm")
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        self.cm.assign_add(cm)
    def reset_state(self): self.cm.assign(tf.zeros_like(self.cm))
    def reset_states(self): self.reset_state()

class PrecisionMacro(_CMM):
    def result(self):
        cm = self.cm; tp = tf.linalg.tensor_diag_part(cm)
        pred = tf.reduce_sum(cm, axis=0)
        return tf.reduce_mean(tf.math.divide_no_nan(tp, pred))

class RecallMacro(_CMM):
    def result(self):
        cm = self.cm; tp = tf.linalg.tensor_diag_part(cm)
        gt = tf.reduce_sum(cm, axis=1)
        return tf.reduce_mean(tf.math.divide_no_nan(tp, gt))

class F1Macro(_CMM):
    def result(self):
        cm = self.cm
        tp = tf.linalg.tensor_diag_part(cm)
        gt = tf.reduce_sum(cm, axis=1)
        pred = tf.reduce_sum(cm, axis=0)
        prec = tf.math.divide_no_nan(tp, pred)
        rec  = tf.math.divide_no_nan(tp, gt)
        return tf.reduce_mean(tf.math.divide_no_nan(2.0*prec*rec, prec+rec))

class SparseMeanIoU(_CMM):
    def result(self):
        cm = self.cm
        tp = tf.linalg.tensor_diag_part(cm)
        gt = tf.reduce_sum(cm, axis=1)
        pred = tf.reduce_sum(cm, axis=0)
        union = gt + pred - tp
        return tf.reduce_mean(tf.math.divide_no_nan(tp, union))

# ======= DGCNN =======
def knn(x, k):
    # x: (B,N,C)
    batch_size = tf.shape(x)[0]
    dist = pairwise_distance(x)              # (B,N,N)
    _, idx = tf.math.top_k(-dist, k=k)       # (B,N,k) menores distancias
    return idx

def pairwise_distance(x):
    xx = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    yy = tf.transpose(xx, perm=[0,2,1])
    xy = tf.matmul(x, x, transpose_b=True)
    dist = xx + yy - 2.0*xy
    return tf.maximum(dist, 0.0)

def get_graph_feature(x, k=20):
    # x: (B,N,C). Devuelve (B,N,k,2C) con (x_j - x_i, x_i)
    idx = knn(x, k)                          # (B,N,k)
    B = tf.shape(x)[0]; N = tf.shape(x)[1]; C = tf.shape(x)[2]
    idx_exp = tf.reshape(idx, (B, N*k))
    feat_neighbors = tf.gather(x, idx_exp, batch_dims=1)       # (B, N*k, C)
    feat_neighbors = tf.reshape(feat_neighbors, (B, N, k, C))  # (B,N,k,C)
    x_i = tf.expand_dims(x, axis=2)                             # (B,N,1,C)
    x_i = tf.tile(x_i, [1,1,k,1])                               # (B,N,k,C)
    edge = tf.concat([feat_neighbors - x_i, x_i], axis=-1)      # (B,N,k,2C)
    return edge

def EdgeConvBlock(x, k, out_channels):
    # x: (B,N,C)
    edge = get_graph_feature(x, k=k)           # (B,N,k,2C)
    h = layers.Conv2D(out_channels, (1,1), activation='relu')(edge)
    h = layers.Conv2D(out_channels, (1,1), activation='relu')(h)
    h = tf.reduce_max(h, axis=2)               # (B,N,out)
    return h

def build_dgcnn_seg(num_points, num_classes, k=20, base=64, dropout=0.5):
    inp = layers.Input(shape=(num_points,3), dtype=tf.float32)
    x = inp

    x1 = EdgeConvBlock(x, k, base)         # (B,N,base)
    x2 = EdgeConvBlock(x1, k, base)        # (B,N,base)
    x3 = EdgeConvBlock(x2, k, base*2)      # (B,N,2b)
    x4 = EdgeConvBlock(x3, k, base*2)      # (B,N,2b)

    x_cat = tf.concat([x1,x2,x3,x4], axis=-1)   # (B,N,base*6)
    xg = layers.GlobalMaxPooling1D()(x_cat)
    xg = tf.expand_dims(xg, axis=1)
    xg = tf.tile(xg, [1, num_points, 1])

    h = tf.concat([x_cat, xg], axis=-1)        # (B,N, base*6 + glob)
    h = layers.Conv1D(base*4, 1, activation='relu')(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Conv1D(base*2, 1, activation='relu')(h)
    out = layers.Conv1D(num_classes, 1, activation='softmax')(h)

    return keras.Model(inputs=inp, outputs=out, name="DGCNN")

# ======= Train loop =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", default="adam", choices=["adam","sgd"])
    ap.add_argument("--metrics_macro", action="store_true")

    ap.add_argument("--augment", dest="augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--rot_deg", type=float, default=10.0)
    ap.add_argument("--jitter_std", type=float, default=0.005)
    ap.add_argument("--scale_low", type=float, default=0.90)
    ap.add_argument("--scale_high", type=float, default=1.10)

    ap.add_argument("--lr_patience", type=int, default=12)
    ap.add_argument("--es_patience", type=int, default=20)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    args = ap.parse_args()

    set_global_seed(args.seed)

    out = Path(args.out_dir)/args.tag
    (out/"checkpoints").mkdir(parents=True, exist_ok=True)

    ds_tr, ds_va, ds_te, info = make_datasets(
        args.data_path, args.batch_size, args.seed,
        args.augment, args.rot_deg, args.jitter_std, args.scale_low, args.scale_high
    )
    num_classes = info["num_classes"]

    spec = ds_tr.element_spec[0].shape
    P = int(spec[1]) if spec.rank is not None and spec[1] is not None else None
    if P is None:
        xb,_ = next(iter(ds_tr))
        P = int(xb.shape[1])

    model = build_dgcnn_seg(P, num_classes, k=args.k, base=args.base_channels, dropout=args.dropout)

    if args.optimizer=="adam":
        opt = keras.optimizers.Adam(learning_rate=args.lr)
    else:
        opt = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)

    metrics = ["accuracy"]
    if args.metrics_macro:
        metrics += [PrecisionMacro(num_classes, "prec_macro"),
                    RecallMacro(num_classes, "rec_macro"),
                    F1Macro(num_classes, "f1_macro"),
                    SparseMeanIoU(num_classes, "miou")]

    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=metrics)

    _ = model(tf.zeros([1,P,3], tf.float32), training=False)
    txt=[]; model.summary(print_fn=lambda s: txt.append(s))
    (out/"model_summary.txt").write_text("\n".join(txt), encoding="utf-8")

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=args.lr_patience, min_lr=args.min_lr, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.es_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=str(out/"checkpoints/best"),
                                        save_best_only=True, monitor="val_loss", save_format="tf", verbose=1),
        keras.callbacks.CSVLogger(str(out/"train_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(out/"tb")),
    ]
    (out/"config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=callbacks, verbose=1)
    (out/"history.json").write_text(json.dumps({k:[float(x) for x in v] for k,v in hist.history.items()}, indent=2), encoding="utf-8")

    print("[EVAL] Test:")
    test_metrics = model.evaluate(ds_te, verbose=1, return_dict=True)
    (out/"test_metrics.json").write_text(json.dumps({k: float(v) for k,v in test_metrics.items()}, indent=2), encoding="utf-8")

    final_dir = out/"final_model"
    final_dir.mkdir(exist_ok=True)
    model.save(final_dir, save_format="tf")
    print(f"[FIN] guardado en: {final_dir}")

if __name__ == "__main__":
    main()
