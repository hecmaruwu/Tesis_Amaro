#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento PointNet++ (mejorado) + métricas macro y por-clase (val/test)
Dataset .npz:
  - X_train.npz["X"]: (N, P, 3)   Y_train.npz["Y"]: (N, P)
  - X_val.npz  ["X"]: (N, P, 3)   Y_val.npz  ["Y"]: (N, P)
  - X_test.npz ["X"]: (N, P, 3)   Y_test.npz ["Y"]: (N, P)
"""

import os, json, argparse, csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============== Utils: reproducibilidad y GPU =================
def set_global_seed(seed: int):
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    tf.random.set_global_generator(tf.random.Generator.from_seed(int(seed)))
    np.random.seed(seed)

def allow_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

# ============== Carga de datos ===============================
def load_npz_split(data_path: Path):
    Xtr = np.load(data_path/"X_train.npz")["X"]; Ytr = np.load(data_path/"Y_train.npz")["Y"]
    Xva = np.load(data_path/"X_val.npz")["X"];   Yva = np.load(data_path/"Y_val.npz")["Y"]
    Xte = np.load(data_path/"X_test.npz")["X"];  Yte = np.load(data_path/"Y_test.npz")["Y"]
    return Xtr, Ytr, Xva, Yva, Xte, Yte

def write_label_map(artifacts_dir: Path, classes: int):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    id2idx = {str(i): i for i in range(classes)}
    idx2id = {i: str(i) for i in range(classes)}
    (artifacts_dir/"label_map.json").write_text(
        json.dumps({"id2idx": id2idx, "idx2id": idx2id}, indent=2),
        encoding="utf-8"
    )

# ============== Augment determinista =========================
def _augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05, rng=None):
    if rng is None: rng = tf.random.get_global_generator()
    theta = rng.uniform([], minval=-rot_deg, maxval=rot_deg, dtype=tf.float32) * (tf.constant(np.pi)/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c, -s, 0.],[s, c, 0.],[0.,0.,1.]], axis=0)
    x = tf.matmul(x, R)
    scale = rng.uniform([], minval=scale_low, maxval=scale_high, dtype=tf.float32)
    x = x * scale
    noise = rng.normal(tf.shape(x), stddev=jitter_std, dtype=tf.float32)
    x = x + noise
    return x, y

def make_datasets(Xtr, Ytr, Xva, Yva, Xte, Yte, batch_size=8, augment=False, seed=42,
                  rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05):
    gen = tf.random.Generator.from_seed(seed)
    AUTOTUNE = tf.data.AUTOTUNE
    def _ds(x, y, shuffle=False, aug=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda a,b: (tf.cast(a, tf.float32), tf.cast(b, tf.int32)), num_parallel_calls=AUTOTUNE)
        if aug:
            ds = ds.map(lambda a,b: _augment_cloud(a,b,rot_deg,jitter_std,scale_low,scale_high,rng=gen),
                        num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return ds
    return (_ds(Xtr,Ytr,shuffle=True, aug=augment),
            _ds(Xva,Yva,shuffle=False, aug=False),
            _ds(Xte,Yte,shuffle=False, aug=False))

# ============== Núcleo PointNet++ mejorado ===================
def index_points(points, idx):
    if len(idx.shape) == 2:
        B = tf.shape(points)[0]; S = tf.shape(idx)[1]
        batch_indices = tf.tile(tf.reshape(tf.range(B), [B,1]), [1,S])
        gather_idx = tf.stack([batch_indices, idx], axis=-1)
        return tf.gather_nd(points, gather_idx)
    else:
        B = tf.shape(points)[0]; S = tf.shape(idx)[1]; K = tf.shape(idx)[2]
        batch_indices = tf.tile(tf.reshape(tf.range(B), [B,1,1]), [1,S,K])
        gather_idx = tf.stack([batch_indices, idx], axis=-1)
        return tf.gather_nd(points, gather_idx)

def farthest_point_sampling(xyz, npoint, rng=None):
    """
    FPS estable con TensorArray (forma fija). xyz:(B,N,3) -> (B,npoint) índices.
    """
    if rng is None: rng = tf.random.get_global_generator()
    B = tf.shape(xyz)[0]; N = tf.shape(xyz)[1]

    # elige primer punto aleatorio por batch
    first = rng.uniform([B], minval=0, maxval=N, dtype=tf.int32)  # (B,)
    centroids_ta = tf.TensorArray(dtype=tf.int32, size=npoint, clear_after_read=False)
    centroids_ta = centroids_ta.write(0, first)

    # distancia mínima a elegidos
    first_xyz = tf.gather(xyz, first, batch_dims=1)              # (B,3)
    dist_min = tf.reduce_sum((xyz - tf.expand_dims(first_xyz,1))**2, axis=-1)  # (B,N)

    i0 = tf.constant(1)
    def cond(i, *_):
        return i < npoint

    def body(i, centroids_ta, dist_min):
        prev = centroids_ta.read(i-1)                             # (B,)
        prev_xyz = tf.gather(xyz, prev, batch_dims=1)            # (B,3)
        d = tf.reduce_sum((xyz - tf.expand_dims(prev_xyz,1))**2, axis=-1)      # (B,N)
        dist_min = tf.minimum(dist_min, d)
        farthest = tf.argmax(dist_min, axis=-1, output_type=tf.int32)          # (B,)
        centroids_ta = centroids_ta.write(i, farthest)
        return i+1, centroids_ta, dist_min

    _, centroids_ta, _ = tf.while_loop(cond, body, [i0, centroids_ta, dist_min],
                                       parallel_iterations=1)
    idx = tf.transpose(centroids_ta.stack(), perm=[1,0])  # (B, npoint)
    return idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    dist = tf.reduce_sum((tf.expand_dims(new_xyz,2) - tf.expand_dims(xyz,1))**2, axis=-1)
    mask = dist <= radius**2
    dist_masked = tf.where(mask, dist, tf.fill(tf.shape(dist), tf.constant(np.inf, dtype=tf.float32)))
    idx = tf.argsort(dist_masked, axis=-1)[:, :, :nsample]
    return idx

class SetAbstraction(layers.Layer):
    def __init__(self, npoint, radius, nsample, mlp_out, name=None):
        super().__init__(name=name)
        self.npoint = int(npoint); self.radius = float(radius); self.nsample = int(nsample)
        self.mlp1 = keras.Sequential([layers.Dense(mlp_out, activation='relu'),
                                      layers.Dense(mlp_out, activation='relu')])
        self.mlp_w = keras.Sequential([layers.Dense(mlp_out, activation='relu'),
                                       layers.Dense(mlp_out, activation='sigmoid')])
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"npoint": self.npoint, "radius": self.radius,
                    "nsample": self.nsample, "mlp_out": self.mlp1.layers[-1].units})
        return cfg
    def call(self, xyz, features, training=False):
        fps_idx = farthest_point_sampling(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - tf.expand_dims(new_xyz, axis=2)
        if features is None:
            grouped_feats = grouped_xyz_norm
        else:
            grouped_feats = index_points(features, group_idx)
        f_in = tf.concat([grouped_xyz_norm, grouped_feats], axis=-1)
        f_prime = self.mlp1(f_in)
        f_mean = tf.reduce_mean(f_prime, axis=2, keepdims=True)
        w_in = tf.concat([grouped_xyz_norm, f_prime - f_mean], axis=-1)
        alpha = self.mlp_w(w_in)
        f_out = tf.reduce_sum(alpha * f_prime, axis=2)
        return new_xyz, f_out

class FeaturePropagation(layers.Layer):
    def __init__(self, mlp_out, name=None):
        super().__init__(name=name)
        self.mlp = keras.Sequential([layers.Dense(mlp_out, activation='relu'),
                                     layers.Dense(mlp_out, activation='relu')])
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mlp_out": self.mlp.layers[-1].units})
        return cfg
    def call(self, xyz1, feat1, xyz2, feat2):
        dists = tf.reduce_sum((tf.expand_dims(xyz1,2) - tf.expand_dims(xyz2,1))**2, axis=-1)
        idx = tf.argsort(dists, axis=-1)[:, :, :3]
        d3 = tf.gather(dists, idx, batch_dims=2); d3 = tf.maximum(d3, 1e-10)
        w = 1.0 / d3; w = w / tf.reduce_sum(w, axis=-1, keepdims=True)
        neighbor_feats = index_points(feat2, idx)
        interpolated = tf.reduce_sum(tf.expand_dims(w,-1) * neighbor_feats, axis=2)
        new_feat = interpolated if feat1 is None else tf.concat([interpolated, feat1], axis=-1)
        return self.mlp(new_feat)

def build_improved_pointnetpp(num_points, num_classes, base=64, dropout=0.5):
    xyz_in = layers.Input(shape=(num_points, 3), name="xyz")
    # SPFE: 9 canales [xyz | normales_dummy(0) | xyz-mean]
    xyz_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(xyz_in)
    xyz_centered = layers.Subtract()([xyz_in, xyz_mean])
    zeros = layers.Lambda(lambda t: tf.zeros_like(t))(xyz_in)
    spfe_in = layers.Concatenate(axis=-1)([xyz_in, zeros, xyz_centered])
    spfe = layers.Dense(64, activation='relu')(spfe_in)
    spfe = layers.Dense(64, activation='relu')(spfe)

    l1_xyz, l1_feat = SetAbstraction(npoint=num_points//4,  radius=0.05, nsample=32, mlp_out=base)(xyz_in, spfe)
    l2_xyz, l2_feat = SetAbstraction(npoint=num_points//8,  radius=0.10, nsample=32, mlp_out=base*2)(l1_xyz, l1_feat)
    l3_xyz, l3_feat = SetAbstraction(npoint=num_points//16, radius=0.20, nsample=32, mlp_out=base*4)(l2_xyz, l2_feat)

    fp2 = FeaturePropagation(base*2)(l2_xyz, l2_feat, l3_xyz, l3_feat)
    fp1 = FeaturePropagation(base)(l1_xyz, l1_feat, l2_xyz, fp2)
    fp0 = FeaturePropagation(base)(xyz_in, spfe, l1_xyz, fp1)

    x = layers.Dense(base*2, activation='relu')(fp0)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(num_classes, activation=None, name="logits")(x)
    out = layers.Activation("softmax", name="softmax")(logits)
    return keras.Model(inputs=xyz_in, outputs=out, name="PointNetPP_Improved")

# ============== Métricas macro correctas ====================
class _ConfMatMetric(keras.metrics.Metric):
    def __init__(self, num_classes: int, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(name="cm", shape=(self.num_classes, self.num_classes),
                                  initializer="zeros", dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_hat  = tf.reshape(tf.argmax(y_pred, axis=-1, output_type=tf.int32), [-1])
        cm_batch = tf.math.confusion_matrix(y_true, y_hat, num_classes=self.num_classes, dtype=tf.float32)
        self.cm.assign_add(cm_batch)
    def reset_states(self):
        tf.keras.backend.set_value(self.cm, tf.zeros_like(self.cm))

class PrecisionMacro(_ConfMatMetric):
    def result(self):
        diag = tf.linalg.diag_part(self.cm)
        pred_pos = tf.reduce_sum(self.cm, axis=0)
        per_class = tf.math.divide_no_nan(diag, pred_pos)
        return tf.reduce_mean(per_class)

class RecallMacro(_ConfMatMetric):
    def result(self):
        diag = tf.linalg.diag_part(self.cm)
        real_pos = tf.reduce_sum(self.cm, axis=1)
        per_class = tf.math.divide_no_nan(diag, real_pos)
        return tf.reduce_mean(per_class)

class F1Macro(_ConfMatMetric):
    def result(self):
        diag = tf.linalg.diag_part(self.cm)
        pred_pos = tf.reduce_sum(self.cm, axis=0)
        real_pos = tf.reduce_sum(self.cm, axis=1)
        prec = tf.math.divide_no_nan(diag, pred_pos)
        rec  = tf.math.divide_no_nan(diag, real_pos)
        f1_c = tf.math.divide_no_nan(2.0*prec*rec, (prec+rec))
        return tf.reduce_mean(f1_c)

class MeanIoU(_ConfMatMetric):
    def result(self):
        diag = tf.linalg.diag_part(self.cm)
        rows = tf.reduce_sum(self.cm, axis=1)
        cols = tf.reduce_sum(self.cm, axis=0)
        denom = rows + cols - diag
        iou_c = tf.math.divide_no_nan(diag, denom)
        return tf.reduce_mean(iou_c)

def make_metrics(num_classes: int, macro: bool):
    m = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if macro:
        m += [PrecisionMacro(num_classes, "prec_macro"),
              RecallMacro(num_classes, "rec_macro"),
              F1Macro(num_classes, "f1_macro"),
              MeanIoU(num_classes, "miou")]
    return m

# ============== Helpers métricas por clase ==================
def _cm_stats(cm: np.ndarray):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    pred_pos = cm.sum(0)
    real_pos = cm.sum(1)
    denom_iou = pred_pos + real_pos - tp
    prec = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos>0)
    rec  = np.divide(tp, real_pos, out=np.zeros_like(tp), where=real_pos>0)
    f1   = np.divide(2*prec*rec, (prec+rec), out=np.zeros_like(tp), where=(prec+rec)>0)
    iou  = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou>0)
    return prec, rec, f1, iou

class PerClassMetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, num_classes: int, out_dir: Path, label_map=None):
        super().__init__()
        self.val_ds = val_ds
        self.C = int(num_classes)
        self.out_dir = Path(out_dir); (self.out_dir/"metrics").mkdir(parents=True, exist_ok=True)
        self.label_map = label_map

    def on_epoch_end(self, epoch, logs=None):
        cm = np.zeros((self.C, self.C), dtype=np.int64)
        for x, y in self.val_ds:
            pred = self.model.predict(x, verbose=0)
            y_true = tf.reshape(y, [-1]).numpy()
            y_hat  = tf.reshape(tf.argmax(pred, axis=-1), [-1]).numpy()
            cm += tf.math.confusion_matrix(y_true, y_hat, num_classes=self.C, dtype=tf.int64).numpy()

        prec, rec, f1, iou = _cm_stats(cm)
        macro = {
            "prec_macro": float(np.mean(prec)),
            "rec_macro":  float(np.mean(rec)),
            "f1_macro":   float(np.mean(f1)),
            "miou":       float(np.mean(iou)),
            "accuracy":   float(np.trace(cm)/max(cm.sum(),1))
        }
        csv_path = self.out_dir/"metrics"/f"per_class_epoch_{epoch+1:03d}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class_idx","class_name","precision","recall","f1","iou","support"])
            for c in range(self.C):
                name = self.label_map.get(c, str(c)) if self.label_map else str(c)
                support = int(cm[c].sum())
                w.writerow([c, name, float(prec[c]), float(rec[c]), float(f1[c]), float(iou[c]), support])
        latest_path = self.out_dir/"metrics"/"per_class_latest.csv"
        Path(latest_path).write_text(Path(csv_path).read_text(encoding="utf-8"), encoding="utf-8")
        with open(self.out_dir/"metrics"/"macro_val_latest.json","w",encoding="utf-8") as f:
            json.dump(macro, f, indent=2)
        print(f"[VAL/PerClass] epoch={epoch+1} macro={macro}")

# ============== Entrenamiento ============================================
def train(args):
    set_global_seed(args.seed)
    allow_gpu_memory_growth()

    data_dir = Path(args.data_path)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"artifacts").mkdir(exist_ok=True, parents=True)

    Xtr, Ytr, Xva, Yva, Xte, Yte = load_npz_split(data_dir)
    P = Xtr.shape[1]
    classes = int(max(Ytr.max(), Yva.max(), Yte.max())+1)

    print(f"[DATA] X_train:{Xtr.shape} X_val:{Xva.shape} X_test:{Xte.shape}")
    print(f"[INFO] GPUs visibles: {tf.config.list_physical_devices('GPU')}")
    print(f"[HPARAMS] {{'epochs': {args.epochs}, 'batch_size': {args.batch_size}, 'lr': {args.lr}, 'dropout': {args.dropout}, 'base_channels': {args.base_channels}}}")

    ds_tr, ds_va, ds_te = make_datasets(
        Xtr,Ytr,Xva,Yva,Xte,Yte,
        batch_size=args.batch_size,
        augment=args.augment,
        seed=args.seed,
        rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05
    )

    model = build_improved_pointnetpp(P, classes, base=args.base_channels, dropout=args.dropout)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    metrics = make_metrics(classes, macro=args.metrics_macro)
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=metrics)

    label_map = {i: str(i) for i in range(classes)}
    write_label_map(out_dir/"artifacts", classes)

    ckpt_dir = out_dir/"ckpts"; ckpt_dir.mkdir(exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(str(ckpt_dir/"best.keras"),
                                           monitor="val_accuracy", save_best_only=True)
    lr_sched = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.5, patience=args.lr_patience, min_lr=1e-6, verbose=1)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.es_patience, restore_best_weights=True)
    perclass_cb = PerClassMetricsCallback(ds_va, classes, out_dir, label_map=label_map)

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs,
                     verbose=1, callbacks=[ckpt, lr_sched, es, perclass_cb])

    eval_te = model.evaluate(ds_te, verbose=1)
    eval_names = [m.name if hasattr(m, "name") else f"m{i}" for i,m in enumerate(model.metrics)]
    with open(out_dir/"eval_test.json","w",encoding="utf-8") as f:
        json.dump({k: float(v) for k,v in zip(eval_names, eval_te)}, f, indent=2)

    # Per-clase TEST
    C = classes
    cm = np.zeros((C,C), dtype=np.int64)
    for x, y in ds_te:
        pred = model.predict(x, verbose=0)
        y_true = tf.reshape(y, [-1]).numpy()
        y_hat  = tf.reshape(tf.argmax(pred, axis=-1), [-1]).numpy()
        cm += tf.math.confusion_matrix(y_true, y_hat, num_classes=C, dtype=tf.int64).numpy()
    def _cm_stats(cm_):
        cm_ = cm_.astype(np.float64)
        tp = np.diag(cm_)
        pred_pos = cm_.sum(0)
        real_pos = cm_.sum(1)
        denom_iou = pred_pos + real_pos - tp
        prec = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos>0)
        rec  = np.divide(tp, real_pos, out=np.zeros_like(tp), where=real_pos>0)
        f1   = np.divide(2*prec*rec, (prec+rec), out=np.zeros_like(tp), where=(prec+rec)>0)
        iou  = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou>0)
        return prec, rec, f1, iou
    prec, rec, f1, iou = _cm_stats(cm)
    per_class_test = []
    for c in range(C):
        per_class_test.append({
            "class_idx": int(c),
            "class_name": label_map.get(c, str(c)),
            "precision": float(prec[c]),
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "iou": float(iou[c]),
            "support": int(cm[c].sum())
        })
    (out_dir/"metrics").mkdir(exist_ok=True, parents=True)
    with open(out_dir/"metrics"/"per_class_test.json","w",encoding="utf-8") as f:
        json.dump(per_class_test, f, indent=2)

    # Guardados modelo e historial (casting a float)
    model.save(out_dir/"final.keras")
    history_cast = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    with open(out_dir/"history.json","w",encoding="utf-8") as f:
        json.dump(history_cast, f, indent=2)
    print("[FIN] Modelo guardado:", out_dir/"final.keras")

# ============== CLI =========================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--metrics_macro", action="store_true")
    ap.add_argument("--lr_patience", type=int, default=12)
    ap.add_argument("--es_patience", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)
    return ap.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
