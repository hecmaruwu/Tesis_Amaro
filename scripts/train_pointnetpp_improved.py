#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointNet++ Mejorado (SPFE + WSLFA) para segmentación de dientes por punto.

Dataset esperado (.npz):
  X_train.npz["X"] -> (N, P, 3)  float32
  Y_train.npz["Y"] -> (N, P)     int [0..C-1]

- Normaliza cada nube (centrado y radio máx=1)
- Augment suave determinista (seed fija)
- Métricas macro propias (precision, recall, f1, mIoU) acumuladas sobre batches
- Guarda: run_dir/{final.keras, ckpts/best.keras, history.json, eval_test.json, artifacts/label_map.json}
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ========================= Reproducibilidad & GPU =========================
def set_global_seed(seed: int):
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    tf.random.set_global_generator(tf.random.Generator.from_seed(int(seed)))
    tf.keras.utils.set_random_seed(int(seed))

def allow_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

# ================================ Normalización ================================
@tf.function
def _normalize_cloud(x):
    # x: (P,3)
    x = tf.cast(x, tf.float32)
    mean = tf.reduce_mean(x, axis=-2, keepdims=True)
    x = x - mean
    r = tf.sqrt(tf.reduce_max(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)))  # escalar
    return x / (r + 1e-6)

# =============================== Augment determinista ==========================
def _augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05, rng=None):
    if rng is None:
        rng = tf.random.get_global_generator()

    # Rotación pequeña alrededor de Z
    theta = rng.uniform([], minval=-rot_deg, maxval=rot_deg, dtype=tf.float32) * (tf.constant(np.pi, tf.float32)/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], axis=0)  # (3,3)
    x = tf.matmul(x, R)

    # Escala isotrópica leve
    s_iso = rng.uniform([], minval=scale_low, maxval=scale_high, dtype=tf.float32)
    x = x * s_iso

    # Jitter gaussiano suave
    noise = rng.normal(tf.shape(x), stddev=jitter_std, dtype=tf.float32)
    x = x + noise
    return x, y

# ============================ Data loading & datasets ==========================
def load_npz_split(data_path: Path):
    Xtr = np.load(data_path/"X_train.npz")["X"].astype(np.float32)
    Ytr = np.load(data_path/"Y_train.npz")["Y"].astype(np.int32)
    Xva = np.load(data_path/"X_val.npz")["X"].astype(np.float32)
    Yva = np.load(data_path/"Y_val.npz")["Y"].astype(np.int32)
    Xte = np.load(data_path/"X_test.npz")["X"].astype(np.float32)
    Yte = np.load(data_path/"Y_test.npz")["Y"].astype(np.int32)
    return Xtr, Ytr, Xva, Yva, Xte, Yte

def make_datasets(Xtr, Ytr, Xva, Yva, Xte, Yte, batch_size=8, augment=False, seed=42):
    AUTOTUNE = tf.data.AUTOTUNE
    rng = tf.random.get_global_generator()

    def _ds(X, Y, shuffle=False, aug=False):
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(4096, X.shape[0]), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda x,y: (_normalize_cloud(x), tf.cast(y, tf.int32)), num_parallel_calls=AUTOTUNE)
        if aug:
            ds = ds.map(lambda x,y: _augment_cloud(x, y, rng=rng), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    return (
        _ds(Xtr, Ytr, shuffle=True,  aug=augment),
        _ds(Xva, Yva, shuffle=False, aug=False),
        _ds(Xte, Yte, shuffle=False, aug=False),
    )

def write_label_map(artifacts_dir: Path, classes: int):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    id2idx = {str(i): i for i in range(classes)}
    idx2id = {i: str(i) for i in range(classes)}
    (artifacts_dir/"label_map.json").write_text(
        json.dumps({"id2idx": id2idx, "idx2id": idx2id}, indent=2), encoding="utf-8"
    )

# ========================== Utils de agrupación/vecinos ========================
def farthest_point_sampling(xyz, npoint):
    """
    FPS simple en TF: xyz (B,N,3) -> índices (B, npoint).
    Determinista vía generador global (ya inicializado).
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    rng = tf.random.get_global_generator()

    # primer índice aleatorio por batch
    farthest = rng.uniform([B], minval=0, maxval=N, dtype=tf.int32)

    # distancias iniciales al primer punto
    centroid_xyz = tf.gather(xyz, farthest, batch_dims=1)                       # (B,3)
    dist = tf.reduce_sum(tf.square(xyz - tf.expand_dims(centroid_xyz, 1)), -1)  # (B,N)

    idx_list = [farthest]
    i = 1
    def body(i, dist, idx_list):
        farthest_i = tf.argmax(dist, axis=-1, output_type=tf.int32)             # (B,)
        centroid_xyz = tf.gather(xyz, farthest_i, batch_dims=1)                 # (B,3)
        d = tf.reduce_sum(tf.square(xyz - tf.expand_dims(centroid_xyz, 1)), -1) # (B,N)
        dist = tf.minimum(dist, d)
        idx_list.append(farthest_i)
        return i+1, dist, idx_list

    while i < npoint:
        i, dist, idx_list = body(i, dist, idx_list)

    idx = tf.stack(idx_list, axis=1)  # (B, npoint)
    return idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    xyz: (B,N,3), new_xyz: (B,S,3)
    Retorna idx (B,S,nsample) con los más cercanos dentro de radius (rellena con los más cercanos si faltan).
    """
    dists = tf.reduce_sum((tf.expand_dims(new_xyz,2) - tf.expand_dims(xyz,1))**2, axis=-1)  # (B,S,N)
    mask = dists <= radius**2
    d_mask = tf.where(mask, dists, tf.fill(tf.shape(dists), tf.constant(np.inf, dtype=tf.float32)))
    idx = tf.argsort(d_mask, axis=-1)[:, :, :nsample]   # (B,S,nsample)
    return idx

def index_points(points, idx):
    """
    points: (B,N,C); idx: (B,S) o (B,S,K)
    return: (B,S,C) o (B,S,K,C)
    """
    if len(idx.shape) == 2:
        B = tf.shape(points)[0]
        S = tf.shape(idx)[1]
        b = tf.tile(tf.reshape(tf.range(B), [B,1]), [1,S])
        gather_idx = tf.stack([b, idx], axis=-1)       # (B,S,2)
        return tf.gather_nd(points, gather_idx)
    else:
        B = tf.shape(points)[0]
        S = tf.shape(idx)[1]
        K = tf.shape(idx)[2]
        b = tf.tile(tf.reshape(tf.range(B), [B,1,1]), [1,S,K])
        gather_idx = tf.stack([b, idx], axis=-1)       # (B,S,K,2)
        return tf.gather_nd(points, gather_idx)

# ============================== Bloques del modelo ==============================
class SetAbstraction(layers.Layer):
    """
    SA con WSLFA (atención local):
      - FPS
      - Ball Query
      - f' = MLP([Δp, feat])
      - α = MLP([Δp, f' - mean(f')]) con sigmoid
      - f = sum_k α_k ⊙ f'_k
    """
    def __init__(self, npoint, radius, nsample, mlp_out, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.mlp_out = int(mlp_out)
        self.mlp_feat = keras.Sequential([
            layers.Dense(self.mlp_out, activation='relu'),
            layers.Dense(self.mlp_out, activation='relu'),
        ])
        self.mlp_w = keras.Sequential([
            layers.Dense(self.mlp_out, activation='relu'),
            layers.Dense(self.mlp_out, activation='sigmoid'),
        ])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "npoint": self.npoint,
            "radius": self.radius,
            "nsample": self.nsample,
            "mlp_out": self.mlp_out,
        })
        return cfg

    def call(self, xyz, features):
        S = self.npoint
        idx = farthest_point_sampling(xyz, S)                 # (B,S)
        new_xyz = index_points(xyz, idx)                      # (B,S,3)

        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)   # (B,S,K)
        grouped_xyz = index_points(xyz, group_idx)                               # (B,S,K,3)
        dxyz = grouped_xyz - tf.expand_dims(new_xyz, axis=2)                    # (B,S,K,3)

        if features is None:
            grouped_feats = dxyz
        else:
            grouped_feats = index_points(features, group_idx)                   # (B,S,K,C)

        f_in = tf.concat([dxyz, grouped_feats], axis=-1)                        # (B,S,K,3+C)
        f_prime = self.mlp_feat(f_in)                                           # (B,S,K,C')

        f_mean = tf.reduce_mean(f_prime, axis=2, keepdims=True)                 # (B,S,1,C')
        w_in = tf.concat([dxyz, f_prime - f_mean], axis=-1)                     # (B,S,K,3+C')
        alpha = self.mlp_w(w_in)                                                # (B,S,K,C') en (0,1)

        f_weighted = alpha * f_prime                                            # (B,S,K,C')
        f_out = tf.reduce_sum(f_weighted, axis=2)                               # (B,S,C')
        return new_xyz, f_out

class FeaturePropagation(layers.Layer):
    """
    Interpolación 3NN + inverse-distance + skip + MLP
    """
    def __init__(self, mlp_out, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mlp_out = int(mlp_out)
        self.mlp = keras.Sequential([
            layers.Dense(self.mlp_out, activation='relu'),
            layers.Dense(self.mlp_out, activation='relu'),
        ])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"mlp_out": self.mlp_out})
        return cfg

    def call(self, xyz1, feat1, xyz2, feat2):
        # upsample de (xyz2,feat2) -> xyz1; concat con feat1; MLP
        # xyz1:(B,N1,3), feat1:(B,N1,C1) | xyz2:(B,N2,3), feat2:(B,N2,C2)
        dists = tf.reduce_sum((tf.expand_dims(xyz1,2) - tf.expand_dims(xyz2,1))**2, axis=-1)  # (B,N1,N2)
        idx = tf.argsort(dists, axis=-1)[:, :, :3]                                            # (B,N1,3)
        d3 = tf.gather(dists, idx, batch_dims=2)                                              # (B,N1,3)
        d3 = tf.maximum(d3, 1e-10)
        w = 1.0 / d3
        w = w / tf.reduce_sum(w, axis=-1, keepdims=True)                                      # (B,N1,3)
        neigh = index_points(feat2, idx)                                                      # (B,N1,3,C2)
        interpol = tf.reduce_sum(tf.expand_dims(w, -1) * neigh, axis=2)                       # (B,N1,C2)

        if feat1 is None:
            new_feat = interpol
        else:
            new_feat = tf.concat([interpol, feat1], axis=-1)                                  # (B,N1,C1+C2)
        return self.mlp(new_feat)

# ================================ Modelo =================================
def build_improved_pointnetpp(num_points, num_classes, base=64, dropout=0.5):
    xyz_in = layers.Input(shape=(num_points, 3), name="xyz")

    # SPFE: 9 canales = [xyz | normales_dummy | xyz_centered]
    xyz_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(xyz_in)
    xyz_centered = layers.Subtract()([xyz_in, xyz_mean])
    zeros = layers.Lambda(lambda t: tf.zeros_like(t))(xyz_in)
    spfe_in = layers.Concatenate(axis=-1)([xyz_in, zeros, xyz_centered])  # (B,P,9)
    spfe = layers.Dense(64, activation='relu')(spfe_in)
    spfe = layers.Dense(64, activation='relu')(spfe)

    # SA (WSLFA)
    l1_xyz, l1_feat = SetAbstraction(npoint=num_points//4,  radius=0.05, nsample=32, mlp_out=base)(xyz_in, spfe)
    l2_xyz, l2_feat = SetAbstraction(npoint=num_points//8,  radius=0.10, nsample=32, mlp_out=base*2)(l1_xyz, l1_feat)
    l3_xyz, l3_feat = SetAbstraction(npoint=num_points//16, radius=0.20, nsample=32, mlp_out=base*4)(l2_xyz, l2_feat)

    # FP
    fp2 = FeaturePropagation(base*2)(l2_xyz, l2_feat, l3_xyz, l3_feat)  # -> N2
    fp1 = FeaturePropagation(base)(l1_xyz, l1_feat, l2_xyz, fp2)        # -> N1
    fp0 = FeaturePropagation(base)(xyz_in,  spfe,    l1_xyz, fp1)       # -> N

    x = layers.Dense(base*2, activation='relu')(fp0)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(num_classes, activation=None, name="logits")(x)
    out = layers.Activation("softmax", name="softmax")(logits)          # (B,P,C)
    return keras.Model(inputs=xyz_in, outputs=out, name="PointNetPP_Improved")

# ================================ Métricas =================================
class _ConfusionMatrixMetric(keras.metrics.Metric):
    def __init__(self, num_classes: int, name="cm_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.cm = self.add_weight(
            name="cm",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])       # (B*P,)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)  # (B,P)
        y_pred = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true, y_pred,
                                      num_classes=self.num_classes, dtype=tf.float32)
        self.cm.assign_add(cm)
    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))
    def reset_states(self):  # compat
        self.reset_state()

class PrecisionMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        pred = tf.reduce_sum(cm, axis=0)  # TP+FP
        prec = tf.math.divide_no_nan(tp, pred)
        return tf.reduce_mean(prec)

class RecallMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp = tf.linalg.tensor_diag_part(cm)
        gt = tf.reduce_sum(cm, axis=1)    # TP+FN
        rec = tf.math.divide_no_nan(tp, gt)
        return tf.reduce_mean(rec)

class F1Macro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        gt   = tf.reduce_sum(cm, axis=1)
        pred = tf.reduce_sum(cm, axis=0)
        prec = tf.math.divide_no_nan(tp, pred)
        rec  = tf.math.divide_no_nan(tp, gt)
        f1 = tf.math.divide_no_nan(2.0 * prec * rec, prec + rec)
        return tf.reduce_mean(f1)

class SparseMeanIoU(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        gt   = tf.reduce_sum(cm, axis=1)
        pred = tf.reduce_sum(cm, axis=0)
        union = gt + pred - tp
        iou = tf.math.divide_no_nan(tp, union)
        return tf.reduce_mean(iou)

def make_metrics(num_classes, macro=False):
    m = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if macro:
        m += [
            PrecisionMacro(num_classes=num_classes, name="prec_macro"),
            RecallMacro(num_classes=num_classes,    name="rec_macro"),
            F1Macro(num_classes=num_classes,        name="f1_macro"),
            SparseMeanIoU(num_classes=num_classes,  name="miou"),
        ]
    return m

# ================================ Entrenamiento ================================
def train_once(args):
    set_global_seed(args.seed)
    allow_gpu_memory_growth()

    data_dir = Path(args.data_path)
    run_dir  = Path(args.out_dir) / args.tag
    (run_dir/"artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir/"ckpts").mkdir(parents=True, exist_ok=True)

    Xtr, Ytr, Xva, Yva, Xte, Yte = load_npz_split(data_dir)
    P = Xtr.shape[1]
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)

    print(f"[DATA] X_train:{Xtr.shape} X_val:{Xva.shape} X_test:{Xte.shape}")
    print(f"[INFO] GPUs visibles: {tf.config.list_physical_devices('GPU')}")

    ds_tr, ds_va, ds_te = make_datasets(
        Xtr, Ytr, Xva, Yva, Xte, Yte,
        batch_size=args.batch_size,
        augment=args.augment,
        seed=args.seed
    )

    model = build_improved_pointnetpp(P, num_classes, base=args.base_channels, dropout=args.dropout)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    metrics = make_metrics(num_classes, macro=args.metrics_macro)

    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=metrics
    )

    # Build para summary correcto
    _ = model(tf.zeros([1, P, 3], dtype=tf.float32), training=False)
    # Guardar summary
    txt = []
    model.summary(print_fn=lambda s: txt.append(s))
    (run_dir/"model_summary.txt").write_text("\n".join(txt), encoding="utf-8")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir/"ckpts/best.keras"),
            monitor='val_loss', save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=args.lr_patience, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.es_patience, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(str(run_dir/"train_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(run_dir/"tb")),
    ]

    print("[HPARAMS]", {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "dropout": args.dropout, "base_channels": args.base_channels
    })

    history = model.fit(
        ds_tr, validation_data=ds_va,
        epochs=args.epochs, verbose=1, callbacks=callbacks
    )

    # Guardar history
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    (run_dir/"history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    # Evaluación
    print("[EVAL] Test:")
    test_metrics = model.evaluate(ds_te, verbose=1, return_dict=True)
    (run_dir/"eval_test.json").write_text(
        json.dumps({k: float(v) for k, v in test_metrics.items()}, indent=2), encoding="utf-8"
    )

    # Modelo final
    model.save(run_dir/"final.keras")
    write_label_map(run_dir/"artifacts", num_classes)
    print(f"[FIN] Modelo guardado en: {run_dir/'final.keras'}")

# ================================== CLI ==================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="Carpeta con X_*.npz / Y_*.npz")
    ap.add_argument("--out_dir",   required=True, help="Carpeta base de salidas")
    ap.add_argument("--tag",       required=True, help="Subcarpeta (p.ej. run_single)")
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
    train_once(args)

if __name__ == "__main__":
    main()
