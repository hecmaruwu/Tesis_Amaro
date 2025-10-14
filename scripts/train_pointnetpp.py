#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pointnetpp.py
Entrena un modelo PointNet++ (segmentation head) para etiquetado por punto.

Entrada esperada (como en train_models.py):
  data_path/
    X_train.npz {"X": (Ntr,P,3)}   Y_train.npz {"Y": (Ntr,P)}
    X_val.npz   {"X": (Nva,P,3)}   Y_val.npz   {"Y": (Nva,P)}
    X_test.npz  {"X": (Nte,P,3)}   Y_test.npz  {"Y": (Nte,P)}

Salida:
  out_dir/tag/run_single/{checkpoints,best,final_model,history.json,config.json,tb,...}

Ejemplo:
  python -m scripts.train_pointnetpp \
    --data_path data/3dteethseg/splits/8192_seed42_pairs \
    --out_dir  runs_grid/pointnetpp_8192_pairs_s42 \
    --tag      run_single \
    --epochs 150 --batch_size 8 --lr 1e-3 \
    --metrics_macro --augment --seed 42
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Semillas / determinismo práctico
# ============================================================
def seed_all(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ============================================================
# GPU / Entorno
# ============================================================
def setup_devices(cuda_visible: str | None, multi_gpu: bool):
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPUs visibles: {len(gpus)}")
        except Exception as e:
            print("[WARN] No pude setear memory_growth:", e)
    else:
        print("[INFO] GPUs visibles: [] (CPU)")

    strategy = None
    if multi_gpu:
        g = tf.config.list_physical_devices('GPU')
        if len(g) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"[INFO] MirroredStrategy con {len(g)} GPUs.")
        else:
            print("[WARN] --multi_gpu activado pero solo 1/0 GPU.")
    return strategy

# ============================================================
# Datos (.npz) + Normalización/Augment (idéntico a train_models.py)
# ============================================================
def load_npz_splits(data_path: Path):
    p = Path(data_path)
    Xtr = np.load(p/"X_train.npz")["X"]     # (N,P,3)
    Ytr = np.load(p/"Y_train.npz")["Y"]     # (N,P)
    Xva = np.load(p/"X_val.npz")["X"]
    Yva = np.load(p/"Y_val.npz")["Y"]
    Xte = np.load(p/"X_test.npz")["X"]
    Yte = np.load(p/"Y_test.npz")["Y"]
    print(f"[DATA] X_train:{Xtr.shape} X_val:{Xva.shape} X_test:{Xte.shape}")
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte), {"num_classes": num_classes}

@tf.function
def _normalize_cloud(x):
    # x: (P,3)
    x = tf.cast(x, tf.float32)
    mean = tf.reduce_mean(x, axis=-2, keepdims=True)  # (1,3)
    x = x - mean
    r = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))  # (P,1)
    r = tf.reduce_max(r, axis=-2, keepdims=True)                      # (1,1)
    x = x / (r + 1e-6)
    return x

def _augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, scale_low=0.9, scale_high=1.1):
    # Rotación pequeña en Z
    theta = tf.random.uniform([], minval=-rot_deg, maxval=rot_deg, dtype=tf.float32) * (np.pi/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], axis=0)  # (3,3)
    x = tf.matmul(x, R)
    # Escala isotrópica leve
    s = tf.random.uniform([], minval=scale_low, maxval=scale_high, dtype=tf.float32)
    x = x * s
    # Jitter suave
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=jitter_std, dtype=tf.float32)
    x = x + noise
    return x, y

def make_datasets(data_path: str, batch_size: int, seed: int,
                  do_augment: bool, rot_deg: float, jitter_std: float,
                  scale_low: float, scale_high: float):
    (Xtr,Ytr), (Xva,Yva), (Xte,Yte), info = load_npz_splits(data_path)

    def _ds(X, Y, shuffle=False, augment=False):
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(4096, X.shape[0]), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda x,y: (_normalize_cloud(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x,y: _augment_cloud(x, y, rot_deg, jitter_std, scale_low, scale_high),
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_tr = _ds(Xtr, Ytr, shuffle=True,  augment=do_augment)
    ds_va = _ds(Xva, Yva, shuffle=False, augment=False)
    ds_te = _ds(Xte, Yte, shuffle=False, augment=False)
    return ds_tr, ds_va, ds_te, info

# ============================================================
# Métricas macro y mIoU (como en train_models.py)
# ============================================================
class _ConfusionMatrixMetric(tf.keras.metrics.Metric):
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
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_pred = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        self.cm.assign_add(cm)
    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))
    def reset_states(self):
        self.reset_state()

class PrecisionMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp   = tf.linalg.tensor_diag_part(cm)
        pred = tf.reduce_sum(cm, axis=0)
        prec = tf.math.divide_no_nan(tp, pred)
        return tf.reduce_mean(prec)

class RecallMacro(_ConfusionMatrixMetric):
    def result(self):
        cm = self.cm
        tp = tf.linalg.tensor_diag_part(cm)
        gt = tf.reduce_sum(cm, axis=1)
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

# ============================================================
# Utils geométricos (FPS, kNN, index_points)
# ============================================================
def index_points(points, idx):
    """
    points: (B, N, C)
    idx: (B, S) o (B, S, K)
    return: (B, S, C) o (B, S, K, C)
    """
    points = tf.convert_to_tensor(points)
    idx    = tf.convert_to_tensor(idx)
    B      = tf.shape(points)[0]
    C      = tf.shape(points)[-1]
    idx_shape = tf.shape(idx)                # [B,S] o [B,S,K]
    flat_idx  = tf.reshape(idx, [B, -1])     # (B, S*K)
    gathered  = tf.gather(points, flat_idx, batch_dims=1)  # (B, S*K, C)
    out_shape = tf.concat([idx_shape, [C]], axis=0)
    out = tf.reshape(gathered, out_shape)
    return out

def pairwise_dist2(a, b):
    """
    a: (B, Na, 3), b: (B, Nb, 3)
    return: (B, Na, Nb) dist^2
    """
    a2 = tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)    # (B,Na,1)
    b2 = tf.reduce_sum(tf.square(b), axis=-1, keepdims=True)    # (B,Nb,1)
    ab = tf.matmul(a, b, transpose_b=True)                      # (B,Na,Nb)
    d2 = a2 - 2.0*ab + tf.transpose(b2, perm=[0,2,1])
    return tf.maximum(d2, 0.0)

def farthest_point_sample(xyz, npoint):
    """
    xyz: (B, N, 3), npoint: int
    devuelve: (B, npoint) índices
    Implementación simple (O(npoint*N)) usando loop tf. Suficiente para P~8k.
    """
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]

    centroids = tf.TensorArray(tf.int32, size=npoint)
    distances = tf.ones([B, N], dtype=tf.float32) * 1e10
    farthest  = tf.zeros([B], dtype=tf.int32)  # inicial (0) por batch

    def body(i, farthest, distances, centroids):
        idx = tf.expand_dims(farthest, axis=1)       # (B,1)
        centroid = index_points(xyz, tf.expand_dims(idx, axis=1))   # (B,1,1,3)
        centroid = tf.squeeze(centroid, axis=[1,2])  # (B,3)
        d2 = tf.reduce_sum(tf.square(xyz - tf.expand_dims(centroid, axis=1)), axis=-1)  # (B,N)
        distances = tf.minimum(distances, d2)
        farthest = tf.argmax(distances, axis=-1, output_type=tf.int32)  # (B,)
        centroids = centroids.write(i, farthest)
        return i+1, farthest, distances, centroids

    i = tf.constant(0)
    _, farthest, distances, centroids = tf.while_loop(
        cond=lambda i, *args: i < npoint,
        body=body,
        loop_vars=[i, farthest, distances, centroids],
        parallel_iterations=1
    )
    idxs = tf.transpose(centroids.stack(), perm=[1,0])  # (B, npoint)
    return idxs

def knn_group(xyz, query_xyz, k):
    """
    xyz: (B,N,3) base
    query_xyz: (B,S,3) centros
    retorna (idx: (B,S,K), dist2: (B,S,K))
    """
    d2 = pairwise_dist2(query_xyz, xyz)      # (B,S,N)
    # top_k entrega los mayores; usamos signo negativo para los menores
    vals, idx = tf.math.top_k(-d2, k=k)      # (B,S,K)
    return idx, -vals

# ============================================================
# Bloques PN++: Set Abstraction (kNN) y Feature Propagation
# ============================================================
class SetAbstraction(layers.Layer):
    def __init__(self, npoint, nsample, mlp_sizes, name=None):
        super().__init__(name=name)
        self.npoint = int(npoint)
        self.nsample = int(nsample)
        self.mlps = [layers.Conv2D(c, kernel_size=1, activation='relu') for c in mlp_sizes]
        self.bn   = [layers.BatchNormalization() for _ in mlp_sizes]

    def call(self, xyz, features=None, training=False):
        # xyz: (B,N,3), features: (B,N,C) o None
        B = tf.shape(xyz)[0]
        N = tf.shape(xyz)[1]
        S = tf.constant(self.npoint, dtype=tf.int32)
        # 1) FPS -> (B,S)
        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx)            # (B,S,3)

        # 2) kNN (sobre xyz completo) -> (B,S,K)
        idx, _ = knn_group(xyz, new_xyz, self.nsample)
        grouped_xyz = index_points(xyz, idx)            # (B,S,K,3)
        grouped_xyz = grouped_xyz - tf.expand_dims(new_xyz, axis=2)  # local coords

        if features is not None:
            grouped_feat = index_points(features, idx)  # (B,S,K,C)
            new_feat = tf.concat([grouped_xyz, grouped_feat], axis=-1)  # (B,S,K,3+C)
        else:
            new_feat = grouped_xyz  # (B,S,K,3)

        # 3) MLP 1x1 sobre K (tratamos K como "anchura" adicional con Conv2D)
        x = tf.expand_dims(new_feat, axis=3)  # (B,S,K,Channels,1) -> no, necesitamos (B,S,K,Channels)
        x = tf.squeeze(x, axis=3)             # (B,S,K,Channels)
        # Conv2D espera (B,H,W,C_in) => usamos H=S, W=K
        x = x  # (B,S,K,Cin)
        for conv, bn in zip(self.mlps, self.bn):
            x = conv(x)            # 1x1 conv en (S,K)
            x = bn(x, training=training)
        # 4) Max-pool sobre vecinos K
        x = tf.reduce_max(x, axis=2)          # (B,S,Cout)
        return new_xyz, x

class FeaturePropagation(layers.Layer):
    def __init__(self, mlp_sizes, name=None):
        super().__init__(name=name)
        self.convs = [layers.Conv1D(c, 1, activation='relu') for c in mlp_sizes]
        self.bns   = [layers.BatchNormalization() for _ in mlp_sizes]

    def call(self, xyz1, xyz2, feat1, feat2, training=False):
        """
        Interpola desde (xyz2, feat2) -> (xyz1, ?), concat con feat1 (skip), pasa MLP1D.
        xyz1: (B,N1,3) (más denso),  feat1: (B,N1,C1) o None
        xyz2: (B,N2,3) (más grueso),  feat2: (B,N2,C2)
        """
        B = tf.shape(xyz1)[0]
        N1 = tf.shape(xyz1)[1]
        # 3-NN
        d2 = pairwise_dist2(xyz1, xyz2)      # (B,N1,N2)
        neg_d2 = -d2
        vals, idx = tf.math.top_k(neg_d2, k=3)   # (B,N1,3) -> menores dist
        d2_nn = -vals + 1e-10                   # evitar div 0
        w = 1.0 / d2_nn
        w = w / tf.reduce_sum(w, axis=-1, keepdims=True)   # (B,N1,3)

        interpolated = tf.reduce_sum(index_points(feat2, idx) * tf.expand_dims(w, axis=-1), axis=2)  # (B,N1,C2)
        if feat1 is not None:
            new_feat = tf.concat([interpolated, feat1], axis=-1)
        else:
            new_feat = interpolated

        x = new_feat
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x, training=training)
        return x  # (B,N1,Cout)

# ============================================================
# Modelo PointNet++ (segmentation)
# ============================================================
def build_pointnetpp_seg(num_points: int, num_classes: int, base=64, dropout=0.5):
    inputs = layers.Input(shape=(num_points, 3), name="points")

    # SA (usamos kNN con nsample=32)
    l0_xyz = inputs
    l0_feat = None

    l1_xyz, l1_feat = SetAbstraction(npoint=num_points//4,  nsample=32, mlp_sizes=[base, base])(l0_xyz, l0_feat)
    l2_xyz, l2_feat = SetAbstraction(npoint=num_points//16, nsample=32, mlp_sizes=[base*2, base*2])(l1_xyz, l1_feat)
    l3_xyz, l3_feat = SetAbstraction(npoint=num_points//64, nsample=32, mlp_sizes=[base*2, base*4])(l2_xyz, l2_feat)

    # FP (arriba)
    l2_up = FeaturePropagation(mlp_sizes=[base*2])(l2_xyz, l3_xyz, l2_feat, l3_feat)
    l1_up = FeaturePropagation(mlp_sizes=[base*2])(l1_xyz, l2_xyz, l1_feat, l2_up)
    l0_up = FeaturePropagation(mlp_sizes=[base])(l0_xyz, l1_xyz, l0_feat, l1_up)

    x = layers.Dropout(dropout)(l0_up)
    x = layers.Conv1D(base, 1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Conv1D(num_classes, 1, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PointNetPP_Seg")
    return model

# ============================================================
# Optimizador + guardados
# ============================================================
def get_optimizer(args):
    if args.optimizer.lower() == "adam":
        try:
            return keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
        except Exception:
            return keras.optimizers.Adam(learning_rate=args.lr)
    elif args.optimizer.lower() == "sgd":
        return keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Optimizer no soportado: {args.optimizer}")

def save_model_summary(model, run_dir: Path):
    txt = []
    model.summary(print_fn=lambda s: txt.append(s))
    (run_dir/"model_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    try:
        keras.utils.plot_model(model, to_file=str(run_dir/"model.png"),
                               show_shapes=True, expand_nested=True)
    except Exception as e:
        (run_dir/"model_plot_error.txt").write_text(str(e), encoding="utf-8")

def save_history(history, run_dir: Path):
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    (run_dir/"history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

# ============================================================
# Entrenamiento
# ============================================================
def train_once(args, run_dir: Path, hparams: dict):
    seed_all(args.seed)
    ds_tr, ds_va, ds_te, dinfo = make_datasets(
        args.data_path, hparams["batch_size"], args.seed,
        do_augment=args.augment,
        rot_deg=args.rot_deg, jitter_std=args.jitter_std,
        scale_low=args.scale_low, scale_high=args.scale_high
    )
    num_classes = dinfo["num_classes"]

    # Crear modelo (descubrir P del dataset)
    spec = ds_tr.element_spec[0].shape
    P = int(spec[1]) if spec.rank is not None and spec[1] is not None else None
    if P is None:
        xb, _ = next(iter(ds_tr))
        P = int(xb.shape[1])

    strategy = setup_devices(args.cuda, args.multi_gpu)
    def compile_model():
        model = build_pointnetpp_seg(P, num_classes, base=args.base_channels, dropout=args.dropout)
        metrics = ['accuracy']
        if args.metrics_macro:
            metrics += [
                PrecisionMacro(num_classes=num_classes, name="prec_macro"),
                RecallMacro(num_classes=num_classes,    name="rec_macro"),
                F1Macro(num_classes=num_classes,        name="f1_macro"),
                SparseMeanIoU(num_classes=num_classes,  name="miou"),
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

    # Build para shapes
    _ = model(tf.zeros([1, P, 3], dtype=tf.float32), training=False)
    save_model_summary(model, run_dir)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                          patience=args.lr_patience, min_lr=args.min_lr, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.es_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_dir/"best"),
                                        save_best_only=True, monitor='val_loss',
                                        save_format='tf', verbose=1),
        keras.callbacks.CSVLogger(str(run_dir/"train_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(run_dir/"tb")),
    ]
    (run_dir/"config.json").write_text(json.dumps({"hparams": hparams, "args": vars(args)}, indent=2), encoding="utf-8")

    # Smoke (opcional)
    if args.smoke:
        print("[SMOKE] ejecutando ensayo corto…")
        ds_tr = ds_tr.take(args.smoke_batches)
        ds_va = ds_va.take(max(1, args.smoke_batches // 2))
        hparams["epochs"] = 1

    print("[HPARAMS]", hparams)
    history = model.fit(ds_tr, validation_data=ds_va,
                        epochs=hparams["epochs"], verbose=1, callbacks=callbacks)
    save_history(history, run_dir)

    print("[EVAL] Test:")
    test_metrics = model.evaluate(ds_te, verbose=1, return_dict=True)
    (run_dir/"test_metrics.json").write_text(json.dumps({k: float(v) for k, v in test_metrics.items()}, indent=2),
                                             encoding="utf-8")

    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_dir, save_format='tf')
    print(f"[FIN] Modelo guardado en: {final_dir}")

# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--tag",       required=True)

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--activation", default="relu")   # no usado aquí, pero por consistencia
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)

    ap.add_argument("--optimizer", default="adam", choices=["adam","sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lr_patience", type=int, default=12)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--es_patience", type=int, default=20)

    ap.add_argument("--cuda", default=None)
    ap.add_argument("--multi_gpu", action="store_true")

    ap.add_argument("--metrics_macro", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_batches", type=int, default=2)

    ap.add_argument("--augment", dest="augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--rot_deg", type=float, default=10.0)
    ap.add_argument("--jitter_std", type=float, default=0.005)
    ap.add_argument("--scale_low", type=float, default=0.90)
    ap.add_argument("--scale_high", type=float, default=1.10)

    args = ap.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / args.tag / "run_single"
    run_dir.mkdir(parents=True, exist_ok=True)

    hp = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "base_channels": args.base_channels,
    }
    train_once(args, run_dir, hp)

if __name__ == "__main__":
    main()
