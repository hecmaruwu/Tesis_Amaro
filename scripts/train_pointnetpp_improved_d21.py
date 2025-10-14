#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena PointNet++ (mejorado) y monitorea métricas por clase + diente 21 en cada época.
Formateado para dataset .npz:
  X_*.npz["X"] -> (N, P, 3)
  Y_*.npz["Y"] -> (N, P) con ints [0..C-1]

Salidas:
- Modelo final y best checkpoint
- history.json y eval_test.json
- artifacts/label_map.json
- per-class en valid: out_dir/per_class/per_class_epoch_{E}.csv
- tracking diente 21: out_dir/per_class/d21_metrics.csv  (acumula por época)
- per-class en test: out_dir/per_class/per_class_test.json
- TensorBoard: out_dir/tb/  (scalars: d21_precision, d21_recall, d21_f1, d21_iou)
"""

import os, json, argparse, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------ Utils reproducibilidad/GPU ------------------------
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

# ------------------------------ Carga .npz ----------------------------------
def load_npz_split(data_path: Path):
    Xtr = np.load(data_path/"X_train.npz")["X"]
    Ytr = np.load(data_path/"Y_train.npz")["Y"]
    Xva = np.load(data_path/"X_val.npz")["X"]
    Yva = np.load(data_path/"Y_val.npz")["Y"]
    Xte = np.load(data_path/"X_test.npz")["X"]
    Yte = np.load(data_path/"Y_test.npz")["Y"]
    return Xtr, Ytr, Xva, Yva, Xte, Yte

def write_label_map(artifacts_dir: Path, classes: int):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    id2idx = {str(i): i for i in range(classes)}
    idx2id = {i: str(i) for i in range(classes)}
    (artifacts_dir/"label_map.json").write_text(
        json.dumps({"id2idx": id2idx, "idx2id": idx2id}, indent=2),
        encoding="utf-8"
    )

# ----------------------- Augment (básico, determinista) ---------------------
def _augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05, rng=None):
    if rng is None:
        rng = tf.random.get_global_generator()
    theta = rng.uniform([], minval=-rot_deg, maxval=rot_deg, dtype=tf.float32) * (tf.constant(np.pi)/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], axis=0)
    x = tf.matmul(x, R)
    scale = rng.uniform([], minval=scale_low, maxval=scale_high, dtype=tf.float32)
    x = x * scale
    noise = rng.normal(tf.shape(x), stddev=jitter_std, dtype=tf.float32)
    return x + noise, y

def make_datasets(Xtr, Ytr, Xva, Yva, Xte, Yte, batch_size=8, augment=False, seed=42,
                  rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05):
    gen = tf.random.Generator.from_seed(seed)
    AUTOTUNE = tf.data.AUTOTUNE

    def _ds(x, y, shuffle=False, aug=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda a,b: (tf.cast(a, tf.float32), tf.cast(b, tf.int32)),
                    num_parallel_calls=AUTOTUNE)
        if aug:
            ds = ds.map(lambda a,b: _augment_cloud(a,b,rot_deg,jitter_std,scale_low,scale_high,rng=gen),
                        num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    return (_ds(Xtr,Ytr,shuffle=True,  aug=augment),
            _ds(Xva,Yva,shuffle=False, aug=False),
            _ds(Xte,Yte,shuffle=False, aug=False))

# ------------------- Núcleo PointNet++ (Set Abstraction) --------------------
def index_points(points, idx):
    # points: (B,N,C); idx: (B,S) o (B,S,K)
    if len(idx.shape) == 2:
        B = tf.shape(points)[0]; S = tf.shape(idx)[1]
        batch_indices = tf.tile(tf.reshape(tf.range(B), [B,1]), [1,S])  # (B,S)
        gather_idx = tf.stack([batch_indices, idx], axis=-1)            # (B,S,2)
        return tf.gather_nd(points, gather_idx)
    else:
        B = tf.shape(points)[0]; S = tf.shape(idx)[1]; K = tf.shape(idx)[2]
        batch_indices = tf.tile(tf.reshape(tf.range(B), [B,1,1]), [1,S,K])
        gather_idx = tf.stack([batch_indices, idx], axis=-1)
        return tf.gather_nd(points, gather_idx)

# --- FPS robusto usando numpy (evita while_loop shape issues en TF) ---------
def _fps_numpy_batch(xyz_np, npoint, seed=42):
    """
    xyz_np: (B,N,3) numpy
    return idx: (B,npoint) numpy int32
    """
    rng = np.random.default_rng(seed)
    B, N, _ = xyz_np.shape
    out = np.zeros((B, npoint), dtype=np.int32)
    for b in range(B):
        pts = xyz_np[b]
        # inicial aleatorio
        far = rng.integers(0, N)
        out_b = np.zeros((npoint,), dtype=np.int32)
        dist = np.full((N,), 1e10, dtype=np.float32)
        for i in range(npoint):
            out_b[i] = far
            d = np.sum((pts - pts[far])**2, axis=1)
            dist = np.minimum(dist, d)
            far = int(np.argmax(dist))
        out[b] = out_b
    return out.astype(np.int32)

def farthest_point_sampling(xyz, npoint, seed=42):
    """
    xyz: (B,N,3) tensor -> (B,npoint) indices. Implementado vía numpy_function.
    """
    def _wrapper(x):
        return _fps_numpy_batch(x, int(npoint), seed)
    idx = tf.numpy_function(func=_wrapper, inp=[xyz], Tout=tf.int32)
    # fija shape parcial para Keras
    B = tf.shape(xyz)[0]
    idx.set_shape([None, int(npoint)])
    return idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """ xyz:(B,N,3) new_xyz:(B,S,3) -> idx:(B,S,nsample) """
    dist = tf.reduce_sum((tf.expand_dims(new_xyz,2) - tf.expand_dims(xyz,1))**2, axis=-1)   # (B,S,N)
    mask = dist <= radius**2
    dist_masked = tf.where(mask, dist, tf.fill(tf.shape(dist), tf.constant(np.inf, dtype=tf.float32)))
    idx = tf.argsort(dist_masked, axis=-1)[:, :, :nsample]
    return idx

class SetAbstraction(layers.Layer):
    def __init__(self, npoint, radius, nsample, mlp_out, name=None, seed=42):
        super().__init__(name=name)
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.seed = int(seed)
        self.mlp1 = keras.Sequential([layers.Dense(mlp_out, activation='relu'),
                                      layers.Dense(mlp_out, activation='relu')])
        self.mlp_w = keras.Sequential([layers.Dense(mlp_out, activation='relu'),
                                       layers.Dense(mlp_out, activation='sigmoid')])
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"npoint": self.npoint, "radius": self.radius,
                    "nsample": self.nsample, "mlp_out": self.mlp1.layers[-1].units,
                    "seed": self.seed})
        return cfg

    def call(self, xyz, features, training=False):
        B = tf.shape(xyz)[0]
        fps_idx = farthest_point_sampling(xyz, self.npoint, seed=self.seed)   # (B,S)
        new_xyz = index_points(xyz, fps_idx)                                  # (B,S,3)

        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz) # (B,S,K)
        grouped_xyz = index_points(xyz, group_idx)                             # (B,S,K,3)
        grouped_xyz_norm = grouped_xyz - tf.expand_dims(new_xyz, axis=2)

        if features is None:
            grouped_feats = grouped_xyz_norm
        else:
            grouped_feats = index_points(features, group_idx)                  # (B,S,K,C)

        f_in = tf.concat([grouped_xyz_norm, grouped_feats], axis=-1)          # (B,S,K,3+C)
        f_prime = self.mlp1(f_in)
        f_mean = tf.reduce_mean(f_prime, axis=2, keepdims=True)
        w_in = tf.concat([grouped_xyz_norm, f_prime - f_mean], axis=-1)
        alpha = self.mlp_w(w_in)                                              # (B,S,K,C')
        f_out = tf.reduce_sum(alpha * f_prime, axis=2)                        # (B,S,C')
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
        dists = tf.reduce_sum((tf.expand_dims(xyz1,2) - tf.expand_dims(xyz2,1))**2, axis=-1)  # (B,N1,N2)
        idx = tf.argsort(dists, axis=-1)[:, :, :3]                                            # (B,N1,3)
        d3 = tf.gather(dists, idx, batch_dims=2)
        d3 = tf.maximum(d3, 1e-10)
        w = 1.0 / d3
        w = w / tf.reduce_sum(w, axis=-1, keepdims=True)
        neighbor_feats = index_points(feat2, idx)   # (B,N1,3,C2)
        interpolated = tf.reduce_sum(tf.expand_dims(w,-1) * neighbor_feats, axis=2)           # (B,N1,C2)
        new_feat = interpolated if feat1 is None else tf.concat([interpolated, feat1], axis=-1)
        return self.mlp(new_feat)

def build_improved_pointnetpp(num_points, num_classes, base=64, dropout=0.5, seed=42):
    xyz_in = layers.Input(shape=(num_points, 3), name="xyz")
    xyz_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(xyz_in)
    xyz_centered = layers.Subtract()([xyz_in, xyz_mean])
    zeros = layers.Lambda(lambda t: tf.zeros_like(t))(xyz_in)
    spfe_in = layers.Concatenate(axis=-1)([xyz_in, zeros, xyz_centered])  # (B,P,9)
    spfe = layers.Dense(64, activation='relu')(spfe_in)
    spfe = layers.Dense(64, activation='relu')(spfe)

    l1_xyz, l1_feat = SetAbstraction(npoint=num_points//4,  radius=0.05, nsample=32, mlp_out=base,   seed=seed)(xyz_in, spfe)
    l2_xyz, l2_feat = SetAbstraction(npoint=num_points//8,  radius=0.10, nsample=32, mlp_out=base*2, seed=seed+1)(l1_xyz, l1_feat)
    l3_xyz, l3_feat = SetAbstraction(npoint=num_points//16, radius=0.20, nsample=32, mlp_out=base*4, seed=seed+2)(l2_xyz, l2_feat)

    fp2 = FeaturePropagation(base*2)(l2_xyz, l2_feat, l3_xyz, l3_feat)
    fp1 = FeaturePropagation(base   )(l1_xyz, l1_feat, l2_xyz, fp2)
    fp0 = FeaturePropagation(base   )(xyz_in,  spfe,   l1_xyz, fp1)

    x = layers.Dense(base*2, activation='relu')(fp0)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(num_classes, activation=None, name="logits")(x)
    out = layers.Activation("softmax", name="softmax")(logits)
    return keras.Model(inputs=xyz_in, outputs=out, name="PointNetPP_Improved")

# ----------------------- Métricas y callbacks por-clase ---------------------
def confusion_from_preds(y_true, y_pred, num_classes):
    """
    y_true/y_pred: (N, P) enteros
    return CxC matrix
    """
    t = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    mask = (t >= 0) & (t < num_classes)
    t = t[mask]; p = p[mask]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = t * num_classes + p
    binc = np.bincount(idx, minlength=num_classes*num_classes)
    cm[:] = binc.reshape((num_classes, num_classes))
    return cm

def per_class_metrics_from_cm(cm):
    """
    cm: CxC
    return dicts: precision, recall, f1, iou  (cada uno: array C)
    """
    C = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
    recall    = np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)>0)
    f1        = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall)>0)
    iou       = np.divide(tp, tp+fp+fn, out=np.zeros_like(tp), where=(tp+fp+fn)>0)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou)

class PerClassMetricsCallback(keras.callbacks.Callback):
    """
    - Evalúa en VALID al final de cada época: confusion, métricas por clase.
    - Guarda per-class CSV por época.
    - Trackea diente 21 (class_idx_21) y lo loguea en consola, CSV acumulativo y TensorBoard.
    """
    def __init__(self, val_ds, out_dir: Path, num_classes: int, class_idx_21: int = 21):
        super().__init__()
        self.val_ds = val_ds
        self.out_dir = Path(out_dir)
        self.num_classes = int(num_classes)
        self.c21 = int(class_idx_21)
        (self.out_dir/"per_class").mkdir(parents=True, exist_ok=True)
        # TB
        self.tb_writer = tf.summary.create_file_writer(str(self.out_dir/"tb"))

        # cabecera CSV d21
        d21_path = self.out_dir/"per_class"/"d21_metrics.csv"
        if not d21_path.exists():
            with open(d21_path, "w", encoding="utf-8") as f:
                f.write("epoch,precision,recall,f1,iou\n")

    def on_epoch_end(self, epoch, logs=None):
        # acumula preds/labels en VALID
        all_t, all_p = [], []
        for xb, yb in self.val_ds:
            pb = self.model.predict(xb, verbose=0)  # (B,P,C)
            yhat = np.argmax(pb, axis=-1).astype(np.int32)  # (B,P)
            all_t.append(yb.numpy().astype(np.int32))
            all_p.append(yhat)
        y_true = np.concatenate(all_t, axis=0)
        y_pred = np.concatenate(all_p, axis=0)

        cm = confusion_from_preds(y_true, y_pred, self.num_classes)
        mets = per_class_metrics_from_cm(cm)

        # dump per-class epoch CSV
        per_epoch_csv = self.out_dir/"per_class"/f"per_class_epoch_{epoch+1:03d}.csv"
        with open(per_epoch_csv, "w", encoding="utf-8") as f:
            f.write("class_idx,precision,recall,f1,iou\n")
            for k in range(self.num_classes):
                f.write(f"{k},{mets['precision'][k]:.6f},{mets['recall'][k]:.6f},{mets['f1'][k]:.6f},{mets['iou'][k]:.6f}\n")

        # extra: diente 21
        if 0 <= self.c21 < self.num_classes:
            p21 = float(mets['precision'][self.c21])
            r21 = float(mets['recall'][self.c21])
            f21 = float(mets['f1'][self.c21])
            i21 = float(mets['iou'][self.c21])

            # imprime y guarda CSV acumulado
            print(f"[D21][epoch {epoch+1}] prec={p21:.4f} rec={r21:.4f} f1={f21:.4f} iou={i21:.4f}")
            with open(self.out_dir/"per_class"/"d21_metrics.csv", "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{p21:.6f},{r21:.6f},{f21:.6f},{i21:.6f}\n")

            # TensorBoard scalars
            with self.tb_writer.as_default():
                tf.summary.scalar("d21_precision", p21, step=epoch+1)
                tf.summary.scalar("d21_recall",    r21, step=epoch+1)
                tf.summary.scalar("d21_f1",        f21, step=epoch+1)
                tf.summary.scalar("d21_iou",       i21, step=epoch+1)
                self.tb_writer.flush()

# ------------------------------ Entrenamiento -------------------------------
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
        rot_deg=args.rot_deg,
        jitter_std=args.jitter_std,
        scale_low=args.scale_low,
        scale_high=args.scale_high
    )

    model = build_improved_pointnetpp(P, classes, base=args.base_channels, dropout=args.dropout, seed=args.seed)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=metrics)

    ckpt_dir = out_dir/"ckpts"; ckpt_dir.mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt_dir/"best.keras"),
                                        monitor="val_accuracy", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=args.lr_patience, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.es_patience,
                                      restore_best_weights=True),
        PerClassMetricsCallback(val_ds=ds_va, out_dir=out_dir, num_classes=classes, class_idx_21=args.class_idx_21)
    ]

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, verbose=1, callbacks=callbacks)

    # Eval test y per-class (test)
    eval_te = model.evaluate(ds_te, verbose=1)
    eval_names = [m.name if hasattr(m, "name") else f"m{i}" for i,m in enumerate(model.metrics)]
    with open(out_dir/"eval_test.json","w",encoding="utf-8") as f:
        json.dump({k: float(v) for k,v in zip(eval_names, eval_te)}, f, indent=2)

    # per-class test
    all_t, all_p = [], []
    for xb, yb in ds_te:
        pb = model.predict(xb, verbose=0)
        yhat = np.argmax(pb, axis=-1).astype(np.int32)
        all_t.append(yb.numpy().astype(np.int32))
        all_p.append(yhat)
    y_true = np.concatenate(all_t, axis=0)
    y_pred = np.concatenate(all_p, axis=0)
    cm = confusion_from_preds(y_true, y_pred, classes)
    mets = per_class_metrics_from_cm(cm)
    (out_dir/"per_class").mkdir(exist_ok=True, parents=True)
    with open(out_dir/"per_class"/"per_class_test.json","w",encoding="utf-8") as f:
        json.dump({k: list(map(float, v)) for k,v in mets.items()}, f, indent=2)

    # Guardados
    model.save(out_dir/"final.keras")
    write_label_map(out_dir/"artifacts", classes)
    # Historial
    safe_hist = {k:[float(x) for x in v] for k,v in hist.history.items()}
    with open(out_dir/"history.json","w",encoding="utf-8") as f:
        json.dump(safe_hist, f, indent=2)

    print("[FIN] Modelo guardado en:", out_dir/"final.keras")

# ---------------------------------- CLI -------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--rot_deg", type=float, default=10.0)
    ap.add_argument("--jitter_std", type=float, default=0.005)
    ap.add_argument("--scale_low", type=float, default=0.95)
    ap.add_argument("--scale_high", type=float, default=1.05)
    ap.add_argument("--metrics_macro", action="store_true")  # reservado
    ap.add_argument("--lr_patience", type=int, default=12)
    ap.add_argument("--es_patience", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--class_idx_21", type=int, default=21, help="Índice de clase del diente 21")
    return ap.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
