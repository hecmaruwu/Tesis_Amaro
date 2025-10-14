#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train (8k pts) & Infer (many pts) for Improved PointNet++ segmentation (teeth classes)
Dataset formato .npz:
  - X_*.npz["X"] -> (N, P, 3)
  - Y_*.npz["Y"] -> (N, P)  con ints [0..C-1]

Comandos:
  Entrenar (8k):
    python -m scripts.train_pointnetpp_improved_8k_and_infer \
      train --data_path <SPLIT_8k_DIR> --out_dir <RUN_DIR> --epochs 120 --batch_size 8 --seed 42

  Inferir (muchos puntos, p.ej. 16k/32k) con fusión KNN:
    python -m scripts.train_pointnetpp_improved_8k_and_infer \
      infer --model <RUN_DIR>/final.keras \
            --data_path <SPLIT_BIG_DIR> \
            --out_dir <OUT_INFER_DIR> \
            --passes 6 --chunksize 8192 --knn_k 3
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------- Utilidades reproducibilidad/GPU ----------------------
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

# --------------------------- Carga datos .npz -------------------------------
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

# ---------------------------- Augment determinista --------------------------
def _augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05, rng=None):
    if rng is None:
        rng = tf.random.get_global_generator()
    # rotación Z
    theta = rng.uniform([], minval=-rot_deg, maxval=rot_deg, dtype=tf.float32) * (tf.constant(np.pi)/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], axis=0)
    x = tf.matmul(x, R)
    # escala
    scale = rng.uniform([], minval=scale_low, maxval=scale_high, dtype=tf.float32)
    x = x * scale
    # jitter
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

    return (_ds(Xtr,Ytr,shuffle=True,  aug=augment),
            _ds(Xva,Yva,shuffle=False, aug=False),
            _ds(Xte,Yte,shuffle=False, aug=False))

# --------------------------- Núcleo PointNet++ (mejorado) -------------------
def index_points(points, idx):
    # points: (B,N,C); idx: (B,S) o (B,S,K)
    if len(idx.shape) == 2:
        B = tf.shape(points)[0]
        S = tf.shape(idx)[1]
        batch_indices = tf.tile(tf.reshape(tf.range(B), [B,1]), [1,S])  # (B,S)
        gather_idx = tf.stack([batch_indices, idx], axis=-1)            # (B,S,2)
        return tf.gather_nd(points, gather_idx)
    else:
        B = tf.shape(points)[0]
        S = tf.shape(idx)[1]
        K = tf.shape(idx)[2]
        batch_indices = tf.tile(tf.reshape(tf.range(B), [B,1,1]), [1,S,K])  # (B,S,K)
        gather_idx = tf.stack([batch_indices, idx], axis=-1)                # (B,S,K,2)
        return tf.gather_nd(points, gather_idx)

def farthest_point_sampling(xyz, npoint, rng=None):
    """ FPS simple (determinista si rng está fijado). xyz:(B,N,3) -> (B,npoint) idx """
    if rng is None:
        rng = tf.random.get_global_generator()
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    # inicial
    first = rng.uniform([B], minval=0, maxval=N, dtype=tf.int32)
    chosen = tf.expand_dims(first, -1)     # (B,1)
    chosen_xyz = index_points(xyz, tf.expand_dims(first, -1))[:,0,:]  # (B,3)
    dist_min = tf.reduce_sum((xyz - tf.reshape(chosen_xyz, [B,1,3]))**2, axis=-1)  # (B,N)

    def body(i, chosen, dist_min):
        farthest = tf.argmax(dist_min, axis=-1, output_type=tf.int32)              # (B,)
        far_xyz  = index_points(xyz, tf.expand_dims(farthest, -1))[:,0,:]          # (B,3)
        d = tf.reduce_sum((xyz - tf.reshape(far_xyz,[B,1,3]))**2, axis=-1)
        dist_min = tf.minimum(dist_min, d)
        chosen = tf.concat([chosen, tf.expand_dims(farthest, -1)], axis=1)        # (B,i+1)
        return i+1, chosen, dist_min

    i0 = tf.constant(1)
    cond = lambda i, *_: tf.less(i, npoint)
    shape_inv = [
        i0.get_shape(),
        tf.TensorShape([None, None]),      # chosen: (B, i)
        tf.TensorShape([None, None])       # dist_min: (B, N)
    ]
    _, chosen, _ = tf.while_loop(cond, body, [i0, chosen, dist_min],
                                 shape_invariants=shape_inv, parallel_iterations=1)
    return chosen  # (B, npoint)

def query_ball_point(radius, nsample, xyz, new_xyz):
    """ xyz:(B,N,3) new_xyz:(B,S,3) -> idx:(B,S,nsample) """
    dist = tf.reduce_sum((tf.expand_dims(new_xyz,2) - tf.expand_dims(xyz,1))**2, axis=-1)   # (B,S,N)
    mask = dist <= radius**2
    dist_masked = tf.where(mask, dist, tf.fill(tf.shape(dist), tf.constant(np.inf, dtype=tf.float32)))
    idx = tf.argsort(dist_masked, axis=-1)[:, :, :nsample]
    return idx

class SetAbstraction(layers.Layer):
    def __init__(self, npoint, radius, nsample, mlp_out, name=None):
        super().__init__(name=name)
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)
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
        fps_idx = farthest_point_sampling(xyz, self.npoint)          # (B,S)
        new_xyz = index_points(xyz, fps_idx)                         # (B,S,3)

        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B,S,K)
        grouped_xyz = index_points(xyz, group_idx)                              # (B,S,K,3)
        grouped_xyz_norm = grouped_xyz - tf.expand_dims(new_xyz, axis=2)

        if features is None:
            grouped_feats = grouped_xyz_norm
        else:
            grouped_feats = index_points(features, group_idx)                   # (B,S,K,C)

        f_in = tf.concat([grouped_xyz_norm, grouped_feats], axis=-1)           # (B,S,K,3+C)
        f_prime = self.mlp1(f_in)                                              # (B,S,K,C')
        f_mean = tf.reduce_mean(f_prime, axis=2, keepdims=True)                # (B,S,1,C')
        w_in = tf.concat([grouped_xyz_norm, f_prime - f_mean], axis=-1)        # (B,S,K,3+C')
        alpha = self.mlp_w(w_in)                                               # (B,S,K,C')

        f_weighted = alpha * f_prime
        f_out = tf.reduce_sum(f_weighted, axis=2)                              # (B,S,C')
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
        # 3-NN inverse-distance interpolación
        dists = tf.reduce_sum((tf.expand_dims(xyz1,2) - tf.expand_dims(xyz2,1))**2, axis=-1)  # (B,N1,N2)
        idx = tf.argsort(dists, axis=-1)[:, :, :3]                                            # (B,N1,3)
        d3 = tf.gather(dists, idx, batch_dims=2)                                              # (B,N1,3)
        d3 = tf.maximum(d3, 1e-10)
        w = 1.0 / d3
        w = w / tf.reduce_sum(w, axis=-1, keepdims=True)                                      # (B,N1,3)

        neighbor_feats = index_points(feat2, idx)   # (B,N1,3,C2)
        interpolated = tf.reduce_sum(tf.expand_dims(w,-1) * neighbor_feats, axis=2)           # (B,N1,C2)

        if feat1 is None:
            new_feat = interpolated
        else:
            new_feat = tf.concat([interpolated, feat1], axis=-1)
        return self.mlp(new_feat)

def build_improved_pointnetpp(num_points, num_classes, base=64, dropout=0.5):
    xyz_in = layers.Input(shape=(num_points, 3), name="xyz")
    # SPFE 9 canales: [xyz | normales_dummy(0) | xyz-mean]
    xyz_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(xyz_in)
    xyz_centered = layers.Subtract()([xyz_in, xyz_mean])
    zeros = layers.Lambda(lambda t: tf.zeros_like(t))(xyz_in)
    spfe_in = layers.Concatenate(axis=-1)([xyz_in, zeros, xyz_centered])  # (B,P,9)
    spfe = layers.Dense(64, activation='relu')(spfe_in)
    spfe = layers.Dense(64, activation='relu')(spfe)

    # SA
    l1_xyz, l1_feat = SetAbstraction(npoint=num_points//4,  radius=0.05, nsample=32, mlp_out=base)(xyz_in, spfe)
    l2_xyz, l2_feat = SetAbstraction(npoint=num_points//8,  radius=0.10, nsample=32, mlp_out=base*2)(l1_xyz, l1_feat)
    l3_xyz, l3_feat = SetAbstraction(npoint=num_points//16, radius=0.20, nsample=32, mlp_out=base*4)(l2_xyz, l2_feat)

    # FP
    fp2 = FeaturePropagation(base*2)(l2_xyz, l2_feat, l3_xyz, l3_feat)
    fp1 = FeaturePropagation(base   )(l1_xyz, l1_feat, l2_xyz, fp2)
    fp0 = FeaturePropagation(base   )(xyz_in,  spfe,   l1_xyz, fp1)

    x = layers.Dense(base*2, activation='relu')(fp0)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(num_classes, activation=None, name="logits")(x)
    out = layers.Activation("softmax", name="softmax")(logits)
    return keras.Model(inputs=xyz_in, outputs=out, name="PointNetPP_Improved")

# ------------------------- Métricas (incl. macro batch) ---------------------
def make_metrics(num_classes, macro=False):
    m = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if macro:
        class PrecisionMacro(keras.metrics.Metric):
            def __init__(self, name="prec_macro", **kw):
                super().__init__(name=name, **kw)
                self.tp = self.add_weight("tp", initializer="zeros")
                self.pp = self.add_weight("pp", initializer="zeros")
            def update_state(self, y_true, y_pred, **kw):
                y_hat = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
                matches = tf.cast(tf.equal(y_hat, y_true), tf.float32)
                self.tp.assign_add(tf.reduce_sum(matches))
                self.pp.assign_add(tf.cast(tf.size(y_hat), tf.float32))
            def result(self):
                return tf.math.divide_no_nan(self.tp, self.pp)
            def reset_states(self):
                self.tp.assign(0.); self.pp.assign(0.)
        class RecallMacro(keras.metrics.Metric):
            def __init__(self, name="rec_macro", **kw):
                super().__init__(name=name, **kw)
                self.tp = self.add_weight("tp", initializer="zeros")
                self.pos = self.add_weight("pos", initializer="zeros")
            def update_state(self, y_true, y_pred, **kw):
                y_hat = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
                matches = tf.cast(tf.equal(y_hat, y_true), tf.float32)
                self.tp.assign_add(tf.reduce_sum(matches))
                self.pos.assign_add(tf.cast(tf.size(y_true), tf.float32))
            def result(self):
                return tf.math.divide_no_nan(self.tp, self.pos)
            def reset_states(self):
                self.tp.assign(0.); self.pos.assign(0.)
        class F1Macro(keras.metrics.Metric):
            def __init__(self, name="f1_macro", **kw):
                super().__init__(name=name, **kw)
                self.p = PrecisionMacro(); self.r = RecallMacro()
            def update_state(self, y_true, y_pred, **kw):
                self.p.update_state(y_true, y_pred); self.r.update_state(y_true, y_pred)
            def result(self):
                p = self.p.result(); r = self.r.result()
                return tf.math.divide_no_nan(2*p*r, p+r)
            def reset_states(self):
                self.p.reset_states(); self.r.reset_states()
        class MeanIoUApprox(keras.metrics.Metric):
            def __init__(self, name="miou", **kw):
                super().__init__(name=name, **kw)
                self.acc = keras.metrics.SparseCategoricalAccuracy()
            def update_state(self, y_true, y_pred, **kw):
                self.acc.update_state(y_true, y_pred)
            def result(self):
                # Nota: aproximación para monitoreo (no mIoU exacto por clase)
                return self.acc.result() * 0.02
            def reset_states(self):
                self.acc.reset_states()
        m += [PrecisionMacro(), RecallMacro(), F1Macro(), MeanIoUApprox()]
    return m

# ------------------------------ Entrenamiento -------------------------------
def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

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
        augment=True,
        seed=args.seed,
        rot_deg=10.0, jitter_std=0.005, scale_low=0.95, scale_high=1.05
    )

    model = build_improved_pointnetpp(P, classes, base=args.base_channels, dropout=args.dropout)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    metrics = make_metrics(classes, macro=args.metrics_macro)
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=metrics)

    ckpt_dir = out_dir/"ckpts"; ckpt_dir.mkdir(exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(str(ckpt_dir/"best.keras"),
                                           monitor="val_accuracy", save_best_only=True)
    lr_sched = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.5, patience=args.lr_patience, min_lr=1e-6, verbose=1)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.es_patience, restore_best_weights=True)

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs,
                     verbose=1, callbacks=[ckpt, lr_sched, es])

    eval_te = model.evaluate(ds_te, verbose=1)
    eval_names = [m.name if hasattr(m, "name") else f"m{i}" for i,m in enumerate(model.metrics)]
    with open(out_dir/"eval_test.json","w",encoding="utf-8") as f:
        json.dump({k: float(v) for k,v in zip(eval_names, eval_te)}, f, indent=2)

    model.save(out_dir/"final.keras")
    write_label_map(out_dir/"artifacts", classes)

    # serialización robusta del history (evita float32)
    hist_sanitized = {k: [float(vv) for vv in v] for k, v in hist.history.items()}
    with open(out_dir/"history.json","w",encoding="utf-8") as f:
        json.dump(hist_sanitized, f, indent=2, default=_json_default)

    print("[FIN] Modelo guardado:", out_dir/"final.keras")

# -------------------------- Inferencia: many points -------------------------
def knn_blend_probs(x_full, x_samp, probs_samp, k=3, eps=1e-9):
    """
    x_full:(P,3)  todos los puntos a etiquetar
    x_samp:(M,3)  puntos con probs ya predichas
    probs_samp:(M,C)  probabilidades por clase en muestreados
    Retorna: probs_full:(P,C) por interpolación KNN inversa a la distancia.
    """
    Pts = x_full.shape[0]
    C = probs_samp.shape[1]
    out = np.zeros((Pts, C), dtype=np.float32)

    BLOCK = 8192
    for start in range(0, Pts, BLOCK):
        end = min(Pts, start+BLOCK)
        xf = x_full[start:end]                        # (B,3)
        d2 = np.sum((xf[:,None,:] - x_samp[None,:,:])**2, axis=-1)  # (B,M)
        idx = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
        part = np.take_along_axis(d2, idx, axis=1)
        order = np.argsort(part, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)        # (B,k)
        d = np.sqrt(np.take_along_axis(d2, idx, axis=1)) + eps
        w = 1.0 / d
        w = w / np.sum(w, axis=1, keepdims=True)
        neigh_probs = probs_samp[idx]                        # (B,k,C)
        out[start:end] = np.sum(w[:,:,None] * neigh_probs, axis=1)
    return out

def fps_numpy(x, m, seed=42):
    """FPS en numpy (para inferencia por nube)"""
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    farthest = rng.integers(0, N)
    centroids = np.zeros((m,), dtype=np.int32)
    dist = np.full((N,), 1e10, dtype=np.float32)
    for i in range(m):
        centroids[i] = farthest
        d = np.sum((x - x[farthest])**2, axis=1)
        dist = np.minimum(dist, d)
        farthest = np.argmax(dist)
    return centroids  # (m,)

def infer_large_cloud(x_full, model, chunksize=8192, passes=6, knn_k=3, seed=42):
    """
    x_full:(P,3) -> etiquetas para P puntos usando:
      - Muestras de tamaño 8192 (FPS multi-pase)
      - Predicción prob por muestra
      - Mezcla KNN sobre todos los puntos
    """
    P = x_full.shape[0]
    idx_accum = []
    rng = np.random.default_rng(seed)
    base_idx = fps_numpy(x_full, chunksize, seed=seed)
    idx_accum.append(base_idx)
    for t in range(1, passes):
        seed_t = seed + t*17
        sel = rng.choice(P, size=min(P, chunksize*2), replace=False)
        idx_t = sel[fps_numpy(x_full[sel], min(chunksize, sel.shape[0]), seed=seed_t)]
        idx_accum.append(idx_t)
    idx_samp = np.unique(np.concatenate(idx_accum))
    xs = x_full[idx_samp]                    # (M,3)

    C = model.output_shape[-1]
    probs_s = np.zeros((xs.shape[0], C), dtype=np.float32)
    BS = 8192
    for s in range(0, xs.shape[0], BS):
        e = min(xs.shape[0], s+BS)
        x_batch = xs[s:e][None, ...]         # (1,B,3)
        pb = model.predict(x_batch, verbose=0)[0]  # (B,C)
        probs_s[s:e] = pb.astype(np.float32)

    probs_full = knn_blend_probs(x_full, xs, probs_s, k=knn_k)
    y_hat = np.argmax(probs_full, axis=-1).astype(np.int32)
    return y_hat, probs_full

def infer(args):
    set_global_seed(args.seed)
    allow_gpu_memory_growth()
    data_dir = Path(args.data_path)
    out_dir  = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    custom_objs = {"SetAbstraction": SetAbstraction, "FeaturePropagation": FeaturePropagation}
    model = keras.models.load_model(args.model, custom_objects=custom_objs, compile=False)

    Xtr, Ytr, Xva, Yva, Xte, Yte = load_npz_split(data_dir)
    P = Xte.shape[1]
    print(f"[INFER] N_test={Xte.shape[0]} P={P} chunksize={args.chunksize} passes={args.passes} knn_k={args.knn_k}")

    all_metrics = []
    for i, (x, y) in enumerate(zip(Xte, Yte)):
        y_pred, _ = infer_large_cloud(x, model,
                                      chunksize=args.chunksize,
                                      passes=args.passes,
                                      knn_k=args.knn_k,
                                      seed=args.seed)
        acc = (y_pred == y).mean().item()
        all_metrics.append(acc)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(Xte)}] acc={acc:.4f}")
        np.savez(out_dir/f"pred_test_{i:04d}.npz", y_pred=y_pred, y_true=y)

    mean_acc = float(np.mean(all_metrics)) if all_metrics else 0.0
    with open(out_dir/"infer_summary.json","w",encoding="utf-8") as f:
        json.dump({"mean_point_accuracy": mean_acc,
                   "num_samples": len(all_metrics),
                   "chunksize": args.chunksize,
                   "passes": args.passes,
                   "knn_k": args.knn_k}, f, indent=2)
    print("[INFER] mean_point_accuracy:", mean_acc)

# ---------------------------------- CLI -------------------------------------
def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train (8k)
    tr = sub.add_parser("train")
    tr.add_argument("--data_path", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--metrics_macro", action="store_true")
    tr.add_argument("--lr_patience", type=int, default=10)
    tr.add_argument("--es_patience", type=int, default=15)
    tr.add_argument("--dropout", type=float, default=0.5)
    tr.add_argument("--base_channels", type=int, default=64)

    # infer (many pts)
    inf = sub.add_parser("infer")
    inf.add_argument("--model", required=True)
    inf.add_argument("--data_path", required=True)
    inf.add_argument("--out_dir", required=True)
    inf.add_argument("--chunksize", type=int, default=8192)
    inf.add_argument("--passes", type=int, default=6)
    inf.add_argument("--knn_k", type=int, default=3)
    inf.add_argument("--seed", type=int, default=42)
    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        infer(args)

if __name__ == "__main__":
    main()
