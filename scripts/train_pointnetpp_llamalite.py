#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento: PointNet++ (SPFE + WSLFA) + Encoder Transformer "LLaMA-lite".
Compatibilidad:
 - Carga npz: X_train/val/test (N, P, 3) y Y_* (N, P) con etiquetas [0..C-1]
 - Métricas macro opcionales (prec/rec/F1) si no están disponibles se omiten
 - Semillas fijas para reproducibilidad (incluida la data pipeline)

Uso:
python -m scripts.train_pointnetpp_llamalite \
  --data_path DATA_ROOT/splits/8192_seed42_pairs \
  --out_dir  runs_grid/pointnetppLL_8192_pairs_s42 \
  --tag      run_llite \
  --epochs 150 --batch_size 8 --lr 1e-3 \
  --metrics_macro --augment --seed 42
"""
import os, json, argparse, math, numpy as np, tensorflow as tf
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------
# Reproducibilidad fuerte
# ---------------------------------------------------------------------
def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)              # tf, numpy, python
    tf.config.experimental.enable_op_determinism()    # ops deterministas

# ---------------------------------------------------------------------
# Utilidades dataset
# ---------------------------------------------------------------------
def load_split_npz(data_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(data_path/f"X_{split}.npz")["X"].astype(np.float32) # (N,P,3)
    Y = np.load(data_path/f"Y_{split}.npz")["Y"].astype(np.int32)   # (N,P)
    return X, Y

def make_tfds(X, Y, batch, shuffle, augment, seed):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x,y: (x, tf.cast(y, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(lambda x,y: augment_cloud(x,y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------
# Aumentación determinista (seed por op usando Generator)
# ---------------------------------------------------------------------
def augment_cloud(x, y):
    # rotación aleatoria leve alrededor de Z + jitter + escala
    g = tf.random.Generator.from_non_deterministic_state()
    theta = g.uniform([], minval=-5.0, maxval=5.0, dtype=tf.float32) * (math.pi/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c,-s,0.0],
                  [s, c,0.0],
                  [0.0,0.0,1.0]], axis=0)
    x = tf.matmul(x, R)

    jitter = g.normal(tf.shape(x), mean=0.0, stddev=0.002, dtype=tf.float32)
    x = x + jitter
    scale = g.uniform([], minval=0.98, maxval=1.02, dtype=tf.float32)
    x = x * scale
    return x, y

# ---------------------------------------------------------------------
# Núcleo PointNet++ (FPS aproximado via farthest-point sampling en tf)
# ---------------------------------------------------------------------
def pairwise_distance(xyz):  # (B,N,3)->(B,N,N)
    xx = tf.reduce_sum(tf.square(xyz), axis=-1, keepdims=True)
    dist = xx - 2.0*tf.matmul(xyz, xyz, transpose_b=True) + tf.transpose(xx, [0,2,1])
    dist = tf.maximum(dist, 0.0)
    return dist

def farthest_point_sampling(xyz, npoint):
    # xyz:(B,N,3) -> idx:(B,npoint)
    B = tf.shape(xyz)[0]
    N = tf.shape(xyz)[1]
    centroids = tf.TensorArray(tf.int32, size=npoint)
    distances = tf.ones((B, N), dtype=tf.float32) * 1e10
    farthest = tf.zeros((B,), dtype=tf.int32)

    def body(i, distances, farthest, centroids):
        centroids = centroids.write(i, farthest)
        centroid = tf.gather(xyz, farthest, batch_dims=1)  # (B,3)
        dist = tf.reduce_sum(tf.square(xyz - tf.expand_dims(centroid, 1)), axis=-1)  # (B,N)
        distances = tf.minimum(distances, dist)
        farthest = tf.argmax(distances, axis=-1, output_type=tf.int32)
        return i+1, distances, farthest, centroids

    i = tf.constant(0)
    _, _, _, centroids = tf.while_loop(lambda i, *_: i < npoint, body,
                                       [i, distances, farthest, centroids])
    idx = tf.transpose(centroids.stack(), [1,0])  # (B,npoint)
    return idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    # retorna los índices de los K vecinos más cercanos dentro de un radio (aprox con knn)
    # xyz:(B,N,3) new_xyz:(B,S,3) -> idx:(B,S,K)
    B = tf.shape(xyz)[0]; N = tf.shape(xyz)[1]; S = tf.shape(new_xyz)[1]
    dists = tf.reduce_sum((tf.expand_dims(new_xyz, 2) - tf.expand_dims(xyz, 1))**2, axis=-1)  # (B,S,N)
    # KNN
    k = tf.minimum(nsample, N)
    idx = tf.argsort(dists, axis=-1)[:,:,:k]  # (B,S,K)
    return idx

def index_points(points, idx):
    # points:(B,N,C), idx:(B,S,K) or (B,S)
    B = tf.shape(points)[0]
    if len(idx.shape) == 3:
        S = tf.shape(idx)[1]; K = tf.shape(idx)[2]
        b = tf.reshape(tf.range(B), (B,1,1))
        b = tf.tile(b, (1,S,K))
        gather_idx = tf.stack([b, idx], axis=-1)  # (B,S,K,2)
        out = tf.gather_nd(points, gather_idx)
    else:
        S = tf.shape(idx)[1]
        b = tf.reshape(tf.range(B), (B,1))
        b = tf.tile(b, (1,S))
        gather_idx = tf.stack([b, idx], axis=-1)
        out = tf.gather_nd(points, gather_idx)
    return out

class SetAbstraction(tf.keras.layers.Layer):
    def __init__(self, npoint, nsample, mlp_sizes, radius=None, name=None):
        super().__init__(name=name)
        self.npoint, self.nsample, self.radius = npoint, nsample, radius
        self.mlps = [tf.keras.layers.Dense(h, activation='relu') for h in mlp_sizes]
        self.bn = [tf.keras.layers.BatchNormalization() for _ in mlp_sizes]
        # para WSLFA (pesos)
        self.mlp_w = [tf.keras.layers.Dense(h, activation='relu') for h in mlp_sizes]
        self.bn_w = [tf.keras.layers.BatchNormalization() for _ in mlp_sizes]

    def call(self, xyz, features, training=False):
        # xyz:(B,N,3), features:(B,N,C) or None
        B = tf.shape(xyz)[0]; N = tf.shape(xyz)[1]
        # FPS
        idx = farthest_point_sampling(xyz, self.npoint)          # (B,S)
        new_xyz = index_points(xyz, idx)                         # (B,S,3)
        # vecinos
        group_idx = query_ball_point(self.radius or 1e9, self.nsample, xyz, new_xyz)  # (B,S,K)
        grouped_xyz = index_points(xyz, group_idx)               # (B,S,K,3)
        grouped_xyz_norm = grouped_xyz - tf.expand_dims(new_xyz, 2)  # (B,S,K,3)

        if features is None:
            grouped_features = grouped_xyz_norm  # usa coords relativas
        else:
            grouped_features = index_points(features, group_idx) # (B,S,K,C)
            grouped_features = tf.concat([grouped_xyz_norm, grouped_features], axis=-1)

        # SPFE + MLP (aplicado punto a punto dentro de la vecindad)
        h = grouped_features
        for dense, bn in zip(self.mlps, self.bn):
            h = dense(h); h = bn(h, training=training)

        # WSLFA: aprender pesos por punto y hacer suma ponderada (∑ α ⊙ f)
        w = grouped_features
        for dense, bn in zip(self.mlp_w, self.bn_w):
            w = dense(w); w = bn(w, training=training)
        # normalización de pesos (softmax sobre K)
        w = tf.nn.softmax(w, axis=2)

        out = tf.reduce_sum(w * h, axis=2)  # (B,S,C')
        return new_xyz, out

class FeaturePropagation(tf.keras.layers.Layer):
    def __init__(self, mlp_sizes, name=None):
        super().__init__(name=name)
        self.mlps = [tf.keras.layers.Dense(h, activation='relu') for h in mlp_sizes]
        self.bn = [tf.keras.layers.BatchNormalization() for _ in mlp_sizes]

    def call(self, xyz1, feat1, xyz2, feat2, training=False):
        # interpola feat2(xyz2) -> xyz1 y concat con feat1
        dists = tf.reduce_sum((tf.expand_dims(xyz1,2)-tf.expand_dims(xyz2,1))**2, axis=-1)  # (B,N1,N2)
        idx = tf.argsort(dists, axis=-1)[:,:, :3]  # 3-NN
        d = tf.gather(dists, idx, batch_dims=2) + 1e-8
        norm = tf.reduce_sum(1.0/d, axis=-1, keepdims=True)
        w = (1.0/d)/norm
        f2_knn = index_points(feat2, idx)                      # (B,N1,3,C2)
        interp = tf.reduce_sum(w[...,None]*f2_knn, axis=2)     # (B,N1,C2)

        h = tf.concat([interp, feat1], axis=-1)
        for dense, bn in zip(self.mlps, self.bn):
            h = dense(h); h = bn(h, training=training)
        return h

# ---------------------------------------------------------------------
# “LLaMA-lite” Transformer encoder (muy pequeño) con RoPE sobre xyz
# ---------------------------------------------------------------------
def rotary_embed(xyz, dim):
    # xyz:(B,P,3) -> pos:(B,P,dim) periódico con frecuencias log-espaciadas
    B = tf.shape(xyz)[0]; P = tf.shape(xyz)[1]
    freqs = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(10000.0), dim//6))
    # expandir xyz en [sin, cos] por cada eje y frecuencia
    comps = []
    for ax in range(3):
        a = tf.expand_dims(xyz[:,:,ax], -1)  # (B,P,1)
        ang = a * tf.reshape(freqs, (1,1,-1))  # (B,P,F)
        comps += [tf.sin(ang), tf.cos(ang)]
    pos = tf.concat(comps, axis=-1)  # (B,P,6F) ≈ dim
    if tf.shape(pos)[-1] < dim:
        pos = tf.pad(pos, [[0,0],[0,0],[0,dim-tf.shape(pos)[-1]]])
    else:
        pos = pos[...,:dim]
    return pos

class TinyTransformer(tf.keras.layers.Layer):
    def __init__(self, d_model=256, n_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.qkv = tf.keras.layers.Dense(3*d_model, use_bias=False)
        self.proj = tf.keras.layers.Dense(d_model)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(int(d_model*mlp_ratio), activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, xyz, x, training=False):
        # xyz:(B,P,3), x:(B,P,C=d_model)
        B = tf.shape(x)[0]; P = tf.shape(x)[1]
        h = self.norm1(x)
        qkv = self.qkv(h)  # (B,P,3C)
        q, k, v = tf.split(qkv, 3, axis=-1)
        # reshape heads
        def reshape_heads(t):
            return tf.transpose(tf.reshape(t, (B, P, self.n_heads, self.d_model//self.n_heads)), [0,2,1,3])
        q = reshape_heads(q); k = reshape_heads(k); v = reshape_heads(v)

        # Rotary PE
        rope = rotary_embed(xyz, self.d_model)
        rope = tf.transpose(tf.reshape(rope, (B, P, self.n_heads, self.d_model//self.n_heads)), [0,2,1,3])
        q = q + rope; k = k + rope

        attn = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.d_model//self.n_heads)  # (B,H,P,P)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.drop(attn, training=training)
        z = tf.matmul(attn, v)                             # (B,H,P,dh)
        z = tf.transpose(z, [0,2,1,3])
        z = tf.reshape(z, (B, P, self.d_model))
        z = self.drop(self.proj(z), training=training)
        x = x + z
        # FFN
        x = x + self.mlp(self.norm2(x), training=training)
        return x

# ---------------------------------------------------------------------
# Construcción del modelo
# ---------------------------------------------------------------------
def build_model(points_per_cloud, num_classes, base=64, d_model=256, n_heads=4, n_tx=1, dropout=0.5):
    xyz_in = tf.keras.Input(shape=(points_per_cloud, 3), name="xyz")

    # SPFE: 9 -> 64 (usamos xyz, normales=0, coords centradas=xyz-mean)
    mean = tf.reduce_mean(xyz_in, axis=1, keepdims=True)
    zmean = xyz_in - mean
    normals = tf.zeros_like(xyz_in)
    spfe_in = tf.concat([xyz_in, normals, zmean], axis=-1)
    h = tf.keras.layers.Dense(64, activation='relu')(spfe_in)
    h = tf.keras.layers.Dense(64, activation='relu')(h)

    # SA1/2/3 con WSLFA
    l1_xyz, l1 = SetAbstraction(npoint=points_per_cloud//4,  nsample=32, mlp_sizes=[base, base])(xyz_in, h)
    l2_xyz, l2 = SetAbstraction(npoint=points_per_cloud//8,  nsample=32, mlp_sizes=[base*2, base*2])(l1_xyz, l1)
    l3_xyz, l3 = SetAbstraction(npoint=points_per_cloud//32, nsample=32, mlp_sizes=[base*4, base*4])(l2_xyz, l2)

    # Proyección a d_model
    e = tf.keras.layers.Dense(d_model, activation='relu')(l3)

    # ---------- Encoder “LLaMA-lite” (1-2 bloques) ----------
    x_tx = e
    for _ in range(n_tx):
        x_tx = TinyTransformer(d_model=d_model, n_heads=n_heads, mlp_ratio=2.0, dropout=0.1)(l3_xyz, x_tx)

    # Decoder con FP (skip)
    up2 = FeaturePropagation([base*2, base*2])(l2_xyz, l2, l3_xyz, x_tx)
    up1 = FeaturePropagation([base, base])(l1_xyz, l1, l2_xyz, up2)
    up0 = FeaturePropagation([base, base])(xyz_in,  h,  l1_xyz, up1)

    out = tf.keras.layers.Dropout(dropout)(up0)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(out)
    probs  = tf.keras.layers.Softmax(name="probs")(logits)

    return tf.keras.Model(inputs=xyz_in, outputs=probs, name="PointNetPP_LLaMA_Lite")

# ---------------------------------------------------------------------
# Métricas macro opcionales
# ---------------------------------------------------------------------
def make_macro_metrics(num_classes):
    def _one_hot(y_true):
        return tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    def prec_m(y_true, y_pred):
        yt = _one_hot(y_true); yp = tf.one_hot(tf.argmax(y_pred,-1), num_classes)
        TP = tf.reduce_sum(yt*yp, axis=[0,1]); PP = tf.reduce_sum(yp, axis=[0,1])+1e-8
        prec_c = TP/PP
        return tf.reduce_mean(prec_c)
    def rec_m(y_true, y_pred):
        yt = _one_hot(y_true); yp = tf.one_hot(tf.argmax(y_pred,-1), num_classes)
        TP = tf.reduce_sum(yt*yp, axis=[0,1]); P  = tf.reduce_sum(yt, axis=[0,1])+1e-8
        rec_c = TP/P
        return tf.reduce_mean(rec_c)
    def f1_m(y_true, y_pred):
        p = prec_m(y_true,y_pred); r = rec_m(y_true,y_pred)
        return 2*p*r/(p+r+1e-8)
    return prec_m, rec_m, f1_m

# ---------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="run_llite")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metrics_macro", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--tx_blocks", type=int, default=1)
    ap.add_argument("--tx_heads", type=int, default=4)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    data_dir = Path(args.data_path)
    Xtr, Ytr = load_split_npz(data_dir, "train")
    Xva, Yva = load_split_npz(data_dir, "val")
    Xte, Yte = load_split_npz(data_dir, "test")
    P = Xtr.shape[1]
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)

    print(f"[DATA] X_train:{Xtr.shape} X_val:{Xva.shape} X_test:{Xte.shape}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] GPUs visibles: {gpus}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    ds_tr = make_tfds(Xtr, Ytr, args.batch_size, shuffle=True,  augment=args.augment, seed=args.seed)
    ds_va = make_tfds(Xva, Yva, args.batch_size, shuffle=False, augment=False,      seed=args.seed)
    ds_te = make_tfds(Xte, Yte, args.batch_size, shuffle=False, augment=False,      seed=args.seed)

    model = build_model(points_per_cloud=P, num_classes=num_classes,
                        base=args.base_channels, d_model=256,
                        n_heads=args.tx_heads, n_tx=args.tx_blocks,
                        dropout=args.dropout)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if args.metrics_macro:
        try:
            p,r,f1 = make_macro_metrics(num_classes)
            metrics += [p, r, f1]
        except Exception as e:
            print("[WARN] Macro métricas no disponibles:", e)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=metrics
    )

    run_dir = Path(args.out_dir)/args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ck = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir/"best.keras"),
        monitor="val_accuracy", mode="max", save_best_only=True
    )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1)
    tb = tf.keras.callbacks.TensorBoard(log_dir=str(run_dir/"tb"))
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    print("[HPARAMS]", {"epochs":args.epochs,"batch_size":args.batch_size,"lr":args.lr,
                         "dropout":args.dropout,"base_channels":args.base_channels,
                         "tx_blocks":args.tx_blocks,"tx_heads":args.tx_heads})

    hist = model.fit(ds_tr, epochs=args.epochs, validation_data=ds_va, callbacks=[ck, rlrop, tb, es])

    # Eval final
    print("[EVAL] Mejor modelo en val_accuracy, evaluando en test...")
    try:
        best = tf.keras.models.load_model(run_dir/"best.keras", compile=False,
                                          custom_objects={"TinyTransformer":TinyTransformer,
                                                          "SetAbstraction":SetAbstraction,
                                                          "FeaturePropagation":FeaturePropagation})
        best.compile(optimizer="adam",
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics=metrics)
        results = best.evaluate(ds_te, verbose=0)
        print(dict(zip(best.metrics_names, results)))
    except Exception as e:
        print("[WARN] No se pudo recargar best.keras:", e)
        results = model.evaluate(ds_te, verbose=0)
        print(dict(zip(model.metrics_names, results)))

    # Guardar artefactos
    model.save(run_dir/"final.keras")
    meta = {
        "points_per_cloud": P,
        "num_classes": num_classes,
        "seed": args.seed,
        "augment": args.augment,
        "metrics": model.metrics_names,
        "history": {k:[float(x) for x in v] for k,v in hist.history.items()}
    }
    (run_dir/"meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[FIN] Modelo guardado")
    print("[RUN_DIR]", str(run_dir))

if __name__ == "__main__":
    main()
