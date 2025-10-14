#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_models_all.py
-------------------
Script maestro para entrenar varios modelos de nubes de puntos (clasificación/segmentación por punto)
sobre splits .npz (X_*.npz, Y_*.npz) con la misma interfaz.

Modelos soportados:
  - pointnet
  - pointnet_plateau (PointNet + ReduceLROnPlateau por defecto)
  - pointnetpp (jerárquico con agrupamiento kNN)
  - pointnetpp_dilated (igual que pointnetpp pero usando dilataciones/atrous en convs 1D)
  - point_region_transformer (transformer ligero por regiones)

Ejemplos:
  python train_models_all.py --data_path data/splits/8192 --out_dir runs --tag exp1 \
      --model pointnet --epochs 60 --batch_size 8 --optimizer adamw --scheduler cosine --metrics_macro

  python train_models_all.py --data_path data/splits/8192 --out_dir runs --tag exp_grid \
      --model all --epochs 60 --batch_size 8 --optimizer adam --scheduler plateau --metrics_macro
"""

import os, json, argparse, math
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Utils: entorno y seeds
# ============================================================
def setup_devices(cuda_visible: str | None, multi_gpu: bool):
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPUs visibles: {len(gpus)} -> {gpus}")
        except Exception as e:
            print("[WARN] memory_growth:", e)
    else:
        print("[INFO] Sin GPU (CPU).")
    strategy = None
    if multi_gpu:
        g = tf.config.list_physical_devices('GPU')
        if len(g) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"[INFO] MirroredStrategy con {len(g)} GPUs.")
        else:
            print("[WARN] --multi_gpu activado pero 1 sola GPU.")
    return strategy

def set_tf_seed(seed: int):
    tf.keras.utils.set_random_seed(seed)

# ============================================================
# Datos (.npz) + normalización/augment
# ============================================================
def load_npz_splits(data_path: Path):
    p = Path(data_path)
    req = ["X_train.npz","Y_train.npz","X_val.npz","Y_val.npz","X_test.npz","Y_test.npz"]
    missing = [r for r in req if not (p/r).exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos en {}:\n  {}".format(p, "\n  ".join(missing)))
    Xtr = np.load(p/"X_train.npz")["X"].astype(np.float32)
    Ytr = np.load(p/"Y_train.npz")["Y"].astype(np.int64)
    Xva = np.load(p/"X_val.npz")["X"].astype(np.float32)
    Yva = np.load(p/"Y_val.npz")["Y"].astype(np.int64)
    Xte = np.load(p/"X_test.npz")["X"].astype(np.float32)
    Yte = np.load(p/"Y_test.npz")["Y"].astype(np.int64)
    num_classes = int(max(Ytr.max(), Yva.max(), Yte.max()) + 1)
    print(f"[DATA] Xtr {Xtr.shape}  Xva {Xva.shape}  Xte {Xte.shape}  num_classes={num_classes}")
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte), {"num_classes": num_classes}

@tf.function
def normalize_cloud(x):
    x = tf.cast(x, tf.float32)
    mu = tf.reduce_mean(x, axis=-2, keepdims=True)
    x = x - mu
    r = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
    r = tf.reduce_max(r, axis=-2, keepdims=True)
    return x / (r + 1e-6)

def augment_cloud(x, y, rot_deg=10.0, jitter_std=0.005, s_low=0.9, s_high=1.1):
    theta = tf.random.uniform([], -rot_deg, rot_deg) * (np.pi/180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.stack([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], axis=0)
    x = tf.matmul(x, R)
    s = tf.random.uniform([], s_low, s_high)
    x = x * s
    noise = tf.random.normal(tf.shape(x), stddev=jitter_std)
    x = x + noise
    return x, y

def make_datasets(data_path: str, batch_size: int, seed: int,
                  do_augment: bool, rot_deg: float, jitter_std: float,
                  scale_low: float, scale_high: float):
    (Xtr,Ytr),(Xva,Yva),(Xte,Yte),info = load_npz_splits(Path(data_path))
    def _ds(X,Y,shuffle=False,augment=False):
        ds = tf.data.Dataset.from_tensor_slices((X,Y))
        if shuffle:
            ds = ds.shuffle(min(4096, X.shape[0]), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda a,b: (normalize_cloud(a), b), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda a,b: augment_cloud(a,b,rot_deg,jitter_std,scale_low,scale_high),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return _ds(Xtr,Ytr,True,do_augment), _ds(Xva,Yva,False,False), _ds(Xte,Yte,False,False), info

# ============================================================
# Métricas macro / mIoU (idénticas a tu script)
# ============================================================
class _CMM(tf.keras.metrics.Metric):
    def __init__(self, num_classes:int, name="cm", **kw):
        super().__init__(name=name, **kw)
        self.num = int(num_classes)
        self.cm = self.add_weight("cm", shape=(self.num,self.num), initializer="zeros", dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        yp = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        yp = tf.reshape(yp, [-1])
        self.cm.assign_add(tf.math.confusion_matrix(yt, yp, num_classes=self.num, dtype=tf.float32))
    def reset_state(self): self.cm.assign(tf.zeros_like(self.cm))
    def reset_states(self): self.reset_state()

class PrecisionMacro(_CMM):
    def result(self):
        cm=self.cm; tp=tf.linalg.diag_part(cm); pred=tf.reduce_sum(cm,axis=0)
        return tf.reduce_mean(tf.math.divide_no_nan(tp, pred))
class RecallMacro(_CMM):
    def result(self):
        cm=self.cm; tp=tf.linalg.diag_part(cm); gt=tf.reduce_sum(cm,axis=1)
        return tf.reduce_mean(tf.math.divide_no_nan(tp, gt))
class F1Macro(_CMM):
    def result(self):
        cm=self.cm; tp=tf.linalg.diag_part(cm); gt=tf.reduce_sum(cm,axis=1); pred=tf.reduce_sum(cm,axis=0)
        p=tf.math.divide_no_nan(tp,pred); r=tf.math.divide_no_nan(tp,gt)
        return tf.reduce_mean(tf.math.divide_no_nan(2*p*r,p+r))
class SparseMeanIoU(_CMM):
    def result(self):
        cm=self.cm; tp=tf.linalg.diag_part(cm); gt=tf.reduce_sum(cm,axis=1); pred=tf.reduce_sum(cm,axis=0)
        union=gt+pred-tp
        return tf.reduce_mean(tf.math.divide_no_nan(tp, union))

# ============================================================
# Optimizadores y Schedulers
# ============================================================
def make_optimizer(name:str, lr:float, weight_decay:float, momentum:float):
    name = name.lower()
    if name == "adamw":
        try:
            return keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        except Exception:
            return keras.optimizers.Adam(learning_rate=lr)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr, momentum=momentum)
    raise ValueError(f"Optimizer no soportado: {name}")

def make_scheduler(name:str, base_lr:float, steps_per_epoch:int, epochs:int):
    name = name.lower()
    if name == "none":
        return None
    if name == "cosine":
        total_steps = steps_per_epoch * max(1, epochs)
        return keras.optimizers.schedules.CosineDecay(initial_learning_rate=base_lr,
                                                      decay_steps=total_steps, alpha=0.1)
    if name == "exponential":
        return keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=base_lr,
                                                           decay_steps=max(1, steps_per_epoch),
                                                           decay_rate=0.96, staircase=True)
    if name == "plateau":
        # se maneja por callback ReduceLROnPlateau
        return base_lr
    # por defecto none
    return None

# ============================================================
# Pérdidas
# ============================================================
class WeightedSparseCE(tf.keras.losses.Loss):
    def __init__(self, class_weights: dict, num_classes: int, name="w_sce"):
        super().__init__(name=name)
        w = np.ones((num_classes,), dtype=np.float32)
        for k,v in (class_weights or {}).items():
            k=int(k)
            if 0<=k<num_classes: w[k]=float(v)
        self.w = tf.constant(w, dtype=tf.float32)
    def call(self, y_true, y_pred):
        y_true=tf.cast(y_true, tf.int32)
        base=tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        w=tf.gather(self.w, y_true)
        return tf.reduce_mean(base*w)

class FocalSparseCE(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha_vec=None, num_classes=None, name="focal_sce"):
        super().__init__(name=name); self.gamma=float(gamma)
        if alpha_vec is not None:
            a=np.ones((num_classes,), dtype=np.float32)
            for k,v in alpha_vec.items():
                k=int(k); 
                if 0<=k<num_classes: a[k]=float(v)
            self.alpha=tf.constant(a, dtype=tf.float32)
        else:
            self.alpha=None
    def call(self, y_true, y_pred):
        eps=tf.keras.backend.epsilon()
        y_true=tf.cast(y_true, tf.int32)
        y_oh=tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        p=tf.clip_by_value(y_pred, eps, 1.0)
        pt=tf.reduce_sum(y_oh*p, axis=-1)
        a=tf.gather(self.alpha, y_true) if self.alpha is not None else 1.0
        loss=- a * tf.pow(1.0-pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)

# ============================================================
# Bloques básicos
# ============================================================
def get_activation(name:str):
    name=name.lower()
    if name=="gelu": return tf.keras.activations.gelu
    if name=="swish": return tf.keras.activations.swish
    return "relu"

class TNet(layers.Layer):
    def __init__(self, K=3, act="relu"):
        super().__init__()
        A=get_activation(act)
        self.K=K
        self.conv1=layers.Conv1D(64,1,activation=A)
        self.conv2=layers.Conv1D(128,1,activation=A)
        self.conv3=layers.Conv1D(1024,1,activation=A)
        self.fc1=layers.Dense(512,activation=A)
        self.fc2=layers.Dense(256,activation=A)
        self.fc3=layers.Dense(K*K)
        self.reshape=layers.Reshape((K,K))
    def call(self, x, training=False):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x)
        x=layers.GlobalMaxPooling1D()(x)
        x=self.fc1(x); x=self.fc2(x); x=self.fc3(x); x=self.reshape(x)
        bs=tf.shape(x)[0]; I=tf.eye(self.K, batch_shape=[bs])
        return x+I

# kNN agrupamiento simple (distancias euclidianas)
def knn_group(x, k:int):
    # x: (B,P,C) -> devuelve índices (B,P,k)
    # cuidado con O(P^2); para P~8k el tiempo es aceptable en GPU
    B=tf.shape(x)[0]; P=tf.shape(x)[1]
    xx=tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)  # (B,P,1)
    dist=xx-2*tf.matmul(x, x, transpose_b=True)+tf.transpose(xx, perm=[0,2,1])  # (B,P,P)
    vals, idx=tf.math.top_k(-dist, k=k) # vecinos más cercanos
    return idx

def gather_idx(x, idx):
    # x:(B,P,C), idx:(B,P,k) -> (B,P,k,C)
    B=tf.shape(x)[0]; P=tf.shape(x)[1]; k=tf.shape(idx)[2]; C=tf.shape(x)[2]
    # expand idx to (B,P,k,C)
    idx4=tf.reshape(idx, [B, P, k, 1])
    idx4=tf.tile(idx4, [1,1,1,C])
    x_expand=tf.tile(tf.reshape(x,[B,1,P,C]), [1,P,1,1])  # (B,P,P,C)
    return tf.gather(x_expand, idx, batch_dims=1)         # (B,P,k,C)

def mlp_block(x, channels: List[int], act="relu", bn=True, name=None, dilation_rate=1):
    A=get_activation(act)
    for i,c in enumerate(channels):
        x=layers.Conv1D(c, 1, activation=A, dilation_rate=dilation_rate, name=None if name is None else f"{name}_conv{i}")(x)
        if bn: x=layers.BatchNormalization()(x)
    return x

# ============================================================
# Modelos
# ============================================================
def build_pointnet(num_classes:int, base=64, dropout=0.5, act="relu"):
    A=get_activation(act)
    inp=layers.Input(shape=(None,3))
    t1=TNet(3, act)(inp)
    x=tf.matmul(inp, t1)
    x=layers.Conv1D(base,1,activation=A)(x)
    x=layers.Conv1D(base,1,activation=A)(x)
    t2=TNet(base, act)(x)
    x=tf.matmul(x, t2)
    x=layers.Conv1D(base,1,activation=A)(x)
    x=layers.Conv1D(2*base,1,activation=A)(x)
    x_local=layers.Conv1D(16*base,1,activation=A)(x)
    g=layers.GlobalMaxPooling1D()(x_local)
    g=tf.expand_dims(g,1); g=tf.tile(g,[1,tf.shape(inp)[1],1])
    x=tf.concat([x_local,g], axis=-1)
    x=layers.Conv1D(8*base,1,activation=A)(x); x=layers.Dropout(dropout)(x)
    x=layers.Conv1D(4*base,1,activation=A)(x); x=layers.Dropout(dropout)(x)
    x=layers.Conv1D(2*base,1,activation=A)(x)
    out=layers.Conv1D(num_classes,1,activation='softmax')(x)
    return keras.Model(inp, out, name="pointnet")

def pointnetpp_set_abstraction(x, k:int, out_ch:int, act="relu", name=None, dilation=1):
    # vecino + max-pool de características (PointNet local)
    idx=knn_group(x, k)
    neigh=gather_idx(x, idx)        # (B,P,k,C)
    # sub-mlp sobre vecinos
    B=tf.shape(x)[0]; P=tf.shape(x)[1]; kN=tf.shape(neigh)[2]
    feat=tf.reshape(neigh, [B, P*kN, tf.shape(neigh)[-1]])
    feat=mlp_block(feat, [out_ch, out_ch], act=act, bn=True, name=name, dilation_rate=dilation)
    feat=tf.reshape(feat, [B, P, kN, out_ch])
    feat=tf.reduce_max(feat, axis=2) # (B,P,out_ch)
    return feat

def build_pointnetpp(num_classes:int, base=64, k=16, levels=3, act="relu", dropout=0.5):
    inp=layers.Input(shape=(None,3))
    x=inp
    # primera elevación de canales
    x=mlp_block(x, [base, base], act=act, bn=True, name="sa0")
    # niveles jerárquicos (sin muestreo espacial duro para mantener forma [B,P,*])
    for i in range(levels):
        x=pointnetpp_set_abstraction(x, k=k, out_ch=base*(2**i), act=act, name=f"sa{i+1}")
    xg=layers.GlobalMaxPooling1D()(x)
    xg=tf.expand_dims(xg,1); xg=tf.tile(xg,[1,tf.shape(inp)[1],1])
    x=tf.concat([x, xg], axis=-1)
    x=mlp_block(x, [4*base, 2*base], act=act, bn=True, name="head1")
    x=layers.Dropout(dropout)(x)
    out=layers.Conv1D(num_classes,1,activation='softmax')(x)
    return keras.Model(inp, out, name="pointnetpp")

def build_pointnetpp_dilated(num_classes:int, base=64, k=16, levels=3, act="relu", dropout=0.5):
    inp=layers.Input(shape=(None,3))
    x=mlp_block(inp, [base, base], act=act, bn=True, name="dsa0")
    for i in range(levels):
        dil = 1 + i  # “dilatación” progresiva mediante dilation_rate en 1D conv
        x=pointnetpp_set_abstraction(x, k=k, out_ch=base*(2**i), act=act, name=f"dsa{i+1}")
        x=mlp_block(x, [base*(2**i)], act=act, bn=True, name=f"dsa{i+1}_mlp", dilation_rate=dil)
    xg=layers.GlobalMaxPooling1D()(x)
    xg=tf.expand_dims(xg,1); xg=tf.tile(xg,[1,tf.shape(inp)[1],1])
    x=tf.concat([x,xg], axis=-1)
    x=mlp_block(x, [4*base, 2*base], act=act, bn=True, name="dhead1")
    x=layers.Dropout(dropout)(x)
    out=layers.Conv1D(num_classes,1,activation='softmax')(x)
    return keras.Model(inp, out, name="pointnetpp_dilated")

# ---------- Point Region Transformer (ligero) ----------
class MHSA(layers.Layer):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads=num_heads
        self.scale=(dim//num_heads) ** -0.5
        self.q=layers.Dense(dim); self.k=layers.Dense(dim); self.v=layers.Dense(dim)
        self.proj=layers.Dense(dim); self.dp=layers.Dropout(dropout)
    def call(self, x):
        B=tf.shape(x)[0]; P=tf.shape(x)[1]; C=tf.shape(x)[2]
        q=self.q(x); k=self.k(x); v=self.v(x)
        q=tf.reshape(q,[B,P,self.num_heads,C//self.num_heads])
        k=tf.reshape(k,[B,P,self.num_heads,C//self.num_heads])
        v=tf.reshape(v,[B,P,self.num_heads,C//self.num_heads])
        att=tf.einsum('bphd,bqhd->bpqh', q*self.scale, k)
        att=tf.nn.softmax(att, axis=2)
        out=tf.einsum('bpqh,bqhd->bphd', att, v)
        out=tf.reshape(out,[B,P,C])
        return self.proj(out)

def transformer_block(x, dim, num_heads=4, mlp_ratio=2.0, act="relu", drop=0.0):
    A=get_activation(act)
    h=layers.LayerNormalization()(x)
    h=MHSA(dim, num_heads=num_heads, dropout=drop)(h)
    x=layers.Add()([x,h])
    h=layers.LayerNormalization()(x)
    h=layers.Dense(int(dim*mlp_ratio), activation=A)(h)
    h=layers.Dense(dim)(h)
    x=layers.Add()([x,h])
    return x

def build_point_region_transformer(num_classes:int, base=64, depth=4, heads=4, act="relu", dropout=0.1):
    inp=layers.Input(shape=(None,3))
    # embedding geométrico inicial
    x=layers.Conv1D(base,1,activation=get_activation(act))(inp)
    # varios bloques transformer (self-attention global; para “regional”, se recomienda usar kNN pre-agrupado,
    # pero mantenemos forma simple y eficiente)
    for i in range(depth):
        x=transformer_block(x, dim=base, num_heads=heads, mlp_ratio=2.0, act=act, drop=dropout)
    xg=layers.GlobalMaxPooling1D()(x)
    xg=tf.expand_dims(xg,1); xg=tf.tile(xg,[1,tf.shape(inp)[1],1])
    x=tf.concat([x,xg], axis=-1)
    x=layers.Conv1D(2*base,1,activation=get_activation(act))(x)
    x=layers.Dropout(dropout)(x)
    out=layers.Conv1D(num_classes,1,activation='softmax')(x)
    return keras.Model(inp, out, name="point_region_transformer")

# ============================================================
# Compilación + callbacks
# ============================================================
def compile_model(m, args, num_classes:int, steps_per_epoch:int):
    # scheduler
    lr_sched = make_scheduler(args.scheduler, args.lr, steps_per_epoch, args.epochs)
    lr = args.lr if (lr_sched is None or args.scheduler=="plateau") else lr_sched
    opt = make_optimizer(args.optimizer, lr if not isinstance(lr, float) else lr, args.weight_decay, args.momentum)

    metrics = ['accuracy']
    if args.metrics_macro:
        metrics += [PrecisionMacro(num_classes, "prec_macro"),
                    RecallMacro(num_classes, "rec_macro"),
                    F1Macro(num_classes, "f1_macro"),
                    SparseMeanIoU(num_classes, "miou")]
    # pérdida
    if args.focal:
        loss = FocalSparseCE(gamma=2.0,
                             alpha_vec=json.load(open(args.class_weights_json)) if args.class_weights_json else None,
                             num_classes=num_classes)
    elif args.class_weights_json:
        cw = json.load(open(args.class_weights_json))
        loss = WeightedSparseCE(cw, num_classes)
    elif args.infer_class_weights:
        loss = None  # ponderamos vía sample_weight (simple) – aquí dejamos CE estándar
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()

    m.compile(optimizer=opt, loss=loss or keras.losses.SparseCategoricalCrossentropy(),
              metrics=metrics)
    return m

def make_callbacks(run_dir: Path, args):
    cbs = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.es_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(run_dir/"best"), monitor='val_loss', save_best_only=True, save_format='tf', verbose=1),
        keras.callbacks.CSVLogger(str(run_dir/"train_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(run_dir/"tb")),
    ]
    if args.scheduler.lower()=="plateau":
        cbs.insert(0, keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                                        patience=args.lr_patience, min_lr=args.min_lr, verbose=1))
    return cbs

def save_summary(model, run_dir: Path):
    txt=[]; model.summary(print_fn=lambda s: txt.append(s))
    (run_dir/"model_summary.txt").write_text("\n".join(txt), encoding="utf-8")

# ============================================================
# Entrenamiento de un modelo
# ============================================================
def train_one(model_name:str, args, ds_tr, ds_va, ds_te, info, run_base: Path):
    steps = int(np.ceil(len(list(ds_tr.as_numpy_iterator()))))  # robusto pero costoso
    # atajo: si el iter anterior consume; recreamos datasets para entrenamiento real
    ds_tr, ds_va, ds_te, info = make_datasets(args.data_path, args.batch_size, args.seed,
                                              args.augment, args.rot_deg, args.jitter_std,
                                              args.scale_low, args.scale_high)
    num_classes=info["num_classes"]

    # construir modelo
    if model_name=="pointnet" or model_name=="pointnet_plateau":
        model = build_pointnet(num_classes, base=args.base_channels, dropout=args.dropout, act=args.activation)
    elif model_name=="pointnetpp":
        model = build_pointnetpp(num_classes, base=args.base_channels, k=args.knn_k, levels=args.levels,
                                 act=args.activation, dropout=args.dropout)
    elif model_name=="pointnetpp_dilated":
        model = build_pointnetpp_dilated(num_classes, base=args.base_channels, k=args.knn_k, levels=args.levels,
                                         act=args.activation, dropout=args.dropout)
    elif model_name=="point_region_transformer":
        model = build_point_region_transformer(num_classes, base=args.base_channels,
                                               depth=args.tr_depth, heads=args.tr_heads,
                                               act=args.activation, dropout=args.dropout)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    run_dir = run_base/model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_summary(model, run_dir)
    (run_dir/"config.json").write_text(json.dumps({"args": vars(args), "model": model_name}, indent=2), encoding="utf-8")

    # compilar
    steps_per_epoch = max(1, int(np.ceil(tf.data.experimental.cardinality(ds_tr).numpy())))
    compiled = compile_model(model, args, num_classes, steps_per_epoch)
    cbs = make_callbacks(run_dir, args)

    # smoke
    if args.smoke:
        ds_tr = ds_tr.take(args.smoke_batches)
        ds_va = ds_va.take(max(1, args.smoke_batches//2))

    # fit
    history = compiled.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, verbose=1, callbacks=cbs)
    (run_dir/"history.json").write_text(json.dumps({k:[float(x) for x in v] for k,v in history.history.items()}, indent=2), encoding="utf-8")

    # eval
    test_metrics = compiled.evaluate(ds_te, verbose=1, return_dict=True)
    (run_dir/"test_metrics.json").write_text(json.dumps({k: float(v) for k,v in test_metrics.items()}, indent=2), encoding="utf-8")

    # guardar final
    (run_dir/"final").mkdir(exist_ok=True, parents=True)
    compiled.save(run_dir/"final", save_format='tf')
    print(f"[OK] {model_name} guardado en {run_dir/'final'}")

# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)

    # entrenamiento
    ap.add_argument("--model", default="all",
                    help="pointnet | pointnet_plateau | pointnetpp | pointnetpp_dilated | point_region_transformer | all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # arquitectura / hiper
    ap.add_argument("--activation", default="relu")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--levels", type=int, default=3, help="Niveles SA para PointNet++")
    ap.add_argument("--knn_k", type=int, default=16)
    ap.add_argument("--tr_depth", type=int, default=4)
    ap.add_argument("--tr_heads", type=int, default=4)

    # optimizers / schedulers
    ap.add_argument("--optimizer", default="adam", choices=["adam","adamw","sgd","rmsprop"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--scheduler", default="plateau", choices=["none","plateau","cosine","exponential"])
    ap.add_argument("--lr_patience", type=int, default=5)
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--es_patience", type=int, default=10)

    # augment
    ap.add_argument("--augment", dest="augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--rot_deg", type=float, default=10.0)
    ap.add_argument("--jitter_std", type=float, default=0.005)
    ap.add_argument("--scale_low", type=float, default=0.90)
    ap.add_argument("--scale_high", type=float, default=1.10)

    # pérdidas
    ap.add_argument("--metrics_macro", action="store_true")
    ap.add_argument("--class_weights_json", type=str, default=None)
    ap.add_argument("--infer_class_weights", action="store_true")
    ap.add_argument("--focal", action="store_true")

    # dispositivos
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--multi_gpu", action="store_true")

    # smoke
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_batches", type=int, default=2)

    args = ap.parse_args()
    set_tf_seed(args.seed)
    strategy = setup_devices(args.cuda, args.multi_gpu)

    # datasets
    ds_tr, ds_va, ds_te, info = make_datasets(args.data_path, args.batch_size, args.seed,
                                              args.augment, args.rot_deg, args.jitter_std,
                                              args.scale_low, args.scale_high)

    out_base = Path(args.out_dir)/args.tag
    out_base.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model!="all" else [
        "pointnet", "pointnet_plateau", "pointnetpp", "pointnetpp_dilated", "point_region_transformer"
    ]

    if strategy:
        with strategy.scope():
            for m in models:
                print(f"\n===== Entrenando {m} =====")
                train_one(m, args, ds_tr, ds_va, ds_te, info, out_base)
    else:
        for m in models:
            print(f"\n===== Entrenando {m} =====")
            train_one(m, args, ds_tr, ds_va, ds_te, info, out_base)

if __name__ == "__main__":
    main()
