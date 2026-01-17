#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-like augmentation and split creation (v3):
Compatible with PointNet / PointNet++ preprocessing (FPS + normalization).
Fixes cap_bg semantics (fraction vs absolute), exact-N subsampling, RNG consistency,
and ensure_coverage safety.
"""

import json
import argparse
from pathlib import Path
import numpy as np

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


# -------------------------------
# Utils
# -------------------------------

def np_rng(seed: int):
    return np.random.default_rng(int(seed))

def json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def as_jsonable_labelmap(id2idx: dict, idx2id: dict):
    return {
        "id2idx": {str(k): int(v) for k, v in id2idx.items()},
        "idx2id": {str(k): int(v) for k, v in idx2id.items()},
    }


# -------------------------------
# Geometric helpers
# -------------------------------

def rotate_z(points: np.ndarray, max_deg=15.0, rng=None) -> np.ndarray:
    theta = float(rng.uniform(-max_deg, max_deg)) * (np.pi / 180.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return (points @ R).astype(np.float32)

def jitter(points: np.ndarray, sigma=0.005, clip=0.02, rng=None) -> np.ndarray:
    # ✅ clip al ruido, no a las coordenadas
    noise = rng.normal(0.0, sigma, points.shape).astype(np.float32)
    noise = np.clip(noise, -clip, clip)
    return (points + noise).astype(np.float32)

def scale(points: np.ndarray, min_s=0.95, max_s=1.05, rng=None) -> np.ndarray:
    s = float(rng.uniform(min_s, max_s))
    return (points * s).astype(np.float32)

def dropout_points(points: np.ndarray, drop_rate=0.05, min_keep=32, rng=None) -> np.ndarray:
    if drop_rate <= 0.0:
        return points
    n = points.shape[0]
    if n <= min_keep:
        return points
    mask = rng.random(n) > drop_rate
    if mask.sum() < min_keep:
        keep_idx = rng.choice(n, min_keep, replace=False)
        mask[:] = False
        mask[keep_idx] = True
    return points[mask]

def augment_once(points: np.ndarray,
                 rotate_deg=15,
                 jitter_sigma=0.005, jitter_clip=0.02,
                 scale_min=0.95, scale_max=1.05,
                 dropout_rate=0.05,
                 N_target=8192,
                 rng=None) -> np.ndarray:
    """Augmentation: rotZ + jitter + scale + dropout + resample exacto a N."""
    x = rotate_z(points, max_deg=rotate_deg, rng=rng)
    x = jitter(x, sigma=jitter_sigma, clip=jitter_clip, rng=rng)
    x = scale(x, min_s=scale_min, max_s=scale_max, rng=rng)
    x = dropout_points(x, drop_rate=dropout_rate, rng=rng)

    n = x.shape[0]
    if n == 0:
        return np.zeros((N_target, 3), dtype=np.float32)

    idx = rng.choice(n, N_target, replace=(n < N_target))
    return x[idx].astype(np.float32)


# -------------------------------
# Labels
# -------------------------------

def build_global_label_map(Y_list):
    vals = set()
    for Y in Y_list:
        vals.update(np.unique(Y).tolist())
    vals = sorted(int(v) for v in vals)
    id2idx = {int(v): i for i, v in enumerate(vals)}
    idx2id = {i: int(v) for i, v in enumerate(vals)}
    return id2idx, idx2id

def apply_remap(Y: np.ndarray, id2idx: dict) -> np.ndarray:
    # vectorize seguro
    vect = np.vectorize(lambda z: id2idx.get(int(z), 0))
    return vect(Y).astype(np.int32)


# -------------------------------
# Subsampling (exact-N)
# -------------------------------

def subsample_with_cap(points, labels, N, cap_bg_frac=None, bg_id=0, rng=None):
    """
    Retorna EXACTAMENTE N puntos.
    cap_bg_frac: fracción [0,1] del total N reservada para background.
    """
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32).ravel()

    n = points.shape[0]
    if n == 0:
        return np.zeros((N, 3), np.float32), np.full((N,), bg_id, np.int32)

    idx_all = np.arange(n, dtype=np.int64)

    # sin cap -> sampleo uniforme a N
    if cap_bg_frac is None:
        sel = rng.choice(idx_all, N, replace=(n < N))
        return points[sel], labels[sel]

    # con cap
    cap_bg_frac = float(cap_bg_frac)
    cap_bg_frac = max(0.0, min(1.0, cap_bg_frac))

    idx_bg = idx_all[labels == bg_id]
    idx_fg = idx_all[labels != bg_id]

    max_bg = int(round(cap_bg_frac * N))
    take_bg = min(max_bg, idx_bg.size)

    # Siempre completar hasta N
    rem = N - take_bg
    sel_bg = rng.choice(idx_bg, take_bg, replace=False) if take_bg > 0 else np.empty((0,), np.int64)

    # Si no hay fg, se rellena con bg (con reemplazo si falta)
    if idx_fg.size > 0:
        sel_fg = rng.choice(idx_fg, rem, replace=(idx_fg.size < rem))
    else:
        sel_fg = rng.choice(idx_bg if idx_bg.size > 0 else idx_all, rem, replace=True)

    sel = np.concatenate([sel_bg, sel_fg], axis=0)
    # seguridad: puede pasar por redondeo, dejamos exacto N
    if sel.shape[0] != N:
        sel = rng.choice(sel, N, replace=(sel.shape[0] < N))
    return points[sel], labels[sel]


# -------------------------------
# Ensure coverage
# -------------------------------

def ensure_coverage_min_moves(splits, rng):
    global_classes = np.unique(np.concatenate([
        np.unique(splits["train"][1]),
        np.unique(splits["val"][1]),
        np.unique(splits["test"][1]),
    ]))
    moves = {"val": 0, "test": 0}
    for target in ["val", "test"]:
        Xt, Yt = splits[target]
        missing = [c for c in global_classes if c not in np.unique(Yt)]
        if not missing:
            continue
        for cls in missing:
            Xd, Yd = splits["train"]
            cand = np.where(np.any(Yd == cls, axis=1))[0]
            if cand.size == 0:
                continue
            pick = int(rng.choice(cand))
            Xt = np.concatenate([Xt, Xd[pick:pick+1]], 0)
            Yt = np.concatenate([Yt, Yd[pick:pick+1]], 0)
            Xd = np.delete(Xd, pick, axis=0)
            Yd = np.delete(Yd, pick, axis=0)
            splits["train"] = (Xd, Yd)
            splits[target] = (Xt, Yt)
            moves[target] += 1
    return moves


# -------------------------------
# Class weights
# -------------------------------

def compute_class_weights_from_train(Y_train, num_classes):
    flat = Y_train.ravel()
    hist = np.bincount(flat, minlength=num_classes).astype(np.float64)
    w = 1.0 / (hist + 1e-8)
    w = w * (num_classes / (w.sum() + 1e-8))
    return w.astype(np.float32), hist.astype(np.int64)


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=8192)

    # cap_bg puede ser fracción [0,1] o absoluto (ej 200000) -> se convierte a fracción cap_bg/N
    ap.add_argument("--cap_bg", type=float, default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_augmentation", action="store_true")
    ap.add_argument("--augment_times", type=int, default=1)
    ap.add_argument("--rotate_deg", type=float, default=15.0)
    ap.add_argument("--jitter_sigma", type=float, default=0.005)
    ap.add_argument("--jitter_clip", type=float, default=0.02)
    ap.add_argument("--scale_min", type=float, default=0.95)
    ap.add_argument("--scale_max", type=float, default=1.05)
    ap.add_argument("--dropout_rate", type=float, default=0.05)
    ap.add_argument("--ensure_coverage", action="store_true")
    args = ap.parse_args()

    rng = np_rng(args.seed)
    ds = Path(args.dataset_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)

    # cap_bg semantics
    cap_bg_frac = None
    if args.cap_bg is not None:
        if args.cap_bg <= 1.0:
            cap_bg_frac = float(args.cap_bg)
        else:
            # absoluto -> fracción sobre N
            cap_bg_frac = float(args.cap_bg) / float(args.N)
        cap_bg_frac = max(0.0, min(1.0, cap_bg_frac))

    # Load splits
    def load_split(name):
        X = np.load(ds / f"X_{name}.npz")["X"]
        Y = np.load(ds / f"Y_{name}.npz")["Y"]
        print(f"[OK] {name}: {X.shape}")
        return X, Y

    Xtr, Ytr = load_split("train")
    Xva, Yva = load_split("val")
    Xte, Yte = load_split("test")

    id2idx, idx2id = build_global_label_map([Ytr, Yva, Yte])
    Ytr = apply_remap(Ytr, id2idx)
    Yva = apply_remap(Yva, id2idx)
    Yte = apply_remap(Yte, id2idx)
    num_classes = len(id2idx)
    print(f"[MAP] Global labels: {num_classes} clases (0..{num_classes-1})")

    def process_array(X, Y, split_name, do_aug=False):
        M = X.shape[0]
        Xo = np.empty((M, args.N, 3), dtype=np.float32)
        Yo = np.empty((M, args.N), dtype=np.int32)
        it = tqdm(range(M), desc=f"Procesando {split_name}", ncols=80) if TQDM else range(M)
        for i in it:
            pts, lbs = subsample_with_cap(X[i], Y[i], args.N, cap_bg_frac=cap_bg_frac, bg_id=0, rng=rng)
            # ✅ seguridad exacta
            if pts.shape[0] != args.N:
                idx = rng.choice(pts.shape[0], args.N, replace=(pts.shape[0] < args.N))
                pts = pts[idx]
                lbs = lbs[idx]
            Xo[i], Yo[i] = pts, lbs

        if do_aug and args.use_augmentation and args.augment_times > 0:
            augX, augY = [], []
            it2 = tqdm(range(M), desc=f"AUG {split_name}", ncols=80) if TQDM else range(M)
            for i in it2:
                for _ in range(args.augment_times):
                    xa = augment_once(
                        Xo[i],
                        rotate_deg=args.rotate_deg,
                        jitter_sigma=args.jitter_sigma,
                        jitter_clip=args.jitter_clip,
                        scale_min=args.scale_min,
                        scale_max=args.scale_max,
                        dropout_rate=args.dropout_rate,
                        N_target=args.N,
                        rng=rng,
                    )
                    augX.append(xa)
                    augY.append(Yo[i])
            if augX:
                Xo = np.concatenate([Xo, np.stack(augX)], 0)
                Yo = np.concatenate([Yo, np.stack(augY)], 0)
                print(f"[AUG] {split_name}: +{len(augX)} muestras (x{args.augment_times})")
        return Xo, Yo

    Xtr_p, Ytr_p = process_array(Xtr, Ytr, "train", do_aug=True)
    Xva_p, Yva_p = process_array(Xva, Yva, "val", do_aug=False)
    Xte_p, Yte_p = process_array(Xte, Yte, "test", do_aug=False)
    splits = {"train": (Xtr_p, Ytr_p), "val": (Xva_p, Yva_p), "test": (Xte_p, Yte_p)}

    if args.ensure_coverage:
        moves = ensure_coverage_min_moves(splits, rng)
        print(f"[COVER] Movimientos realizados: val={moves['val']} test={moves['test']}")

    def save_npz(path, key, arr):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: arr})

    save_npz(out / "X_train.npz", "X", Xtr_p)
    save_npz(out / "Y_train.npz", "Y", Ytr_p)
    save_npz(out / "X_val.npz", "X", Xva_p)
    save_npz(out / "Y_val.npz", "Y", Yva_p)
    save_npz(out / "X_test.npz", "X", Xte_p)
    save_npz(out / "Y_test.npz", "Y", Yte_p)

    print(f"[SAVE] train: {Xtr_p.shape}")
    print(f"[SAVE] val: {Xva_p.shape}")
    print(f"[SAVE] test: {Xte_p.shape}")

    json_dump(as_jsonable_labelmap(id2idx, idx2id), out / "artifacts" / "label_map.json")

    cw, hist = compute_class_weights_from_train(Ytr_p, num_classes)
    json_dump({
        "class_weights": {str(i): float(cw[i]) for i in range(num_classes)},
        "hist_train": {str(i): int(hist[i]) for i in range(num_classes)}
    }, out / "artifacts" / "class_weights.json")

    json_dump({
        "N": args.N,
        "cap_bg_input": args.cap_bg,
        "cap_bg_frac_used": cap_bg_frac,
        "seed": args.seed,
        "use_augmentation": args.use_augmentation,
        "augment_times": args.augment_times,
        "ensure_coverage": args.ensure_coverage,
        "num_classes": num_classes,
    }, out / "artifacts" / "meta.json")

    print("\n[EDA] Validación por split:")
    for name, (X, Y) in splits.items():
        bg = ((Y == 0).sum() / Y.size) * 100.0
        print(f"  {name}: min={Y.min()} max={Y.max()} unique={len(np.unique(Y))} bg={bg:.2f}%")

    print(f"\n[DONE] Splits listos en: {out}")
    print(f"[META] Total clases: {num_classes}")

if __name__ == "__main__":
    main()
