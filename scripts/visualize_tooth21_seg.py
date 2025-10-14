#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

def load_cloud(p): return np.load(p).astype(np.float32)

def normalize_unit_sphere(pts):
    c = pts.mean(axis=0, keepdims=True); pts = pts - c
    r = np.linalg.norm(pts, axis=1).max()
    return pts / (r if r > 0 else 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ufrn_root", required=True)
    ap.add_argument("--struct_rel", default="processed_struct")
    ap.add_argument("--patient_id", required=True)          # ej: paciente_19
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--co_modules", default="")
    ap.add_argument("--co_alias", default="")
    ap.add_argument("--co_patch", default="")
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--tooth_label_index", type=int, default=21,
                    help="Índice de clase del diente 21 en TU dataset (no FDI).")
    args = ap.parse_args()

    # ---- cargar modelo con o sin custom_objects
    custom = {}
    import importlib, inspect, types, sys
    def _inject_dummy(names=("matplotlib","matplotlib.pyplot")):
        for n in names:
            if n not in sys.modules:
                sys.modules[n] = types.ModuleType(n)
    def _collect(modname):
        _inject_dummy()
        m = importlib.import_module(modname)
        return {k:v for k,v in inspect.getmembers(m)
                if (inspect.isclass(v) or inspect.isfunction(v)) and not k.startswith("_")}
    if args.co_modules:
        for m in [x.strip() for x in args.co_modules.split(",") if x.strip()]:
            try: custom.update(_collect(m))
            except Exception as e: print("[WARN] no pude importar", m, e)
    if args.co_alias.strip():
        try:
            am = json.loads(args.co_alias)
            for alias, real in am.items():
                if real in custom: custom[alias] = custom[real]
                else:
                    # intento por coincidencia parcial
                    cands = [k for k in custom if real.lower() in k.lower()]
                    if len(cands)==1: custom[alias] = custom[cands[0]]
        except Exception as e:
            print("[WARN] co_alias inválido:", e)

    try:
        model = tf.keras.models.load_model(args.model_path, compile=False, custom_objects=custom)
    except Exception as e:
        print("[ERR] cargando modelo:", e); return

    root = Path(args.ufrn_root)/args.struct_rel
    up = load_cloud(root/"upper"/f"{args.patient_id}_full"/"point_cloud.npy")
    lo = load_cloud(root/"lower"/f"{args.patient_id}_full"/"point_cloud.npy")

    # entrada con contexto (mitad upper + mitad lower) para respetar shape del modelo
    nU = args.n_points//2; nL = args.n_points-nU
    def sample_n(pts, n):
        idx = np.random.choice(pts.shape[0], size=n, replace=(pts.shape[0]<n))
        return pts[idx]
    up_in = sample_n(up, nU); lo_in = sample_n(lo, nL)
    x = np.concatenate([up_in, lo_in], axis=0)
    x = normalize_unit_sphere(x)

    probs = model.predict(x[None,...], verbose=0)  # (1,P,C)
    preds = np.argmax(probs, axis=-1)[0]

    # sólo la parte upper (primeros nU)
    mask21 = (preds[:nU] == int(args.tooth_label_index))
    tooth21_pts = up_in[mask21]
    other_pts   = up_in[~mask21]

    print(f"[INFO] {args.patient_id}: puntos 21={tooth21_pts.shape[0]}, otros={other_pts.shape[0]}")

    if not _HAS_MPL:
        np.save(f"{args.patient_id}_tooth21.npy", tooth21_pts)
        np.save(f"{args.patient_id}_upper_others.npy", other_pts)
        print(f"[OUT] Guardado NPYs: {args.patient_id}_tooth21.npy / _upper_others.npy")
        return

    # plot simple
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(9,4), dpi=140)
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.scatter(other_pts[:,0], other_pts[:,1], other_pts[:,2], s=1, c="0.7")
    ax.set_title("Upper (otros)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    if tooth21_pts.size>0:
        ax2.scatter(tooth21_pts[:,0], tooth21_pts[:,1], tooth21_pts[:,2], s=2, c="tab:green")
    ax2.set_title("Predicción diente 21")
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    fig.tight_layout()
    fig.savefig(f"{args.patient_id}_seg_tooth21.png")
    plt.close(fig)
    print(f"[OUT] {args.patient_id}_seg_tooth21.png")

if __name__ == "__main__":
    main()
