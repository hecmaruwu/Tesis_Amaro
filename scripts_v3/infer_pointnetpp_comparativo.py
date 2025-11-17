#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualización comparativa PointNet++:
  🟩 Verde: predicción correcta
  🟥 Rojo: falso positivo (predijo 21 donde no lo es)
  🟦 Azul: falso negativo (no predijo 21 donde sí lo es)
  🟧 Naranja: otros dientes (ni GT ni predicho como 21)
"""
import sys, torch, numpy as np, pathlib, plotly.graph_objects as go
sys.path.append("/home/htaucare/Tesis_Amaro/scripts_v3")
from train_models_v10_paperlike import build_model, normalize_cloud

# === Config ===
data_dir = pathlib.Path("/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/stratified_8192_ep400_bs4_d03_v2")
ckpt_path = "/home/htaucare/Tesis_Amaro/runs_v10/pointnetpp_20251025_014857/checkpoints/best.pt"
out_dir = pathlib.Path("/home/htaucare/Tesis_Amaro/plots_inferencia_pointnetpp_comparativo")
out_dir.mkdir(parents=True, exist_ok=True)

sample_idx = 0
tooth_id = 21
num_classes = 26

# === Cargar datos ===
X = np.load(data_dir/"X_test.npz")["X"]
Y = np.load(data_dir/"Y_test.npz")["Y"]
pts = torch.tensor(X[sample_idx:sample_idx+1], dtype=torch.float32).cuda()
gt = Y[sample_idx]

# === Modelo ===
model = build_model("pointnetpp", num_classes).cuda()
state = torch.load(ckpt_path)
model.load_state_dict(state["model"])
model.eval()

with torch.no_grad():
    preds = model(normalize_cloud(pts)).argmax(-1).cpu().numpy()[0]

# === Clasificar tipo de punto ===
tp = (preds == tooth_id) & (gt == tooth_id)      # correcto (verde)
fp = (preds == tooth_id) & (gt != tooth_id)      # falso positivo (rojo)
fn = (preds != tooth_id) & (gt == tooth_id)      # falso negativo (azul)
others = ~(tp | fp | fn)                         # otros (naranja)

colors = np.zeros((len(gt), 3))
colors[tp] = [0, 1, 0]       # verde
colors[fp] = [1, 0, 0]       # rojo
colors[fn] = [0, 0, 1]       # azul
colors[others] = [1, 0.5, 0] # naranja

# === Plotly scatter ===
pts_np = X[sample_idx]
fig = go.Figure(
    data=[go.Scatter3d(
        x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2],
        mode="markers",
        marker=dict(size=3, color=[f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})" for r, g, b in colors]),
        opacity=0.9
    )]
)
fig.update_layout(scene=dict(aspectmode="data"), title=f"Comparativo – Diente {tooth_id}")

# === Guardar resultados ===
fig.write_html(str(out_dir / f"comparativo_sample{sample_idx}_d{tooth_id}.html"))
fig.write_image(str(out_dir / f"comparativo_sample{sample_idx}_d{tooth_id}.png"))
print(f"[OK] Guardado en: {out_dir}")
