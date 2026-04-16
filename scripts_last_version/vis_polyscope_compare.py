import polyscope as ps
import numpy as np
import torch
from pathlib import Path


#debo poner la ruta de los modelos/outputs con los mejores modelos guardados. En este caso, solo tengo el DGCNN, 
# pero se pueden agregar más modelos siguiendo el mismo formato. Luego, el script carga los datos de prueba, ejecuta 
# cada modelo para obtener las predicciones, y visualiza todo en Polyscope, incluyendo un análisis de error específico para la clase D21. 
# =========================
# CONFIG
# =========================
DATA_DIR = "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2"

MODEL_PATHS = {
    "DGCNN": "/home/htaucare/Tesis_Amaro/outputs/dgcnn/gpu1_run_v9_patch/best.pt",
    # agrega más:
    # "PointNet": "...",
    # "PointNet++": "..."
}

DEVICE = "cuda"
SAMPLE_IDX = 0
D21 = 8


# =========================
# LOAD DATA
# =========================
X = np.load(Path(DATA_DIR) / "X_test.npz")["X"]
Y = np.load(Path(DATA_DIR) / "Y_test.npz")["Y"]

xyz = torch.tensor(X[SAMPLE_IDX], dtype=torch.float32).unsqueeze(0).to(DEVICE)
gt = Y[SAMPLE_IDX]


# =========================
# LOAD MODEL (DGCNN)
# =========================
from train_dgcnn_classic_only_fixed_v9_patch import DGCNN_Seg

def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)

    model = DGCNN_Seg(
        num_classes=ckpt["num_classes"],
        k=20,
        emb_dims=1024,
        dropout=0.5,
        knn_chunk_size=1024
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# =========================
# RUN MODELS
# =========================
predictions = {}

for name, path in MODEL_PATHS.items():
    model = load_model(path)

    with torch.no_grad():
        logits = model(xyz)
        pred = logits.argmax(-1)[0].cpu().numpy()

    predictions[name] = pred


# =========================
# POLYSCOPE
# =========================
ps.init()

points = xyz[0].cpu().numpy()

# --- GT ---
pc_gt = ps.register_point_cloud("GT", points)
pc_gt.add_scalar_quantity("labels", gt, enabled=True)

# --- MODELOS ---
for name, pred in predictions.items():
    pc = ps.register_point_cloud(name, points)
    pc.add_scalar_quantity("pred", pred, enabled=True)

# --- ERROR D21 ---
for name, pred in predictions.items():
    gt_pos = (gt == D21)
    pred_pos = (pred == D21)

    error = np.zeros_like(gt)

    error[(gt_pos & pred_pos)] = 1  # TP
    error[(~gt_pos & pred_pos)] = 2 # FP
    error[(gt_pos & ~pred_pos)] = 3 # FN

    pc = ps.register_point_cloud(f"{name}_d21_error", points)
    pc.add_scalar_quantity("error", error, enabled=True)

ps.show()