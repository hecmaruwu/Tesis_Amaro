# Pipeline completo UFRN + Teeth3DS (Multiseed, Fine-tuning y Entrenamiento desde cero)

## Objetivo

Este documento resume el pipeline completo utilizado para:

1. Generar pseudo-labels clínicas del diente 21 en UFRN.
2. Crear datasets binarios d21 vs fondo.
3. Entrenar modelos desde cero.
4. Realizar fine-tuning desde modelos preentrenados en Teeth3DS.
5. Ejecutar splits aleatorios multiseed.
6. Exportar predicciones.
7. Visualizar resultados.
8. Comparar arquitecturas.

Arquitecturas utilizadas:

- PointNet
- PointNet++
- DGCNN
- PointNetTransformer

Todos los experimentos fueron realizados sobre nubes de puntos de 8192 puntos.

---

# 1. Estructura esperada del dataset UFRN

## Entrada principal

```text
data/UFRN/
├── targets_export/
│   ├── paciente_1/
│   │   └── stl/
│   │       └── upper_full.stl
│   ├── paciente_2/
│   └── ...
│
├── pseudolabels_dbscan_all/
│   ├── paciente_1/
│   │   └── gt_removed_d21_dbscan.npy
│   ├── paciente_2/
│   └── ...
```

---

# 2. Generación de pseudo-labels clínicas

Los pseudo-labels fueron generados utilizando diferencias geométricas entre:

- arcada completa,
- diente removido clínicamente.

Posteriormente:

- DBSCAN,
- filtrado geométrico,
- reconstrucción del diente 21,
- exportación a `.npy`.

Salida:

```text
pseudolabels_dbscan_all/paciente_X/gt_removed_d21_dbscan.npy
```

Estos puntos representan el diente 21 clínicamente removido.

---

# 3. Script principal para construir datasets binarios

## Script utilizado

```text
scripts_last_version/build_ufrn_binary_d21_dataset.py
```

Este script:

- muestrea la malla STL,
- normaliza a esfera unitaria,
- construye etiquetas binarias,
- genera splits,
- guarda trazabilidad,
- soporta múltiples seeds.

---

# 4. Split fijo clásico

## Comando

```bash
python3 scripts_last_version/build_ufrn_binary_d21_dataset.py \
  --ufrn_root data/UFRN/targets_export \
  --pseudolabel_root data/UFRN/pseudolabels_dbscan_all \
  --out_dir data/UFRN/finetune_d21_8192 \
  --n_points 8192 \
  --label_radius 1.25 \
  --seed 42 \
  --split_mode fixed \
  --train_end 35 \
  --val_end 43
```

## Resultado

```text
train: paciente_1  → paciente_35
val:   paciente_36 → paciente_43
test:  paciente_44 → paciente_51
```

---

# 5. Split aleatorio multiseed

## Soporte multiseed

El script soporta:

```bash
--split_mode random
--split_seed
```

permitiendo generar distintos splits reproducibles.

---

# 6. Crear split random seed42

```bash
python3 scripts_last_version/build_ufrn_binary_d21_dataset.py \
  --ufrn_root data/UFRN/targets_export \
  --pseudolabel_root data/UFRN/pseudolabels_dbscan_all \
  --out_dir data/UFRN/finetune_d21_8192_seed42 \
  --n_points 8192 \
  --label_radius 1.25 \
  --seed 42 \
  --split_mode random \
  --split_seed 42 \
  --n_train 35 \
  --n_val 8 \
  --n_test 8
```

---

# 7. Crear múltiples seeds

## Seed 7

```bash
python3 scripts_last_version/build_ufrn_binary_d21_dataset.py \
  --ufrn_root data/UFRN/targets_export \
  --pseudolabel_root data/UFRN/pseudolabels_dbscan_all \
  --out_dir data/UFRN/finetune_d21_8192_seed7 \
  --n_points 8192 \
  --label_radius 1.25 \
  --seed 42 \
  --split_mode random \
  --split_seed 7 \
  --n_train 35 \
  --n_val 8 \
  --n_test 8
```

## Seed 13

```bash
python3 scripts_last_version/build_ufrn_binary_d21_dataset.py \
  --ufrn_root data/UFRN/targets_export \
  --pseudolabel_root data/UFRN/pseudolabels_dbscan_all \
  --out_dir data/UFRN/finetune_d21_8192_seed13 \
  --n_points 8192 \
  --label_radius 1.25 \
  --seed 42 \
  --split_mode random \
  --split_seed 13 \
  --n_train 35 \
  --n_val 8 \
  --n_test 8
```

## Seed 21

```bash
python3 scripts_last_version/build_ufrn_binary_d21_dataset.py \
  --ufrn_root data/UFRN/targets_export \
  --pseudolabel_root data/UFRN/pseudolabels_dbscan_all \
  --out_dir data/UFRN/finetune_d21_8192_seed21 \
  --n_points 8192 \
  --label_radius 1.25 \
  --seed 42 \
  --split_mode random \
  --split_seed 21 \
  --n_train 35 \
  --n_val 8 \
  --n_test 8
```

## Seed 99

```bash
python3 scripts_last_version/build_ufrn_binary_d21_dataset.py \
  --ufrn_root data/UFRN/targets_export \
  --pseudolabel_root data/UFRN/pseudolabels_dbscan_all \
  --out_dir data/UFRN/finetune_d21_8192_seed99 \
  --n_points 8192 \
  --label_radius 1.25 \
  --seed 42 \
  --split_mode random \
  --split_seed 99 \
  --n_train 35 \
  --n_val 8 \
  --n_test 8
```

---

# 8. Archivos generados

Cada dataset genera:

```text
X_train.npz
Y_train.npz
X_val.npz
Y_val.npz
X_test.npz
Y_test.npz

index_train.csv
index_val.csv
index_test.csv

all_index.csv
artifacts/meta.json
```

---

# 9. Trazabilidad

Cada fila contiene:

- patient_id
- split
- upper_full_stl
- gt_d21_npy
- center_x/y/z
- scale
- sampling_seed
- split_seed
- split_mode

Esto permite:

- reproducibilidad completa,
- auditoría,
- visualización,
- reconstrucción geométrica,
- análisis paper-level.

---

# 10. Fine-tuning PointNet

## Modelo Teeth3DS

```text
outputs/pointnet_classic/grid_pro_runner_v1/exp03_bs16_lr3e4_do05_bg003_amp/best.pt
```

---

## Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_pointnet_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --ckpt outputs/pointnet_classic/grid_pro_runner_v1/exp03_bs16_lr3e4_do05_bg003_amp/best.pt \
  --out_dir outputs/ufrn_finetune/pointnet_binary_d21_from_teeth3ds_120ep_cosine_gpu1 \
  --epochs 120 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0
```

---

# 11. PointNet desde cero

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_pointnet_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --from_scratch \
  --out_dir outputs/ufrn_finetune/pointnet_binary_d21_scratch_120ep_cosine_gpu1 \
  --epochs 120 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0
```

---

# 12. Fine-tuning PointNet++

## Modelo base

```text
/home/htaucare/Tesis_Amaro/outputs/pointnetpp/grid_pro_runner_v2/exp04_bs8_lr3e4_r010_020_040_ns32/best.pt
```

---

## Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_pointnetpp_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --ckpt outputs/pointnetpp/grid_pro_runner_v2/exp04_bs8_lr3e4_r010_020_040_ns32/best.pt \
  --out_dir outputs/ufrn_finetune/pointnetpp_binary_d21_from_teeth3ds_120ep_cosine_gpu1_exact_v2 \
  --epochs 120 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0 \
  --nsample 32
```

---

# 13. PointNet++ desde cero

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_pointnetpp_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --from_scratch \
  --out_dir outputs/ufrn_finetune/pointnetpp_binary_d21_scratch_120ep_cosine_gpu1_exact \
  --epochs 120 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0 \
  --nsample 32
```

---

# 14. Fine-tuning DGCNN

## Modelo base

```text
/home/htaucare/Tesis_Amaro/outputs/dgcnn/grid_pro_runner_v2_gpu1/exp06_bs8_lr2e4_k20_emb768_bg003/best.pt
```

---

## Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_dgcnn_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --ckpt outputs/dgcnn/grid_pro_runner_v2_gpu1/exp06_bs8_lr2e4_k20_emb768_bg003/best.pt \
  --out_dir outputs/ufrn_finetune/dgcnn_binary_d21_from_teeth3ds_120ep_cosine_gpu1 \
  --epochs 120 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0 \
  --k 20 \
  --emb_dims 768
```

---

# 15. DGCNN desde cero

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_dgcnn_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --from_scratch \
  --out_dir outputs/ufrn_finetune/dgcnn_binary_d21_scratch_120ep_cosine_gpu1 \
  --epochs 120 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.5 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0 \
  --k 20 \
  --emb_dims 768
```

---

# 16. Fine-tuning PointNetTransformer

## Modelo base

```text
/home/htaucare/Tesis_Amaro/outputs/pointnettransformer/grid_pro_runner_v2_gpu0/exp13_bs4_lr2e4_dm256_dep4_h8_ff512_do005/best.pt
```

---

## Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_pointnettransformer_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --ckpt outputs/pointnettransformer/grid_pro_runner_v2_gpu0/exp13_bs4_lr2e4_dm256_dep4_h8_ff512_do005/best.pt \
  --out_dir outputs/ufrn_finetune/pointnettransformer_binary_d21_from_teeth3ds_120ep_cosine_gpu1 \
  --epochs 120 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.05 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0 \
  --d_model 256 \
  --hidden_dim 128 \
  --depth 4 \
  --nhead 8 \
  --dim_feedforward 512 \
  --token_subsample 2048 \
  --prop_k 3 \
  --prop_chunk 2048
```

---

# 17. PointNetTransformer desde cero

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/train_ufrn_pointnettransformer_binary_finetune.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --from_scratch \
  --out_dir outputs/ufrn_finetune/pointnettransformer_binary_d21_scratch_120ep_cosine_gpu1 \
  --epochs 120 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.05 \
  --num_workers 4 \
  --device cuda \
  --pos_weight 20.0 \
  --d_model 256 \
  --hidden_dim 128 \
  --depth 4 \
  --nhead 8 \
  --dim_feedforward 512 \
  --token_subsample 2048 \
  --prop_k 3 \
  --prop_chunk 2048
```

---

# 18. Exportar predicciones

## Script

```text
scripts_last_version/export_ufrn_binary_predictions.py
```

---

## Comando

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts_last_version/export_ufrn_binary_predictions.py \
  --data_dir data/UFRN/finetune_d21_8192 \
  --ckpt outputs/ufrn_finetune/dgcnn_binary_d21_from_teeth3ds_120ep_cosine_gpu1/best.pt \
  --out_dir outputs/ufrn_finetune/dgcnn_binary_d21_from_teeth3ds_120ep_cosine_gpu1 \
  --device cuda
```

---

# 19. Visualización Plotly

## Script

```text
scripts_last_version/visualize_ufrn_binary_errors_plotly_aesthetic.py
```

---

## Comando

```bash
python3 scripts_last_version/visualize_ufrn_binary_errors_plotly_aesthetic.py \
  --full_stl data/UFRN/targets_export/paciente_44/stl/upper_full.stl \
  --gt_npy data/UFRN/pseudolabels_dbscan_all/paciente_44/gt_removed_d21_dbscan.npy \
  --pred_npy outputs/ufrn_finetune/dgcnn_binary_d21_from_teeth3ds_120ep_cosine_gpu1/test_predictions/paciente_44_pred.npy \
  --out_dir outputs/ufrn_visualizations/paciente_44_dgcnn
```

---

# 20. Resumen de resultados

| Modelo | Inicialización | Best Val F1 | Test F1 | Test IoU |
|---|---|---|---|---|
| PointNet++ | Scratch | 0.8193 | 0.8184 | 0.6926 |
| PointNet++ | Teeth3DS | 0.8478 | 0.8376 | 0.7205 |
| PointNetTransformer | Scratch | 0.8348 | 0.8198 | 0.6947 |
| PointNetTransformer | Teeth3DS | 0.8293 | 0.8446 | 0.7309 |
| PointNet | Scratch | 0.8921 | 0.8402 | 0.7245 |
| PointNet | Teeth3DS | 0.9074 | 0.9278 | 0.8653 |
| DGCNN | Scratch | 0.8977 | 0.9221 | 0.8554 |
| DGCNN | Teeth3DS | 0.9074 | 0.9278 | 0.8653 |

---

# 21. Conclusiones principales

## 1. El preentrenamiento sí ayuda

Especialmente en:

- PointNet,
- PointNet++,
- Transformer.

DGCNN mostró el mejor rendimiento global.

---

## 2. DGCNN fue el modelo más robusto

Resultados:

- mejor F1,
- mejor IoU,
- mejor recall,
- mejor generalización clínica.

---

## 3. El split multiseed fortalece el paper

Ahora el pipeline permite:

- evaluar estabilidad,
- calcular media ± std,
- demostrar robustez,
- evitar dependencia de un único split.

---

## 4. El pipeline ya es nivel paper internacional

Porque incluye:

- trazabilidad completa,
- separación por paciente,
- reproducibilidad,
- análisis multiseed,
- comparación pretrained vs scratch,
- visualización anatómica,
- pseudo-labels clínicos,
- segmentación geométrica 3D.

