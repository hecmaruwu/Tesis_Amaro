# README — Entrenamiento con múltiples semillas de split y resumen estadístico

## 1. Objetivo

Este documento resume cómo se entrenaron los modelos usando **distintas semillas de partición real del dataset** y cómo calcular el reporte estadístico de desempeño mediante media, desviación estándar, mínimo y máximo.

El objetivo de estos experimentos es evaluar la estabilidad de cada arquitectura frente a distintas particiones `train/val/test`, manteniendo fija la configuración del modelo y la semilla de entrenamiento.

---

## 2. Dataset correcto para estos experimentos

Los experimentos correctos de estabilidad por split usan los datasets:

```bash
/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed1_subseed42
/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed7_subseed42
/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed21_subseed42
/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed42_subseed42
/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed100_subseed42
```

Estos datasets fueron generados variando la semilla del split real del dataset base de 200k puntos, y manteniendo fija la semilla del submuestreo final en `42`.

La estructura conceptual es:

```text
merged_200000_splitseed{s}
→ upper_only_splitseed{s}_subseed42
→ entrenamiento con trainseed42
```

---

## 3. Diferencia importante con datasets antiguos

No confundir con:

```bash
upper_only_seed1
upper_only_seed7
upper_only_seed21
upper_only_seed42
upper_only_seed100
```

Esos datasets corresponden a variaciones del submuestreo/augmentación, pero no necesariamente a splits reales distintos.

Para el análisis robusto frente a distintas particiones, usar siempre:

```bash
upper_only_splitseed${s}_subseed42
```

---

## 4. Seeds utilizadas

```bash
1 7 21 42 100
```

En todos los modelos se mantiene:

```bash
--seed 42
```

para fijar la inicialización/entrenamiento y aislar el efecto del cambio de split del dataset.

---

# 5. Entrenamiento de modelos

---

## 5.1 PointNet Classic

### Carpeta de salida correcta

```bash
/home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config
```

### Comando

```bash
cd /home/htaucare/Tesis_Amaro

for s in 1 7 21 42 100; do

  DATA_DIR=/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed${s}_subseed42
  OUT_DIR=/home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config/splitseed${s}_trainseed42

  mkdir -p $OUT_DIR

  CUDA_VISIBLE_DEVICES=1 python3 -u /home/htaucare/Tesis_Amaro/scripts_last_version/pointnet_classic_final_v8_patch.py \
    --data_dir $DATA_DIR \
    --epochs 120 \
    --batch_size 16 \
    --lr 0.0003 \
    --weight_decay 0.0001 \
    --dropout 0.5 \
    --num_workers 6 \
    --infer_num_workers 0 \
    --device cuda \
    --use_amp \
    --grad_clip 1.0 \
    --d21_internal 8 \
    --bg_index 0 \
    --bg_weight 0.03 \
    --neighbor_teeth d11:1,d22:9 \
    --neighbor_eval_split both \
    --neighbor_every 1 \
    --seed 42 \
    --train_metrics_eval \
    --do_infer \
    --infer_split test \
    --infer_examples 20 \
    --index_csv ${DATA_DIR}/index_test.csv \
    --out_dir $OUT_DIR

done
```

---

## 5.2 PointNet++

### Carpeta de salida correcta

```bash
/home/htaucare/Tesis_Amaro/outputs/pointnetpp/stability_real_splits_v1/exp04_best_config
```

### Comando

```bash
cd /home/htaucare/Tesis_Amaro

for s in 1 7 21 42 100; do

  DATA_DIR=/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed${s}_subseed42
  OUT_DIR=/home/htaucare/Tesis_Amaro/outputs/pointnetpp/stability_real_splits_v1/exp04_best_config/splitseed${s}_trainseed42

  mkdir -p $OUT_DIR

  CUDA_VISIBLE_DEVICES=0 python3 -u /home/htaucare/Tesis_Amaro/scripts_last_version/pointnetpp_classic_final_v1_patch.py \
    --data_dir $DATA_DIR \
    --epochs 120 \
    --batch_size 8 \
    --lr 0.0003 \
    --weight_decay 0.0001 \
    --dropout 0.5 \
    --num_workers 6 \
    --infer_num_workers 0 \
    --device cuda \
    --grad_clip 1.0 \
    --d21_internal 8 \
    --bg_index 0 \
    --bg_weight 0.03 \
    --neighbor_teeth d11:1,d22:9 \
    --neighbor_eval_split both \
    --neighbor_every 1 \
    --seed 42 \
    --train_metrics_eval \
    --do_infer \
    --infer_split test \
    --infer_examples 20 \
    --index_csv ${DATA_DIR}/index_test.csv \
    --sa1_npoint 1024 \
    --sa1_radius 0.1 \
    --sa1_nsample 32 \
    --sa2_npoint 256 \
    --sa2_radius 0.2 \
    --sa2_nsample 32 \
    --sa3_npoint 64 \
    --sa3_radius 0.4 \
    --sa3_nsample 32 \
    --out_dir $OUT_DIR

done
```

---

## 5.3 DGCNN

### Carpeta recomendada para splits reales

```bash
/home/htaucare/Tesis_Amaro/outputs/dgcnn/dgcnn_real_splits_v1/exp06_best_config
```

### Comando recomendado

```bash
cd /home/htaucare/Tesis_Amaro

mkdir -p /home/htaucare/Tesis_Amaro/outputs/dgcnn/dgcnn_real_splits_v1/exp06_best_config

for s in 1 7 21 42 100; do

  DATA_DIR=/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed${s}_subseed42
  OUT_DIR=/home/htaucare/Tesis_Amaro/outputs/dgcnn/dgcnn_real_splits_v1/exp06_best_config/splitseed${s}_trainseed42

  mkdir -p $OUT_DIR

  CUDA_VISIBLE_DEVICES=1 python3 -u /home/htaucare/Tesis_Amaro/scripts_last_version/train_dgcnn_classic_only_fixed_v9_patch.py \
    --data_dir $DATA_DIR \
    --epochs 120 \
    --batch_size 8 \
    --lr 0.0002 \
    --weight_decay 0.0001 \
    --dropout 0.5 \
    --num_workers 6 \
    --infer_num_workers 0 \
    --device cuda \
    --grad_clip 1.0 \
    --k 20 \
    --emb_dims 768 \
    --knn_chunk_size 1024 \
    --bg_class 0 \
    --bg_weight 0.03 \
    --d21_internal 8 \
    --neighbor_teeth d11:1,d22:9 \
    --neighbor_eval_split both \
    --neighbor_every 1 \
    --seed 42 \
    --normalize \
    --train_metrics_eval \
    --do_infer \
    --infer_split test \
    --infer_examples 20 \
    --index_csv ${DATA_DIR}/index_test.csv \
    --out_dir $OUT_DIR

done
```

---

## 5.4 PointNet-Transformer

### Carpeta de salida correcta

```bash
/home/htaucare/Tesis_Amaro/outputs/pointnettransformer/pointnettransformer_real_splits_v1/exp13_best_config
```

### Comando

```bash
cd /home/htaucare/Tesis_Amaro

mkdir -p /home/htaucare/Tesis_Amaro/outputs/pointnettransformer/pointnettransformer_real_splits_v1/exp13_best_config

for s in 1 7 21 42 100; do

  DATA_DIR=/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_splitseed${s}_subseed42
  OUT_DIR=/home/htaucare/Tesis_Amaro/outputs/pointnettransformer/pointnettransformer_real_splits_v1/exp13_best_config/splitseed${s}_trainseed42

  mkdir -p $OUT_DIR

  CUDA_VISIBLE_DEVICES=0 python3 -u /home/htaucare/Tesis_Amaro/scripts_last_version/pointnettransformer_classic_final_v5_patch.py \
    --data_dir $DATA_DIR \
    --epochs 120 \
    --batch_size 4 \
    --lr 0.0002 \
    --weight_decay 0.0001 \
    --num_workers 6 \
    --infer_num_workers 0 \
    --device cuda \
    --seed 42 \
    --use_amp \
    --grad_clip 1.0 \
    --bg_index 0 \
    --bg_weight 0.03 \
    --d21_internal 8 \
    --neighbor_teeth d11:1,d22:9 \
    --neighbor_eval_split both \
    --neighbor_every 1 \
    --index_csv ${DATA_DIR}/index_test.csv \
    --do_infer \
    --infer_examples 20 \
    --infer_split test \
    --train_metrics_eval \
    --d_model 256 \
    --embed_hidden 128 \
    --depth 4 \
    --nhead 8 \
    --dim_feedforward 512 \
    --dropout 0.05 \
    --head_hidden 256 \
    --head_dropout 0.05 \
    --use_pos_mlp \
    --pos_hidden 128 \
    --norm_first \
    --activation gelu \
    --token_subsample 2048 \
    --prop_k 3 \
    --prop_chunk 2048 \
    --out_dir $OUT_DIR

done
```

---

# 6. Cálculo de media y desviación estándar

El script para resumir métricas es:

```bash
scripts_last_version/compute_metrics_summary.py
```

Este script puede calcular:

- media
- desviación estándar
- mínimo
- máximo
- número de corridas válidas

sobre distintas fuentes:

```bash
--mode test_filtered_json
--mode test_json
--mode best_val_d21_f1
--mode last_epoch
```

---

## 6.1 Reporte principal recomendado

Para resultados principales de tesis/paper se recomienda usar:

```bash
--mode test_filtered_json
```

porque lee:

```bash
test_metrics_filtered.json
```

Este archivo excluye muestras no válidas para inferencia.

---

## 6.2 Reporte de PointNet Classic

```bash
cd /home/htaucare/Tesis_Amaro

python scripts_last_version/compute_metrics_summary.py \
  --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnet_classic/stability_dataset_seeds_v2/exp03_best_config \
  --pattern "splitseed*_trainseed42" \
  --mode test_filtered_json \
  --out_csv summary_pointnet_test_filtered.csv
```

---

## 6.3 Reporte de PointNet++

```bash
cd /home/htaucare/Tesis_Amaro

python scripts_last_version/compute_metrics_summary.py \
  --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnetpp/stability_real_splits_v1/exp04_best_config \
  --pattern "splitseed*_trainseed42" \
  --mode test_filtered_json \
  --out_csv summary_pointnetpp_test_filtered.csv
```

---

## 6.4 Reporte de DGCNN

```bash
cd /home/htaucare/Tesis_Amaro

python scripts_last_version/compute_metrics_summary.py \
  --base_dir /home/htaucare/Tesis_Amaro/outputs/dgcnn/dgcnn_real_splits_v1/exp06_best_config \
  --pattern "splitseed*_trainseed42" \
  --mode test_filtered_json \
  --out_csv summary_dgcnn_test_filtered.csv
```

---

## 6.5 Reporte de PointNet-Transformer

```bash
cd /home/htaucare/Tesis_Amaro

python scripts_last_version/compute_metrics_summary.py \
  --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnettransformer/pointnettransformer_real_splits_v1/exp13_best_config \
  --pattern "splitseed*_trainseed42" \
  --mode test_filtered_json \
  --out_csv summary_pointnettransformer_test_filtered.csv
```

---

# 7. Reportes de entrenamiento/validación

Para analizar comportamiento de entrenamiento y validación no filtrado, se puede usar:

```bash
--mode best_val_d21_f1
```

Ejemplo para PointNet++:

```bash
python scripts_last_version/compute_metrics_summary.py \
  --base_dir /home/htaucare/Tesis_Amaro/outputs/pointnetpp/stability_real_splits_v1/exp04_best_config \
  --pattern "splitseed*_trainseed42" \
  --mode best_val_d21_f1 \
  --out_csv summary_pointnetpp_best_epoch_train_val.csv
```

Este reporte usa `metrics_epoch.csv`; por lo tanto, no corresponde al test filtrado. Sirve para analizar convergencia, estabilidad y dinámica de entrenamiento.

---

# 8. Qué reporte usar en la tesis

## Tabla principal

Usar:

```bash
test_metrics_filtered.json
```

mediante:

```bash
--mode test_filtered_json
```

Esto entrega métricas filtradas de test.

## Análisis de entrenamiento

Usar:

```bash
metrics_epoch.csv
```

mediante:

```bash
--mode best_val_d21_f1
```

o:

```bash
--mode last_epoch
```

Esto permite analizar entrenamiento y validación, pero no es la tabla principal de resultados.

---

# 9. Resumen metodológico

Los resultados principales deben reportarse como:

```text
media ± desviación estándar
```

sobre las semillas:

```text
1, 7, 21, 42, 100
```

La tabla principal debe basarse en test filtrado, mientras que las curvas y análisis de entrenamiento pueden basarse en métricas no filtradas de `metrics_epoch.csv`.
