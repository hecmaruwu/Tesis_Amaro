# 🧠 3D Dental Segmentation Pipeline (Teeth3DS)

## 📌 Descripción general

Este repositorio implementa un pipeline completo para segmentación dental 3D basado en nubes de puntos, utilizando el dataset Teeth3DS.

El flujo abarca desde datos crudos hasta entrenamiento de modelos como:

- PointNet
- PointNet++
- DGCNN
- Point Transformer

El pipeline está diseñado para ser:

✔ reproducible  
✔ trazable  
✔ consistente entre modelos  
✔ alineado con estándares de investigación  

---

## 🧬 PIPELINE GENERAL

RAW → SURFACE SAMPLING (200k) → MERGED → SUBSAMPLING + AUGMENT (8192) → TRAIN

⚠️ Importante:  
NO se utiliza FPS en ninguna etapa del pipeline final

---

## 📂 ESTRUCTURA DEL DATASET

data/Teeth_3ds/raw/
 ├── data_part_1/
 ├── data_part_2/
 ├── ...
 ├── data_part_7/
     ├── upper / lower
         ├── PATIENT_ID/
             ├── *.obj / *.ply
             ├── *.json

---

## 🟢 ETAPA 1 — Preprocesamiento SAFE

Script:
scripts_last_version/preprocess_and_flatten_safe.py

Funcionalidad:
- Muestreo de superficie usando trimesh.sample_surface
- 200.000 puntos por arcada
- Normalización (centro + radio unitario)
- Etiquetado con cKDTree
- Exclusión de muelas del juicio → background

Comando:
python scripts_last_version/preprocess_and_flatten_safe.py   --in_root data/Teeth_3ds/raw   --out_struct_root data/Teeth_3ds/processed_struct_safe   --out_flat_root data/Teeth_3ds/processed_flat_safe   --parts 1 2 3 4 5 6 7   --jaws upper   --n_points 200000   --sample_mode global   --exclude_wisdom_to_0   --seed 42

---

## 🟢 ETAPA 2 — Fusión (merged 200k)

Script:
scripts_last_version/build_merged_pointcloud_dataset_surf_global.py

Funcionalidad:
- Unión de muestras
- 200k puntos por muestra
- Generación de index_*.csv
- label_map.json

---

## 🟢 ETAPA 3 — Submuestreo + augment + split

Script:
scripts_last_version/augmentation_and_split_paperlike_v3.py

Funcionalidad:
- 200k → 8192 puntos
- Submuestreo aleatorio
- cap_bg = 0.8
- ensure_coverage
- Data augmentation
- Split reproducible

---

## 🟢 ETAPA 4 — Entrenamiento

Modelos:
- PointNet
- PointNet++
- DGCNN
- Point Transformer

Outputs:
- history.json
- metrics_epoch.csv
- test_metrics.json
- test_metrics_filtered.json
- plots/
- inference/

---

## 📊 MÉTRICAS

Globales:
- IoU macro
- F1 macro
- Accuracy sin background

Diente 21:
- d21_acc
- d21_f1
- d21_iou

Filtrado:
- test_metrics.json (no filtrado)
- test_metrics_filtered.json (filtrado)

---

## 📌 RESUMEN

Pipeline basado en:
- Surface sampling
- Subsampling controlado
- Dataset parcialmente balanceado
- Evaluación robusta con filtrado
