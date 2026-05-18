# DATASETS Y PIPELINES DE MUESTREO — TESIS_AMARO

Este documento resume los distintos datasets utilizados para entrenar modelos
de segmentación dental 3D, incluyendo:
- tipo de muestreo,
- cantidad de puntos,
- uso de augmentation,
- comandos de generación,
- restauración de trazabilidad.

## DATASET BASE

```bash
data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only
```

Características:
- upper only
- SAFE pipeline
- sin muelas del juicio
- 15 clases finales
- surface sampling global

---

# 1. RANDOM — 8192 — CON AUGMENTATION

## Dataset

```bash
data/Teeth_3ds/fixed_split/8192/random_upper_only_seed42_aug2
```

## Generación

```bash
cd /home/htaucare/Tesis_Amaro

python scripts_last_version/augmentation_and_split_paperlike_v3.py \
  --dataset_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --out_dir data/Teeth_3ds/fixed_split/8192/random_upper_only_seed42_aug2 \
  --N 8192 \
  --cap_bg 0.8 \
  --seed 42 \
  --use_augmentation \
  --augment_times 2 \
  --ensure_coverage \
  2>&1 | tee build_random_8192_aug2.log
```

## Restaurar trazabilidad

```bash
python scripts_last_version/create_traceability_indices_for_fixed_split.py \
  --source_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --target_dir data/Teeth_3ds/fixed_split/8192/random_upper_only_seed42_aug2 \
  --augment_times 2 \
  --overwrite
```

---

# 2. RANDOM — 8192 — SIN AUGMENTATION

## Dataset

```bash
data/Teeth_3ds/fixed_split/8192/random_upper_only_seed42_noaug
```

## Generación

```bash
cd /home/htaucare/Tesis_Amaro

python scripts_last_version/augmentation_and_split_paperlike_v3.py \
  --dataset_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --out_dir data/Teeth_3ds/fixed_split/8192/random_upper_only_seed42_noaug \
  --N 8192 \
  --cap_bg 0.8 \
  --seed 42 \
  --ensure_coverage \
  2>&1 | tee build_random_noaug_original_pipeline.log
```

## Restaurar trazabilidad

```bash
python scripts_last_version/create_traceability_indices_for_fixed_split.py \
  --source_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --target_dir data/Teeth_3ds/fixed_split/8192/random_upper_only_seed42_noaug \
  --augment_times 0 \
  --overwrite
```

---

# 3. FPS — 8192 — CON AUGMENTATION

## Dataset

```bash
data/Teeth_3ds/fixed_split/8192/fps_upper_only_seed42_aug2
```

## Generación

```bash
cd /home/htaucare/Tesis_Amaro

python -u scripts_last_version/augmentation_and_split_paperlike_v3_sampleo_FPS_SAFE_RESUME.py \
  --dataset_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --out_dir data/Teeth_3ds/fixed_split/8192/fps_upper_only_seed42_aug2 \
  --N 8192 \
  --sampling_method fps \
  --cap_bg 0.8 \
  --seed 42 \
  --use_augmentation \
  --augment_times 2 \
  --ensure_coverage \
  --use_safe_resume \
  2>&1 | tee build_fps_8192_aug2_safe_resume.log
```

## Restaurar trazabilidad

```bash
python scripts_last_version/create_traceability_indices_for_fixed_split.py \
  --source_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --target_dir data/Teeth_3ds/fixed_split/8192/fps_upper_only_seed42_aug2 \
  --augment_times 2 \
  --overwrite
```

---

# 4. FPS — 8192 — SIN AUGMENTATION

## Dataset

```bash
data/Teeth_3ds/fixed_split/8192/fps_upper_only_seed42_noaug_safe_resume
```

## Generación

```bash
cd /home/htaucare/Tesis_Amaro

python -u scripts_last_version/augmentation_and_split_paperlike_v3_sampleo_FPS_SAFE_RESUME.py \
  --dataset_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --out_dir data/Teeth_3ds/fixed_split/8192/fps_upper_only_seed42_noaug_safe_resume \
  --N 8192 \
  --sampling_method fps \
  --cap_bg 0.8 \
  --seed 42 \
  --ensure_coverage \
  --use_safe_resume \
  2>&1 | tee build_fps8192_noaug_safe_resume.log
```

## Restaurar trazabilidad

```bash
python scripts_last_version/create_traceability_indices_for_fixed_split.py \
  --source_dir data/Teeth_3ds/merged_200000_safe_excl_wisdom_upper_only \
  --target_dir data/Teeth_3ds/fixed_split/8192/fps_upper_only_seed42_noaug_safe_resume \
  --augment_times 0 \
  --overwrite
```
