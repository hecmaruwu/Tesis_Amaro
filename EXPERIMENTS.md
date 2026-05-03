# Mini README — Runners de experimentos (Tesis_Amaro)

Este README resume, de forma general, qué hacen los 4 scripts de barrido de experimentos, cómo ejecutarlos y qué puntos importantes considerar.

## 1. ¿Qué hacen estos scripts?

Los 4 runners sirven para **automatizar experimentos** de entrenamiento sobre una misma base de datos dental 3D, evitando lanzar cada corrida manualmente.

Cada runner:

- define una **grilla de hiperparámetros**,
- ejecuta los experimentos **uno por uno**,
- guarda un **log por experimento**,
- recoge automáticamente métricas desde los outputs del entrenamiento,
- genera un resumen global,
- y elige automáticamente el **mejor experimento** según la métrica principal.

La métrica principal usada para rankear es normalmente:

- `test_filtered_d21_f1`

con desempates usando otras métricas como:

- `test_filtered_f1_macro`
- `test_filtered_iou_macro`
- `test_d21_f1`
- `val_best_d21_f1`

---

## 2. Scripts incluidos

### A. PointNet clásico
Script runner:
- `run_pointnet_classic_experiments_pro_v1.py`

Script de entrenamiento esperado:
- `pointnet_classic_final_v8_patch.py`

Uso general:
- barre hiperparámetros de PointNet clásico,
- guarda resumen de resultados,
- elige el mejor run.

---

### B. PointNet++
Script runner:
- `run_pointnetpp_experiments_pro_v2.py`

Script de entrenamiento esperado:
- `pointnetpp_classic_final_v1_patch.py`

Uso general:
- corre múltiples configuraciones de PointNet++,
- incluye versión v2 con **reintento automático por OOM**,
- si falla por memoria, reduce `batch_size` automáticamente.

---

### C. DGCNN
Script runner:
- `run_dgcnn_experiments_pro_v2_gpu1.py`  
  o la variante que usted haya guardado para GPU 0/1.

Script de entrenamiento esperado:
- `train_dgcnn_classic_only_fixed_v9_patch.py`

Uso general:
- barre hiperparámetros de DGCNN,
- maneja errores por memoria,
- reintenta con `batch_size` menor si detecta OOM.

---

### D. PointNet-Transformer
Script runner:
- `run_pointnettransformer_experiments_pro_v2_gpu1.py`  
  o la versión corregida para GPU 0.

Script de entrenamiento esperado:
- `pointnettransformer_classic_final_v5_patch.py`

Uso general:
- corre distintas configuraciones del backbone Transformer,
- ajusta parámetros como `d_model`, `depth`, `nhead`, `dim_feedforward`, `token_subsample`, etc.,
- también reintenta automáticamente si hay OOM.

---

## 3. Estructura general de funcionamiento

Todos los runners siguen esta lógica:

1. Definen:
   - `SCRIPT_PATH`
   - `DATA_DIR`
   - `INDEX_CSV`
   - `RUNS_ROOT`
   - `LOGS_ROOT`
   - `GPU_ID`
   - `EXPERIMENT_GRID`

2. Para cada experimento:
   - construyen el comando de entrenamiento,
   - crean una carpeta de salida,
   - ejecutan el entrenamiento,
   - guardan el log,
   - leen:
     - `run.log`
     - `test_metrics.json`
     - `test_metrics_filtered.json`

3. Al final generan:
   - `experiments_summary.csv`
   - `experiments_summary.json`
   - `best_experiment.json`

---

## 4. Archivos de salida importantes

Dentro del root de cada runner quedan:

- `experiments_summary.csv`  
  Resumen tabular de todos los experimentos

- `experiments_summary.json`  
  Resumen en JSON

- `best_experiment.json`  
  Mejor experimento encontrado

Y dentro de cada carpeta de experimento, normalmente:

- `best.pt`
- `last.pt`
- `run.log`
- `test_metrics.json`
- `test_metrics_filtered.json`
- `history.json`
- `metrics_epoch.csv`
- `plots/`
- `inference/`

---

## 5. Cómo ejecutarlos

### Paso previo
Activar entorno:

```bash
conda activate enviroment
```

### Ejemplo general

```bash
python3 -u /home/htaucare/Tesis_Amaro/run_experiments/NOMBRE_DEL_RUNNER.py
```

### Ejemplos concretos

#### PointNet clásico
```bash
python3 -u /home/htaucare/Tesis_Amaro/run_experiments/run_pointnet_classic_experiments_pro_v1.py
```

#### PointNet++
```bash
python3 -u /home/htaucare/Tesis_Amaro/run_experiments/run_pointnetpp_experiments_pro_v2.py
```

#### DGCNN
```bash
python3 -u /home/htaucare/Tesis_Amaro/run_experiments/run_dgcnn_experiments_pro_v2_gpu1.py
```

#### PointNet-Transformer
```bash
python3 -u /home/htaucare/Tesis_Amaro/run_experiments/run_pointnettransformer_experiments_pro_v2_gpu1.py
```

---

## 6. Uso de GPU

Cada runner tiene una línea tipo:

```python
GPU_ID = "0"
```

o

```python
GPU_ID = "1"
```

Eso define qué GPU usará internamente el script.

### Recomendación
Antes de correr, revisar que el `GPU_ID` del runner sea el correcto.

Además, si quiere reforzarlo desde bash:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u ...
```

Pero si el runner ya fija `GPU_ID`, normalmente no hace falta.

---

## 7. Qué hacer si se ejecutó en la GPU equivocada

Si un runner se lanzó en la GPU incorrecta:

1. borrar la carpeta de outputs de ese runner,
2. corregir `GPU_ID`,
3. opcionalmente cambiar `RUNS_ROOT`/`LOGS_ROOT` para que el nombre quede consistente,
4. volver a ejecutar.

---

## 8. OOM (Out Of Memory)

Los runners v2 para PointNet++, DGCNN y PointNet-Transformer están diseñados para detectar OOM.

Si un experimento falla por memoria:

- revisan el log,
- detectan el error,
- y reintentan con `batch_size` más pequeño.

Fallback típico:

- `8 -> 4 -> 2 -> 1`

Esto es útil especialmente para:

- DGCNN
- PointNet-Transformer

porque suelen consumir más VRAM que PointNet clásico.

---

## 9. Qué revisar antes de correr

Antes de lanzar cualquier runner:

- confirmar que `SCRIPT_PATH` apunta al script correcto,
- confirmar que `DATA_DIR` existe,
- confirmar que `INDEX_CSV` existe,
- revisar `GPU_ID`,
- revisar que `RUNS_ROOT` no mezcle resultados viejos con nuevos,
- revisar que la grilla (`EXPERIMENT_GRID`) tenga hiperparámetros razonables.

---

## 10. Recomendaciones prácticas

### A. Mantener un runner por arquitectura
Esto facilita:
- trazabilidad,
- comparación,
- orden de outputs.

### B. No mezclar runs de GPU 0 y GPU 1 en la misma carpeta
Mejor usar nombres como:
- `grid_pro_runner_v2_gpu0`
- `grid_pro_runner_v2_gpu1`

### C. Revisar el mejor experimento al final
El archivo más importante suele ser:

- `best_experiment.json`

### D. Abrir el resumen en CSV
El archivo:

- `experiments_summary.csv`

sirve para revisar rápidamente qué configuración rindió mejor.

---

## 11. Resumen corto

- **PointNet clásico**: runner simple para barrido de hiperparámetros.
- **PointNet++**: runner con soporte de reintento por OOM.
- **DGCNN**: runner robusto con manejo de memoria y ranking automático.
- **PointNet-Transformer**: runner adaptado a arquitectura Transformer, más sensible a VRAM, idealmente con batch pequeño y AMP en GPU potente como RTX 3090.

---

## 12. Nota final

Estos runners no cambian la lógica interna del modelo:  
solo automatizan la ejecución de múltiples configuraciones y consolidan los resultados para elegir el mejor modelo de forma ordenada y reproducible.
