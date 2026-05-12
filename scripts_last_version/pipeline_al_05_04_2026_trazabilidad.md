# Pipeline completo de segmentación dental 3D con trazabilidad
**Fecha:** 05/04/2026  
**Proyecto:** Tesis_Amaro  

---

# 1. Visión general del pipeline

Este pipeline permite:

- Procesar mallas dentales 3D
- Generar nubes de puntos estructuradas
- Construir datasets reproducibles
- Entrenar modelos (PointNet, PointNet++, DGCNN, Transformer)
- Realizar inferencia trazable a nivel de paciente

---

# 2. Flujo completo

```
RAW → PREPROCESS → MERGED → FIXED_SPLIT → TRACEABILITY → TRAIN → INFERENCE
```

---

# 3. ETAPA 0 — RAW DATA

Ruta base:
```
data/Teeth_3ds/raw/
```

Contiene:
- Archivos OBJ/STL
- Datos por paciente

No existe aún:
- Normalización
- Etiquetas estructuradas
- Consistencia geométrica

---

# 4. ETAPA 1 — PREPROCESS

Script:
```
preprocess_and_flatten_safe.py
```

Salida:
```
processed_struct_safe/200000/<jaw>/<sample>/
    point_cloud.npy
    labels.npy
```

### Funciones principales:

- Muestreo de superficie (trimesh.sample_surface)
- 200.000 puntos por arcada
- Normalización a esfera unitaria
- Limpieza de NaN/Inf
- Exclusión de muelas del juicio → clase 0

Resultado:
- Cada paciente queda representado como nube independiente

---

# 5. ETAPA 2 — MERGED DATASET (CRÍTICA)

Script:
```
build_merged_pointcloud_dataset.py
```

Salida:
```
merged_200000_safe_excl_wisdom_upper_only/
    X_train.npz
    X_val.npz
    X_test.npz

    Y_train.npz
    Y_val.npz
    Y_test.npz

    index_train.csv
    index_val.csv
    index_test.csv

    artifacts/
```

---

## 5.1 Qué ocurre aquí

### Construcción del dataset global

```
X.shape = (N_samples, 200000, 3)
Y.shape = (N_samples, 200000)
```

---

### Split reproducible

```
train / val / test
seed = 42
```

---

### Creación de trazabilidad (PUNTO CLAVE)

Cada fila queda asociada a:

```
idx, sample_name, jaw, path
```

Ejemplo:

```
X_test[0] ↔ sample_name = 01MAVHYZ
```

---

## 5.2 Importancia

Este script define:

- Qué pacientes pertenecen a cada split
- El orden de las muestras
- La correspondencia fila → paciente

👉 Este es el ORIGEN de la trazabilidad

---

# 6. ETAPA 3 — FIXED SPLIT (SUBSAMPLING)

Script:
```
augmentation_and_split_paperlike_v3.py
```

Entrada:
```
merged_200000...
```

Salida:
```
fixed_split/12000/.../
    X_train.npz
    X_val.npz
    X_test.npz

    Y_train.npz
    Y_val.npz
    Y_test.npz
```

---

## 6.1 Qué hace

- Submuestreo:
```
200000 → 12000 puntos
```

- Augmentation (solo train)
- Balanceo implícito de clases
- Conservación del orden de muestras

---

## 6.2 Problema

NO genera:

```
index_*.csv
```

👉 Se pierde trazabilidad explícita

---

# 7. ETAPA 4 — RESTAURACIÓN DE TRAZABILIDAD

Script creado:
```
create_traceability_indices_for_fixed_split.py
```

---

## 7.1 Función

- Copia trazabilidad desde merged
- Reconstruye index para fixed_split

---

## 7.2 Comportamiento

### val/test

```
1:1 con merged
```

### train

```
original + augmentaciones
```

Ejemplo:

```
760 originales
+ 760 augment 1
+ 760 augment 2
= 2280
```

---

## 7.3 Resultado final

```
fixed_split/.../
    X_*.npz
    Y_*.npz
    index_*.csv
```

---

# 8. ETAPA 5 — ENTRENAMIENTO

Modelos:

- PointNet
- PointNet++
- DGCNN
- Transformer

---

## Input clave

```
--data_dir
--index_csv
```

---

## Funcionamiento

```
X_test[i]
↓
index_test.csv[i]
↓
sample_name
↓
paciente real
```

---

# 9. ETAPA 6 — INFERENCIA

Salida:

```
outputs/.../inference/
    inference_all/
        01MAVHYZ.png
        01FR2S9H.png
```

---

## Resultado final

```
predicción → paciente → visualización
```

---

# 10. TRAZABILIDAD — ANTES VS AHORA

---

## Antes

- La trazabilidad existía implícitamente
- Dependía del orden del dataset
- No se guardaba en fixed_split

Resultado:

```
test_row0.png
```

---

## Ahora

- Trazabilidad explícita
- Persistida en index_*.csv
- Consistente entre datasets

Resultado:

```
01MAVHYZ.png
```

---

# 11. CONDICIÓN DE VALIDEZ

El sistema funciona SI:

- fixed_split proviene del mismo merged
- no se reordena el dataset
- no se cambia el split

---

# 12. CONTRIBUCIÓN DEL TRABAJO

Se implementó:

- Restauración de trazabilidad en datasets derivados
- Compatibilidad con distintos tamaños (8192, 12000, etc.)
- Reproducibilidad completa del pipeline

---

# 13. FRASE PARA TESIS

"La trazabilidad entre predicciones y muestras originales se establece en la etapa de construcción del dataset global (merged), donde cada muestra es asociada a un identificador único. Dado que las etapas posteriores de submuestreo no preservan explícitamente esta información, se implementó un mecanismo adicional para restaurar dicha correspondencia en los datasets derivados, asegurando la reproducibilidad completa del pipeline."

---

# 14. CONCLUSIÓN

El pipeline final permite:

- Comparaciones justas entre configuraciones
- Evaluación reproducible
- Análisis clínicamente interpretable

---

# 15. PIPELINE FINAL

```
RAW
↓
PREPROCESS (200k)
↓
MERGED (dataset + index)
↓
SUBSAMPLING (N puntos)
↓
RESTORE TRACEABILITY
↓
TRAIN
↓
INFERENCE
```

---

**Estado:** VALIDADO Y FUNCIONAL

