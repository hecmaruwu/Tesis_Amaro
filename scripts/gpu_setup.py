#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, tensorflow as tf

# Si no viene definido, por defecto usa GPU 1 (la segunda)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("[GPU] GPUs:", gpus, " | memory growth = True")
    except Exception as e:
        print("[GPU][WARN] No se pudo setear memory growth:", e)
else:
    print("[GPU][WARN] No se detectó GPU disponible")
