#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_utils.py
----------------
Utilidades de data augmentation para nubes de puntos.
Incluye rotación en Z, jitter, escala global, dropout de puntos y normalización.
"""

import tensorflow as tf
import numpy as np

def normalize_cloud_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    x = x - mean
    r = np.linalg.norm(x, axis=1, keepdims=True).max()
    return x / (r + 1e-6)

def rotate_z(points, max_deg=15.0):
    theta = tf.random.uniform([], -max_deg, max_deg) * (np.pi / 180.0)
    c, s = tf.cos(theta), tf.sin(theta)
    R = tf.convert_to_tensor([[c, -s, 0.0],
                              [s,  c, 0.0],
                              [0.0, 0.0, 1.0]], dtype=tf.float32)
    return tf.matmul(points, R)

def jitter(points, sigma=0.005, clip=0.02):
    noise = tf.clip_by_value(
        sigma * tf.random.normal(tf.shape(points)), -clip, clip
    )
    return points + noise

def scale(points, min_s=0.95, max_s=1.05):
    s = tf.random.uniform([], min_s, max_s)
    return points * s

def dropout_points(points, drop_rate=0.05):
    B, P, C = tf.shape(points)[0], tf.shape(points)[1], tf.shape(points)[2]
    mask = tf.random.uniform([B, P], 0, 1) > drop_rate
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, -1)
    pts = points * mask
    # Reemplazar puntos dropout con vecinos (remuestreo)
    keep_idx = tf.where(tf.reduce_sum(mask, axis=-1) > 0, 1, 0)
    keep_idx = tf.cast(keep_idx, tf.bool)
    pts = tf.where(tf.expand_dims(keep_idx, -1), pts, points)
    return pts

def augment(points,
            rotate_deg=15,
            jitter_sigma=0.005, jitter_clip=0.02,
            scale_min=0.95, scale_max=1.05,
            dropout_rate=0.05):
    x = rotate_z(points, max_deg=rotate_deg)
    x = jitter(x, sigma=jitter_sigma, clip=jitter_clip)
    x = scale(x, min_s=scale_min, max_s=scale_max)
    x = dropout_points(tf.expand_dims(x, 0), drop_rate=dropout_rate)
    return tf.squeeze(x, axis=0)
