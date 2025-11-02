#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador 3D de mallas dentales con color sólido por diente.
"""

import argparse, json
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt

LABEL_COLORS = {
    0: 'red', 11: 'blue', 12: 'green', 13: 'orange', 14: 'purple', 15: 'cyan',
    16: 'magenta', 17: 'yellow', 18: 'brown', 21: 'lime', 22: 'navy', 23: 'teal',
    24: 'violet', 25: 'salmon', 26: 'gold', 27: 'lightblue', 28: 'coral', 31: 'olive',
    32: 'silver', 33: 'gray', 34: 'black', 35: 'darkred', 36: 'darkgreen',
    37: 'darkblue', 38: 'darkviolet', 41: 'peru', 42: 'chocolate', 43: 'mediumvioletred',
    44: 'lightskyblue', 45: 'lightpink', 46: 'plum', 47: 'khaki', 48: 'powderblue'
}

def load_mesh_safe(path: Path):
    if not path.exists():
        print(f"[WARN] No existe: {path}")
        return None, None
    try:
        mesh = trimesh.load_mesh(path, process=False)
        return mesh.vertices, mesh.faces
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {path}: {e}")
        return None, None

def load_labels_safe(json_path: Path, num_vertices: int):
    if not json_path.exists():
        print(f"[WARN] No existe JSON: {json_path}")
        return np.zeros(num_vertices, dtype=int)
    try:
        data = json.load(open(json_path))
        labels = np.array(data.get("labels") or data.get("vertex_labels") or [])
        if labels.size == 0:
            print(f"[WARN] JSON sin campo de labels: {json_path}")
            return np.zeros(num_vertices, dtype=int)
        if labels.shape[0] != num_vertices:
            print(f"[WARN] {json_path.name}: etiquetas ({labels.shape[0]}) ≠ vértices ({num_vertices})")
            labels = np.resize(labels, num_vertices)
        return labels
    except Exception as e:
        print(f"[ERROR] No se pudieron cargar etiquetas de {json_path}: {e}")
        return np.zeros(num_vertices, dtype=int)

def plot_mesh_colored(ax, vertices, faces, labels):
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = np.isin(faces, np.where(labels == lbl)).any(axis=1)
        sub_faces = faces[mask]
        color = LABEL_COLORS.get(int(lbl), 'lightgray')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=sub_faces, color=color, linewidth=0.05, alpha=0.9)

def visualize_upper_lower(upper, lower, upper_labels, lower_labels, pid, out_png):
    fig = plt.figure(figsize=(12, 6))

    # --- Upper ---
    ax1 = fig.add_subplot(121, projection='3d')
    if upper[0] is not None:
        plot_mesh_colored(ax1, upper[0], upper[1], upper_labels)
        ax1.set_title(f"Upper — {pid}")
    else:
        ax1.text(0.5, 0.5, 0.5, "No Upper", color='red', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Upper — [no data]")
    ax1.axis('off')

    # --- Lower ---
    ax2 = fig.add_subplot(122, projection='3d')
    if lower[0] is not None:
        plot_mesh_colored(ax2, lower[0], lower[1], lower_labels)
        ax2.set_title(f"Lower — {pid}")
    else:
        ax2.text(0.5, 0.5, 0.5, "No Lower", color='red', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Lower — [no data]")
    ax2.axis('off')

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Imagen guardada: {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    root, pid = Path(args.data_root), args.patient

    upper_mesh = root / "upper" / pid / f"{pid}_upper.obj"
    lower_mesh = root / "lower" / pid / f"{pid}_lower.obj"
    upper_json = root / "upper" / pid / f"{pid}_upper.json"
    lower_json = root / "lower" / pid / f"{pid}_lower.json"

    upper = load_mesh_safe(upper_mesh)
    lower = load_mesh_safe(lower_mesh)
    upper_labels = load_labels_safe(upper_json, len(upper[0])) if upper[0] is not None else np.array([])
    lower_labels = load_labels_safe(lower_json, len(lower[0])) if lower[0] is not None else np.array([])

    if upper[0] is None and lower[0] is None:
        print(f"[ERROR] No se encontraron mallas para {pid}.")
        return

    visualize_upper_lower(upper, lower, upper_labels, lower_labels, pid, args.out_png)

if __name__ == "__main__":
    main()
