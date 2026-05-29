#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAMERAS = {
    "isometrica": {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {"x": 1.35, "y": 1.35, "z": 0.85},
        "projection": {"type": "perspective"},
    },
    "oclusal_oblicua": {
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {"x": 0.35, "y": 0.35, "z": 2.15},
        "projection": {"type": "perspective"},
    },
    "frontal_oblicua": {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0.08},
        "eye": {"x": 0.15, "y": 2.05, "z": 0.45},
        "projection": {"type": "perspective"},
    },
    "lateral_derecha": {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {"x": 2.10, "y": 0.10, "z": 0.35},
        "projection": {"type": "perspective"},
    },
}


def build_injection(camera_name: str, camera: dict) -> str:
    camera_json = json.dumps(camera, ensure_ascii=False)

    return f"""
<script>
(function() {{
    const CAMERA_NAME = {json.dumps(camera_name)};
    const MASTER_CAMERA = {camera_json};
    const SCENES = ["scene", "scene2", "scene3", "scene4", "scene5", "scene6"];

    function applyCamera() {{
        const gd = document.querySelector(".plotly-graph-div");

        if (!gd || typeof Plotly === "undefined" || !gd._fullLayout) {{
            setTimeout(applyCamera, 500);
            return;
        }}

        const update = {{}};
        SCENES.forEach(s => {{
            update[s + ".camera"] = MASTER_CAMERA;
        }});

        update["title.text"] = (gd._fullLayout.title && gd._fullLayout.title.text
            ? gd._fullLayout.title.text
            : "Vista cualitativa") + " — " + CAMERA_NAME;

        Plotly.relayout(gd, update).then(() => {{
            console.log("[OK] Cámara aplicada:", CAMERA_NAME);
            console.log(JSON.stringify(MASTER_CAMERA, null, 2));
        }});

        let syncing = false;
        gd.on("plotly_relayout", function(e) {{
            if (syncing) return;

            const cam =
                e["scene.camera"] ||
                e["scene2.camera"] ||
                e["scene3.camera"] ||
                e["scene4.camera"] ||
                e["scene5.camera"] ||
                e["scene6.camera"];

            if (cam) {{
                syncing = true;
                const syncUpdate = {{}};
                SCENES.forEach(s => {{
                    syncUpdate[s + ".camera"] = cam;
                }});

                Plotly.relayout(gd, syncUpdate).then(() => {{
                    syncing = false;
                    console.log("[SYNC_CAMERA]");
                    console.log(JSON.stringify(cam, null, 2));
                }});
            }}
        }});

        window.printCameras = function() {{
            const out = {{}};
            SCENES.forEach(s => {{
                out[s] = gd._fullLayout[s] ? gd._fullLayout[s].camera : null;
            }});
            console.log(JSON.stringify(out, null, 2));
        }};

        window.printMasterCamera = function() {{
            console.log(JSON.stringify(MASTER_CAMERA, null, 2));
        }};
    }}

    setTimeout(applyCamera, 800);
}})();
</script>
"""


def inject_html(src_html: Path, out_html: Path, camera_name: str, camera: dict) -> None:
    html = src_html.read_text(encoding="utf-8", errors="replace")
    injection = build_injection(camera_name, camera)

    if "</body>" in html:
        html = html.replace("</body>", injection + "\n</body>")
    else:
        html = html + "\n" + injection

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_html",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17/median_row089_015RHV4X_upper/paper_dgcnn_median_v17.html",
    )
    ap.add_argument(
        "--out_dir",
        default="/home/htaucare/Tesis_Amaro/case_comparisons/QUALI_CASES_V17/median_row089_015RHV4X_upper/camera_test_views_from_html",
    )
    ap.add_argument("--prefix", default="dgcnn_median")
    args = ap.parse_args()

    src_html = Path(args.src_html)
    out_dir = Path(args.out_dir)

    if not src_html.exists():
        raise FileNotFoundError(f"No existe src_html: {src_html}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cam in CAMERAS.items():
        out_html = out_dir / f"{args.prefix}_{name}.html"
        inject_html(src_html, out_html, name, cam)
        print(f"[OK] {name}: {out_html}")

    meta = {
        "src_html": str(src_html),
        "out_dir": str(out_dir),
        "prefix": args.prefix,
        "cameras": CAMERAS,
        "note": "Abrir cada HTML, revisar la vista y usar el botón Download plot as png si se quiere exportar.",
    }

    (out_dir / "camera_test_views_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[DONE] HTMLs generados.")


if __name__ == "__main__":
    main()
