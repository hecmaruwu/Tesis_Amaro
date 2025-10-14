#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, itertools, subprocess, os, csv, datetime
from pathlib import Path

# ==== Mapea tus flags reales de train_models.py ====
FLAG_MAP = {
    "model": "--model",
    "epochs": "--epochs",
    "batch_size": "--batch_size",
    "lr": "--lr",
    "seed": "--seed",
    "num_points": "--num_points",
    "augment": "--augment",
    "data_dir": "--data_dir",
    "run_dir": "--run_dir"
}
# ===================================================

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def main():
    cfg_path = Path("configs/experiments.json")
    if not cfg_path.exists():
        raise SystemExit(f"No encuentro {cfg_path}")
    cfg = json.load(open(cfg_path, "r"))

    global_cfg = cfg.get("global", {})
    sweeps = cfg.get("sweeps", [])
    if not sweeps:
        raise SystemExit("No hay sweeps en configs/experiments.json")

    run_root = Path(global_cfg.get("run_root", "runs_grid"))
    ensure_dir(run_root)

    # fija GPU=1
    env_base = os.environ.copy()
    env_base["CUDA_VISIBLE_DEVICES"] = str(global_cfg.get("gpu", "1"))

    summary_csv = run_root / f"summary_{now_tag()}.csv"
    with open(summary_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["suite","model","epochs","batch_size","lr","seed","num_points","augment","run_dir","status","retcode"])

        for sweep in sweeps:
            suite_name = sweep["name"]
            models = to_list(sweep["model"])
            batch_sizes = to_list(sweep.get("batch_size", global_cfg.get("batch_size", 8)))
            lrs = to_list(sweep.get("lr", global_cfg.get("lr", 1e-3)))
            seeds = to_list(sweep.get("seed", [42]))
            num_points = to_list(sweep.get("num_points", global_cfg.get("num_points", 8192)))
            epochs = to_list(sweep.get("epochs", global_cfg.get("epochs", 120)))
            augment = to_list(sweep.get("augment", global_cfg.get("augment", "")))

            for (model, bs, lr, sd, npnts, ep, aug) in itertools.product(models, batch_sizes, lrs, seeds, num_points, epochs, augment):
                tag = f"{suite_name}_{model}_bs{bs}_lr{lr}_np{npnts}_s{sd}"
                run_dir = run_root / tag
                ensure_dir(run_dir)

                cmd = ["python", "-u", "scripts/train_models.py"]
                args_map = {
                    "model": model, "epochs": ep, "batch_size": bs, "lr": lr, "seed": sd,
                    "num_points": npnts, "augment": aug, "data_dir": global_cfg.get("data_dir"),
                    "run_dir": str(run_dir)
                }
                for k, v in args_map.items():
                    if v is None or v == "": continue
                    flag = FLAG_MAP.get(k)
                    if flag: cmd += [flag, str(v)]

                log_path = run_dir / "train.log"
                print(f"[RUN] {tag}")
                print("      ", " ".join(cmd))
                with open(log_path, "w") as flog:
                    ret = subprocess.call(cmd, stdout=flog, stderr=subprocess.STDOUT, env=env_base)

                status = "OK" if ret == 0 else "FAIL"
                writer.writerow([suite_name, model, ep, bs, lr, sd, npnts, aug, str(run_dir), status, ret])

    print(f"\n✅ Terminado. Resumen: {summary_csv}")

if __name__ == "__main__":
    main()
