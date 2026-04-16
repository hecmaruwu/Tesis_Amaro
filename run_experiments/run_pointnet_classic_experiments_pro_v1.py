#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pointnet_classic_experiments_pro_v1.py

Runner "nivel pro" para barrer experimentos de PointNet clásico de forma secuencial,
guardar logs, resumir métricas, rankear resultados y elegir automáticamente
el mejor experimento.
"""

import csv
import json
import os
import shlex
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

PYTHON_BIN = "python3"
SCRIPT_PATH = "/home/htaucare/Tesis_Amaro/scripts_last_version/pointnet_classic_final_v8_patch.py"
GPU_ID = "0"

DATA_DIR = "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2"
INDEX_CSV = "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2/index_test.csv"

RUNS_ROOT = "/home/htaucare/Tesis_Amaro/outputs/pointnet_classic/grid_pro_runner_v1"
LOGS_ROOT = "/home/htaucare/Tesis_Amaro/outputs/pointnet_classic/grid_pro_runner_v1/logs"

SKIP_IF_DONE = True
CONTINUE_ON_ERROR = True

PRIMARY_SCORE_KEY = "test_filtered_d21_f1"
TIEBREAK_KEYS = [
    "test_filtered_f1_macro",
    "test_filtered_iou_macro",
    "test_d21_f1",
    "val_best_d21_f1",
]

BASE_ARGS: Dict[str, Any] = {
    "--data_dir": DATA_DIR,
    "--epochs": 120,
    "--batch_size": 16,
    "--lr": 2e-4,
    "--weight_decay": 1e-4,
    "--dropout": 0.5,
    "--num_workers": 6,
    "--infer_num_workers": 0,
    "--device": "cuda",
    "--use_amp": True,
    "--grad_clip": 1.0,
    "--d21_internal": 8,
    "--bg_index": 0,
    "--bg_weight": 0.03,
    "--neighbor_teeth": "d11:1,d22:9",
    "--neighbor_eval_split": "both",
    "--neighbor_every": 1,
    "--seed": 42,
    "--train_metrics_eval": True,
    "--do_infer": True,
    "--infer_split": "test",
    "--infer_examples": 20,
    "--index_csv": INDEX_CSV,
}

EXPERIMENT_GRID: List[Dict[str, Any]] = [
    {"name": "exp01_bs16_lr2e4_do05_bg003_amp", "--batch_size": 16, "--lr": 2e-4, "--dropout": 0.5, "--bg_weight": 0.03, "--use_amp": True},
    {"name": "exp02_bs16_lr1e4_do05_bg003_amp", "--batch_size": 16, "--lr": 1e-4, "--dropout": 0.5, "--bg_weight": 0.03, "--use_amp": True},
    {"name": "exp03_bs16_lr3e4_do05_bg003_amp", "--batch_size": 16, "--lr": 3e-4, "--dropout": 0.5, "--bg_weight": 0.03, "--use_amp": True},
    {"name": "exp04_bs16_lr2e4_do03_bg003_amp", "--batch_size": 16, "--lr": 2e-4, "--dropout": 0.3, "--bg_weight": 0.03, "--use_amp": True},
    {"name": "exp05_bs16_lr2e4_do06_bg003_amp", "--batch_size": 16, "--lr": 2e-4, "--dropout": 0.6, "--bg_weight": 0.03, "--use_amp": True},
    {"name": "exp06_bs16_lr2e4_do05_bg005_amp", "--batch_size": 16, "--lr": 2e-4, "--dropout": 0.5, "--bg_weight": 0.05, "--use_amp": True},
    {"name": "exp07_bs16_lr2e4_do05_bg001_amp", "--batch_size": 16, "--lr": 2e-4, "--dropout": 0.5, "--bg_weight": 0.01, "--use_amp": True},
    {"name": "exp08_bs8_lr2e4_do05_bg003_amp", "--batch_size": 8, "--lr": 2e-4, "--dropout": 0.5, "--bg_weight": 0.03, "--use_amp": True},
    {"name": "exp09_bs16_lr2e4_do05_bg003_noamp", "--batch_size": 16, "--lr": 2e-4, "--dropout": 0.5, "--bg_weight": 0.03, "--use_amp": False},
    {"name": "exp10_bs8_lr2e4_do05_bg003_noamp", "--batch_size": 8, "--lr": 2e-4, "--dropout": 0.5, "--bg_weight": 0.03, "--use_amp": False},
]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def stringify(v: Any) -> str:
    if isinstance(v, bool):
        return ""
    return str(v)

def build_command(exp_args: Dict[str, Any], out_dir: Path) -> List[str]:
    args = deepcopy(BASE_ARGS)
    args.update({k: v for k, v in exp_args.items() if k != "name"})
    args["--out_dir"] = str(out_dir)
    cmd = [PYTHON_BIN, "-u", SCRIPT_PATH]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(k)
        else:
            cmd.extend([k, stringify(v)])
    return cmd

def command_to_shell(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)

def experiment_done(out_dir: Path) -> bool:
    return (out_dir / "best.pt").exists() or (out_dir / "test_metrics.json").exists()

def read_best_val_from_runlog(run_log: Path):
    if not run_log.exists():
        return None, None
    try:
        lines = run_log.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines):
            if "best_epoch=" in line and "best_val_d21_f1=" in line:
                try:
                    best_epoch = int(line.split("best_epoch=")[1].split()[0])
                    best_val = float(line.split("best_val_d21_f1=")[1].split()[0])
                    return best_epoch, best_val
                except Exception:
                    pass
    except Exception:
        pass
    return None, None

def metric_or_none(d: Optional[dict], key: str) -> Optional[float]:
    if not isinstance(d, dict):
        return None
    try:
        v = d.get(key, None)
        return None if v is None else float(v)
    except Exception:
        return None

def sortable_value(v: Optional[float]) -> float:
    return -1e18 if v is None else float(v)

@dataclass
class ExperimentResult:
    name: str
    out_dir: str
    status: str
    return_code: Optional[int]
    elapsed_sec: Optional[float]
    command: str
    best_epoch: Optional[int] = None
    val_best_d21_f1: Optional[float] = None
    test_d21_f1: Optional[float] = None
    test_f1_macro: Optional[float] = None
    test_iou_macro: Optional[float] = None
    test_filtered_d21_f1: Optional[float] = None
    test_filtered_f1_macro: Optional[float] = None
    test_filtered_iou_macro: Optional[float] = None
    score: Optional[float] = None
    rank: Optional[int] = None

def collect_result(name: str, out_dir: Path, status: str, return_code: Optional[int], elapsed_sec: Optional[float], cmd_shell: str) -> ExperimentResult:
    test = load_json(out_dir / "test_metrics.json")
    test_f = load_json(out_dir / "test_metrics_filtered.json")
    best_epoch, best_val = read_best_val_from_runlog(out_dir / "run.log")
    res = ExperimentResult(
        name=name,
        out_dir=str(out_dir),
        status=status,
        return_code=return_code,
        elapsed_sec=elapsed_sec,
        command=cmd_shell,
        best_epoch=best_epoch,
        val_best_d21_f1=best_val,
        test_d21_f1=metric_or_none(test, "d21_f1"),
        test_f1_macro=metric_or_none(test, "f1_macro"),
        test_iou_macro=metric_or_none(test, "iou_macro"),
        test_filtered_d21_f1=metric_or_none(test_f, "d21_f1"),
        test_filtered_f1_macro=metric_or_none(test_f, "f1_macro"),
        test_filtered_iou_macro=metric_or_none(test_f, "iou_macro"),
    )
    res.score = getattr(res, PRIMARY_SCORE_KEY, None)
    return res

def rank_results(results: List[ExperimentResult]) -> List[ExperimentResult]:
    def key_fn(r: ExperimentResult):
        return tuple([sortable_value(getattr(r, PRIMARY_SCORE_KEY, None))] + [sortable_value(getattr(r, k, None)) for k in TIEBREAK_KEYS])
    ordered = sorted(results, key=key_fn, reverse=True)
    for i, r in enumerate(ordered, start=1):
        r.rank = i
    return ordered

def write_csv(results: List[ExperimentResult], out_csv: Path):
    rows = [asdict(r) for r in results]
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    ensure_dir(Path(RUNS_ROOT))
    ensure_dir(Path(LOGS_ROOT))
    all_results: List[ExperimentResult] = []

    for i, exp in enumerate(EXPERIMENT_GRID, start=1):
        name = exp["name"]
        out_dir = ensure_dir(Path(RUNS_ROOT) / name)
        log_path = Path(LOGS_ROOT) / f"{name}.log"
        cmd = build_command(exp, out_dir)
        cmd_shell = command_to_shell(cmd)

        print("=" * 100)
        print(f"[{i}/{len(EXPERIMENT_GRID)}] {name}")
        print(cmd_shell)
        print("=" * 100, flush=True)

        if SKIP_IF_DONE and experiment_done(out_dir):
            res = collect_result(name, out_dir, "skipped_existing", None, None, cmd_shell)
            all_results.append(res)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = GPU_ID
        env["PYTHONUNBUFFERED"] = "1"

        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(cmd_shell + "\n\n")
            logf.flush()
            try:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, text=True, check=False)
                elapsed = time.time() - t0
                status = "ok" if proc.returncode == 0 else "failed"
                res = collect_result(name, out_dir, status, proc.returncode, elapsed, cmd_shell)
                all_results.append(res)
                print(f"[{status.upper()}] {name} | return_code={proc.returncode} | elapsed={elapsed/3600:.2f} h", flush=True)
                if proc.returncode != 0 and not CONTINUE_ON_ERROR:
                    break
            except KeyboardInterrupt:
                elapsed = time.time() - t0
                res = collect_result(name, out_dir, "interrupted", -999, elapsed, cmd_shell)
                all_results.append(res)
                break
            except Exception as e:
                elapsed = time.time() - t0
                with log_path.open("a", encoding="utf-8") as logf2:
                    logf2.write(f"\n\n[RUNNER_EXCEPTION] {repr(e)}\n")
                res = collect_result(name, out_dir, f"runner_exception:{type(e).__name__}", -998, elapsed, cmd_shell)
                all_results.append(res)
                if not CONTINUE_ON_ERROR:
                    break

        ranked_partial = rank_results(all_results[:])
        save_json([asdict(r) for r in ranked_partial], Path(RUNS_ROOT) / "experiments_summary.json")
        write_csv(ranked_partial, Path(RUNS_ROOT) / "experiments_summary.csv")
        if ranked_partial:
            save_json(asdict(ranked_partial[0]), Path(RUNS_ROOT) / "best_experiment.json")

    ranked = rank_results(all_results[:])
    save_json([asdict(r) for r in ranked], Path(RUNS_ROOT) / "experiments_summary.json")
    write_csv(ranked, Path(RUNS_ROOT) / "experiments_summary.csv")
    if ranked:
        save_json(asdict(ranked[0]), Path(RUNS_ROOT) / "best_experiment.json")

    print("\n" + "=" * 100)
    print("RESUMEN FINAL")
    print("=" * 100)
    for r in ranked:
        print(f"rank={r.rank:02d} | {r.name} | status={r.status} | score={r.score} | val_best_d21_f1={r.val_best_d21_f1} | test_filtered_d21_f1={r.test_filtered_d21_f1} | test_d21_f1={r.test_d21_f1}")
    if ranked:
        print("\nMEJOR EXPERIMENTO")
        print(json.dumps(asdict(ranked[0]), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
