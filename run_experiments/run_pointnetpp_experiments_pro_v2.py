#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pointnetpp_experiments_pro_v2.py

Runner "nivel pro v2" para barrer experimentos de PointNet++ de forma secuencial,
guardar logs, resumir métricas, rankear resultados y reintentar automáticamente
si detecta OOM (CUDA out of memory).

Mejoras sobre v1:
- Detecta OOM leyendo el return code y/o el log
- Reintenta con batch más chico automáticamente
- Marca reintentos y razón del fallo
- Resume estado final real del experimento

Archivos generados:
- experiments_summary.csv
- experiments_summary.json
- best_experiment.json

Uso:
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u run_pointnetpp_experiments_pro_v2.py
"""

import csv
import json
import os
import re
import shlex
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


# ============================================================
# CONFIG
# ============================================================

PYTHON_BIN = "python3"
SCRIPT_PATH = "/home/htaucare/Tesis_Amaro/scripts_last_version/pointnetpp_classic_final_v1_patch.py"
GPU_ID = "1"

DATA_DIR = "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2"
INDEX_CSV = "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2/index_test.csv"

RUNS_ROOT = "/home/htaucare/Tesis_Amaro/outputs/pointnetpp/grid_pro_runner_v2"
LOGS_ROOT = "/home/htaucare/Tesis_Amaro/outputs/pointnetpp/grid_pro_runner_v2/logs"

SKIP_IF_DONE = True
CONTINUE_ON_ERROR = True

PRIMARY_SCORE_KEY = "test_filtered_d21_f1"
TIEBREAK_KEYS = [
    "test_filtered_f1_macro",
    "test_filtered_iou_macro",
    "test_d21_f1",
    "val_best_d21_f1",
]

# Reintentos automáticos por OOM
ENABLE_OOM_RETRY = True
OOM_BATCH_FALLBACKS = [8, 4, 2, 1]   # intentará batch menores que el original, en este orden
MAX_ATTEMPTS_PER_EXPERIMENT = 5

BASE_ARGS: Dict[str, Any] = {
    "--data_dir": DATA_DIR,
    "--epochs": 120,
    "--weight_decay": 1e-4,
    "--dropout": 0.5,
    "--num_workers": 6,
    "--infer_num_workers": 0,
    "--device": "cuda",
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
    {
        "name": "exp01_bs8_lr2e4_r010_020_040_ns32",
        "--batch_size": 8,
        "--lr": 2e-4,
        "--sa1_npoint": 1024, "--sa1_radius": 0.10, "--sa1_nsample": 32,
        "--sa2_npoint": 256,  "--sa2_radius": 0.20, "--sa2_nsample": 32,
        "--sa3_npoint": 64,   "--sa3_radius": 0.40, "--sa3_nsample": 32,
    },
    {
        "name": "exp02_bs8_lr2e4_r015_030_050_ns32",
        "--batch_size": 8,
        "--lr": 2e-4,
        "--sa1_npoint": 1024, "--sa1_radius": 0.15, "--sa1_nsample": 32,
        "--sa2_npoint": 256,  "--sa2_radius": 0.30, "--sa2_nsample": 32,
        "--sa3_npoint": 64,   "--sa3_radius": 0.50, "--sa3_nsample": 32,
    },
    {
        "name": "exp03_bs8_lr1e4_r010_020_040_ns32",
        "--batch_size": 8,
        "--lr": 1e-4,
        "--sa1_npoint": 1024, "--sa1_radius": 0.10, "--sa1_nsample": 32,
        "--sa2_npoint": 256,  "--sa2_radius": 0.20, "--sa2_nsample": 32,
        "--sa3_npoint": 64,   "--sa3_radius": 0.40, "--sa3_nsample": 32,
    },
    {
        "name": "exp04_bs8_lr3e4_r010_020_040_ns32",
        "--batch_size": 8,
        "--lr": 3e-4,
        "--sa1_npoint": 1024, "--sa1_radius": 0.10, "--sa1_nsample": 32,
        "--sa2_npoint": 256,  "--sa2_radius": 0.20, "--sa2_nsample": 32,
        "--sa3_npoint": 64,   "--sa3_radius": 0.40, "--sa3_nsample": 32,
    },
    {
        "name": "exp05_bs4_lr2e4_r010_020_040_ns32",
        "--batch_size": 4,
        "--lr": 2e-4,
        "--sa1_npoint": 1024, "--sa1_radius": 0.10, "--sa1_nsample": 32,
        "--sa2_npoint": 256,  "--sa2_radius": 0.20, "--sa2_nsample": 32,
        "--sa3_npoint": 64,   "--sa3_radius": 0.40, "--sa3_nsample": 32,
    },
]


# ============================================================
# HELPERS
# ============================================================

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
    return (
        (out_dir / "best.pt").exists()
        or (out_dir / "test_metrics.json").exists()
    )


def read_best_val_from_runlog(run_log: Path):
    if not run_log.exists():
        return None, None
    best_epoch = None
    best_val = None
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
    return best_epoch, best_val


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


def detect_oom_from_log(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore").lower()
        patterns = [
            "cuda out of memory",
            "outofmemoryerror",
            "torch.outofmemoryerror",
            "tried to allocate",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
        ]
        return any(p in txt for p in patterns)
    except Exception:
        return False


def next_smaller_batch(current_bs: int) -> Optional[int]:
    cands = [b for b in OOM_BATCH_FALLBACKS if b < current_bs]
    return cands[0] if cands else None


@dataclass
class ExperimentResult:
    name: str
    out_dir: str
    status: str
    return_code: Optional[int]
    elapsed_sec: Optional[float]
    command: str

    attempted_batch_size: Optional[int] = None
    retries: int = 0
    oom_detected: bool = False
    final_reason: Optional[str] = None

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


def collect_result(name: str, out_dir: Path, status: str, return_code: Optional[int],
                   elapsed_sec: Optional[float], cmd_shell: str,
                   attempted_batch_size: Optional[int], retries: int,
                   oom_detected: bool, final_reason: Optional[str]) -> ExperimentResult:
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
        attempted_batch_size=attempted_batch_size,
        retries=retries,
        oom_detected=oom_detected,
        final_reason=final_reason,
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
        return tuple(
            [sortable_value(getattr(r, PRIMARY_SCORE_KEY, None))]
            + [sortable_value(getattr(r, k, None)) for k in TIEBREAK_KEYS]
        )
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


def run_single_attempt(exp: Dict[str, Any], out_dir: Path, log_path: Path):
    cmd = build_command(exp, out_dir)
    cmd_shell = command_to_shell(cmd)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(cmd_shell + "\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    elapsed = time.time() - t0
    oom = detect_oom_from_log(log_path)
    return proc.returncode, elapsed, cmd_shell, oom


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(Path(RUNS_ROOT))
    ensure_dir(Path(LOGS_ROOT))

    all_results: List[ExperimentResult] = []

    for i, exp0 in enumerate(EXPERIMENT_GRID, start=1):
        exp = deepcopy(exp0)
        name = exp["name"]
        out_dir = ensure_dir(Path(RUNS_ROOT) / name)
        base_log_path = Path(LOGS_ROOT) / f"{name}.log"

        print("=" * 100)
        print(f"[{i}/{len(EXPERIMENT_GRID)}] {name}")
        print("=" * 100, flush=True)

        if SKIP_IF_DONE and experiment_done(out_dir):
            res = collect_result(
                name=name,
                out_dir=out_dir,
                status="skipped_existing",
                return_code=None,
                elapsed_sec=None,
                cmd_shell="(existing)",
                attempted_batch_size=exp.get("--batch_size"),
                retries=0,
                oom_detected=False,
                final_reason="existing_outputs",
            )
            all_results.append(res)
            ranked_partial = rank_results(all_results[:])
            save_json([asdict(r) for r in ranked_partial], Path(RUNS_ROOT) / "experiments_summary.json")
            write_csv(ranked_partial, Path(RUNS_ROOT) / "experiments_summary.csv")
            save_json(asdict(ranked_partial[0]), Path(RUNS_ROOT) / "best_experiment.json")
            continue

        retries = 0
        final_result = None
        attempted_bs = int(exp.get("--batch_size", 8))
        last_reason = None
        oom_detected_any = False

        while retries < MAX_ATTEMPTS_PER_EXPERIMENT:
            log_path = base_log_path if retries == 0 else Path(LOGS_ROOT) / f"{name}.retry{retries}.log"
            print(f"[RUN] {name} | attempt={retries+1} | batch_size={attempted_bs}", flush=True)

            try:
                return_code, elapsed, cmd_shell, oom_detected = run_single_attempt(exp, out_dir, log_path)
                oom_detected_any = oom_detected_any or oom_detected

                if return_code == 0:
                    final_result = collect_result(
                        name=name,
                        out_dir=out_dir,
                        status="ok" if retries == 0 else "ok_after_retry",
                        return_code=return_code,
                        elapsed_sec=elapsed,
                        cmd_shell=cmd_shell,
                        attempted_batch_size=attempted_bs,
                        retries=retries,
                        oom_detected=oom_detected_any,
                        final_reason="success",
                    )
                    break

                # fracaso
                if ENABLE_OOM_RETRY and oom_detected:
                    smaller = next_smaller_batch(attempted_bs)
                    if smaller is not None:
                        retries += 1
                        exp["--batch_size"] = smaller
                        attempted_bs = smaller
                        last_reason = f"oom_retry_to_bs{smaller}"
                        print(f"[OOM] {name} -> reintentando con batch_size={smaller}", flush=True)
                        continue
                    else:
                        final_result = collect_result(
                            name=name,
                            out_dir=out_dir,
                            status="failed_oom",
                            return_code=return_code,
                            elapsed_sec=elapsed,
                            cmd_shell=cmd_shell,
                            attempted_batch_size=attempted_bs,
                            retries=retries,
                            oom_detected=True,
                            final_reason="oom_no_smaller_batch_available",
                        )
                        break
                else:
                    final_result = collect_result(
                        name=name,
                        out_dir=out_dir,
                        status="failed",
                        return_code=return_code,
                        elapsed_sec=elapsed,
                        cmd_shell=cmd_shell,
                        attempted_batch_size=attempted_bs,
                        retries=retries,
                        oom_detected=oom_detected_any,
                        final_reason="non_oom_failure",
                    )
                    break

            except KeyboardInterrupt:
                final_result = collect_result(
                    name=name,
                    out_dir=out_dir,
                    status="interrupted",
                    return_code=-999,
                    elapsed_sec=None,
                    cmd_shell="(interrupted)",
                    attempted_batch_size=attempted_bs,
                    retries=retries,
                    oom_detected=oom_detected_any,
                    final_reason="keyboard_interrupt",
                )
                all_results.append(final_result)
                ranked_partial = rank_results(all_results[:])
                save_json([asdict(r) for r in ranked_partial], Path(RUNS_ROOT) / "experiments_summary.json")
                write_csv(ranked_partial, Path(RUNS_ROOT) / "experiments_summary.csv")
                save_json(asdict(ranked_partial[0]), Path(RUNS_ROOT) / "best_experiment.json")
                print("[INTERRUPTED] Ejecución interrumpida por usuario.", flush=True)
                return

            except Exception as e:
                final_result = collect_result(
                    name=name,
                    out_dir=out_dir,
                    status=f"runner_exception:{type(e).__name__}",
                    return_code=-998,
                    elapsed_sec=None,
                    cmd_shell="(runner_exception)",
                    attempted_batch_size=attempted_bs,
                    retries=retries,
                    oom_detected=oom_detected_any,
                    final_reason=repr(e),
                )
                if not CONTINUE_ON_ERROR:
                    all_results.append(final_result)
                    break
                break

        if final_result is None:
            final_result = collect_result(
                name=name,
                out_dir=out_dir,
                status="failed_unknown",
                return_code=-997,
                elapsed_sec=None,
                cmd_shell="(unknown)",
                attempted_batch_size=attempted_bs,
                retries=retries,
                oom_detected=oom_detected_any,
                final_reason=last_reason or "unknown",
            )

        all_results.append(final_result)

        ranked_partial = rank_results(all_results[:])
        save_json([asdict(r) for r in ranked_partial], Path(RUNS_ROOT) / "experiments_summary.json")
        write_csv(ranked_partial, Path(RUNS_ROOT) / "experiments_summary.csv")
        if ranked_partial:
            save_json(asdict(ranked_partial[0]), Path(RUNS_ROOT) / "best_experiment.json")

        if final_result.status.startswith("failed") and not CONTINUE_ON_ERROR:
            print("[STOP] Se detuvo la grilla por error.", flush=True)
            break

    ranked = rank_results(all_results[:])
    save_json([asdict(r) for r in ranked], Path(RUNS_ROOT) / "experiments_summary.json")
    write_csv(ranked, Path(RUNS_ROOT) / "experiments_summary.csv")
    if ranked:
        save_json(asdict(ranked[0]), Path(RUNS_ROOT) / "best_experiment.json")

    print("\n" + "=" * 100)
    print("RESUMEN FINAL")
    print("=" * 100)
    for r in ranked:
        print(
            f"rank={r.rank:02d} | {r.name} | status={r.status} | "
            f"bs={r.attempted_batch_size} | retries={r.retries} | oom={r.oom_detected} | "
            f"score={r.score} | test_filtered_d21_f1={r.test_filtered_d21_f1} | "
            f"test_d21_f1={r.test_d21_f1} | val_best_d21_f1={r.val_best_d21_f1}"
        )

    if ranked:
        print("\nMEJOR EXPERIMENTO")
        print(json.dumps(asdict(ranked[0]), indent=2, ensure_ascii=False))
        print(f"\nArchivo resumen JSON: {Path(RUNS_ROOT) / 'experiments_summary.json'}")
        print(f"Archivo resumen CSV : {Path(RUNS_ROOT) / 'experiments_summary.csv'}")
        print(f"Archivo best JSON   : {Path(RUNS_ROOT) / 'best_experiment.json'}")


if __name__ == "__main__":
    main()
