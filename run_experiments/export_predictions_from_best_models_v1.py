#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_predictions_from_best_models_v1.py

Exporta predicciones por punto desde modelos YA ENTRENADOS, sin reentrenar.

Objetivo:
  best.pt + data_dir + index_test.csv
  -> inference/predictions/*_pred.npy
  -> inference/predictions/*_gt.npy
  -> inference/predictions/*_xyz.npy
  -> actualiza/crea inference_manifest.csv con columnas pred_npy/gt_npy/xyz_npy

Uso directo:
  python3 -u export_predictions_from_best_models_v1.py --device cuda --split test --max_examples 20

Exportar todo el test:
  python3 -u export_predictions_from_best_models_v1.py --device cuda --split test --max_examples -1
"""

import os, re, csv, json, time, inspect, argparse, importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEFAULT_MODELS = [
    {
        "name": "pointnet",
        "model_type": "pointnet",
        "script_path": "/home/htaucare/Tesis_Amaro/scripts_last_version/pointnet_classic_final_v8_patch.py",
        "out_dir": "/home/htaucare/Tesis_Amaro/outputs/pointnet_classic/grid_pro_runner_v1/exp03_bs16_lr3e4_do05_bg003_amp",
        "data_dir": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2",
        "index_csv": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2/index_test.csv",
        "class_name": "PointNetSeg",
        "extra_kwargs": {"dropout": 0.5},
    },
    {
        "name": "pointnetpp",
        "model_type": "pointnetpp",
        "script_path": "/home/htaucare/Tesis_Amaro/scripts_last_version/pointnetpp_classic_final_v1_patch.py",
        "out_dir": "/home/htaucare/Tesis_Amaro/outputs/pointnetpp/grid_pro_runner_v2/exp04_bs8_lr3e4_r010_020_040_ns32",
        "data_dir": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2",
        "index_csv": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2/index_test.csv",
        "class_name": "",
        "extra_kwargs": {
            "dropout": 0.5,
            "sa1_npoint": 1024, "sa1_radius": 0.10, "sa1_nsample": 32,
            "sa2_npoint": 256, "sa2_radius": 0.20, "sa2_nsample": 32,
            "sa3_npoint": 64, "sa3_radius": 0.40, "sa3_nsample": 32,
        },
    },
    {
        "name": "dgcnn",
        "model_type": "dgcnn",
        "script_path": "/home/htaucare/Tesis_Amaro/scripts_last_version/train_dgcnn_classic_only_fixed_v9_patch.py",
        "out_dir": "/home/htaucare/Tesis_Amaro/outputs/dgcnn/grid_pro_runner_v2_gpu1/exp06_bs8_lr2e4_k20_emb768_bg003",
        "data_dir": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2",
        "index_csv": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2/index_test.csv",
        "class_name": "",
        "extra_kwargs": {"dropout": 0.5, "k": 20, "emb_dims": 768, "knn_chunk_size": 1024},
    },
    {
        "name": "pointnettransformer",
        "model_type": "transformer",
        "script_path": "/home/htaucare/Tesis_Amaro/scripts_last_version/pointnettransformer_classic_final_v5_patch.py",
        "out_dir": "/home/htaucare/Tesis_Amaro/outputs/pointnettransformer/grid_pro_runner_v2_gpu0/exp13_bs4_lr2e4_dm256_dep4_h8_ff512_do005",
        "data_dir": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2",
        "index_csv": "/home/htaucare/Tesis_Amaro/data/Teeth_3ds/fixed_split/8192/upper_only_surf_global_excl_wisdom_seed42_aug2/index_test.csv",
        "class_name": "",
        "extra_kwargs": {
            "d_model": 256, "embed_hidden": 128, "depth": 4, "nhead": 8,
            "dim_feedforward": 512, "dropout": 0.05, "head_hidden": 256,
            "head_dropout": 0.05, "use_pos_mlp": True, "pos_hidden": 128,
            "norm_first": True, "activation": "gelu",
            "token_subsample": 2048, "prop_k": 3, "prop_chunk": 2048,
        },
    },
]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_unit_sphere_np(xyz: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    c = xyz.mean(axis=0, keepdims=True)
    x = xyz - c
    r = max(float(np.linalg.norm(x, axis=1).max()), eps)
    return (x / r).astype(np.float32)


class NPZInferDataset(Dataset):
    def __init__(self, data_dir: Path, split: str = "test", normalize: bool = True):
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.X = np.load(self.data_dir / f"X_{self.split}.npz")["X"].astype(np.float32)
        self.Y = np.load(self.data_dir / f"Y_{self.split}.npz")["Y"].astype(np.int64)
        assert self.X.ndim == 3 and self.X.shape[-1] == 3, self.X.shape
        assert self.Y.ndim == 2, self.Y.shape
        assert self.X.shape[:2] == self.Y.shape[:2], (self.X.shape, self.Y.shape)
        self.normalize = bool(normalize)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, i: int):
        x = np.ascontiguousarray(self.X[int(i)], dtype=np.float32)
        y = np.ascontiguousarray(self.Y[int(i)], dtype=np.int64)
        if self.normalize:
            x = normalize_unit_sphere_np(x)
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(int(i), dtype=torch.long)


def make_loader(data_dir: Path, split: str, batch_size: int, num_workers: int, normalize: bool):
    ds = NPZInferDataset(data_dir=data_dir, split=split, normalize=normalize)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
                      pin_memory=True, persistent_workers=(int(num_workers) > 0), drop_last=False)


def infer_num_classes(data_dir: Path) -> int:
    vals = []
    for sp in ("train", "val", "test"):
        p = Path(data_dir) / f"Y_{sp}.npz"
        if p.exists():
            vals.append(int(np.load(p)["Y"].max()))
    if not vals:
        raise FileNotFoundError(f"No encontré Y_train/val/test en {data_dir}")
    return int(max(vals)) + 1


def read_index_csv(index_csv: Optional[Path]) -> Dict[int, Dict[str, str]]:
    if index_csv is None or not Path(index_csv).exists():
        return {}
    with open(index_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        fields = {h.strip(): h for h in reader.fieldnames}
        def pick(*cands):
            for c in cands:
                if c in fields:
                    return fields[c]
            return None
        row_key = pick("row_i", "row", "i", "idx", "index")
        name_key = pick("sample_name", "sample", "name", "patient")
        jaw_key = pick("jaw", "arch")
        path_key = pick("path", "file_path")
        out = {}
        for r in reader:
            try:
                ri = int(r[row_key]) if row_key else len(out)
            except Exception:
                ri = len(out)
            out[ri] = {
                "sample_name": r.get(name_key, "") if name_key else "",
                "jaw": r.get(jaw_key, "") if jaw_key else "",
                "path": r.get(path_key, "") if path_key else "",
            }
        return out


def sanitize_tag(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:int(maxlen)] if len(s) > int(maxlen) else s


def read_existing_manifest(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out = {}
        for r in reader:
            try:
                out[int(r["row_i"])] = dict(r)
            except Exception:
                continue
        return out


def write_manifest(path: Path, rows: List[Dict[str, Any]]):
    ensure_dir(path.parent)
    base = ["row_i", "tag", "sample_name", "jaw", "path", "png_all", "png_errors", "png_d21", "pred_npy", "gt_npy", "xyz_npy"]
    extra = []
    for r in rows:
        for k in r.keys():
            if k not in base and k not in extra:
                extra.append(k)
    fields = base + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def import_module_from_path(script_path: Path):
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"No existe script_path: {script_path}")
    module_name = "export_dyn_" + re.sub(r"[^a-zA-Z0-9_]+", "_", script_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No pude importar: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_nn_classes(module):
    out = {}
    for name, obj in vars(module).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            out[name] = obj
    return out


def candidate_class_names(model_type: str) -> List[str]:
    mt = str(model_type).lower()
    if mt == "pointnet":
        return ["PointNetSeg", "PointNetSegmentation", "PointNetDenseCls"]
    if mt == "pointnetpp":
        return ["PointNet2Seg", "PointNetPPSeg", "PointNetPPSegNet", "PointNetPP",
                "PointNet2SemSeg", "PointNet2ClassicSeg", "PointNetPlusPlusSeg",
                "PointNet2Segmentation", "PointNetPPClassic"]
    if mt == "dgcnn":
        return ["DGCNNSeg", "DGCNN", "DGCNNSemSeg", "DGCNNPartSeg", "DGCNNClassicSeg", "DGCNNSegmentation"]
    if mt in ("transformer", "pointnettransformer", "pointtransformer"):
        return ["PointNetTransformerSeg", "PointNetTransformer", "PointNetTransformerClassic",
                "PointNetTransformerSegModel", "PointTransformerSeg", "PointTransformer",
                "TransformerSeg", "Transformer3DSeg", "Transformer3D"]
    return []


def load_run_meta(out_dir: Path) -> Dict[str, Any]:
    for p in [out_dir / "run_meta.json", out_dir / "runner_meta.json", out_dir / "meta.json"]:
        if p.exists():
            obj = load_json(p, default={})
            if isinstance(obj, dict):
                return obj
    return {}


def build_param_pool(C: int, cfg: Dict[str, Any], run_meta: Dict[str, Any]) -> Dict[str, Any]:
    pool = {}
    if isinstance(run_meta, dict):
        pool.update(run_meta)
        if isinstance(run_meta.get("args"), dict):
            pool.update(run_meta["args"])
    pool.update(cfg.get("extra_kwargs") or {})
    pool.update({
        "C": C, "num_classes": C, "n_classes": C, "classes": C,
        "num_class": C, "num_cls": C, "out_channels": C,
        "seg_classes": C, "output_channels": C,
    })
    pool.setdefault("dropout", 0.5)
    pool.setdefault("k", 20)
    pool.setdefault("emb_dims", 1024)
    pool.setdefault("knn_chunk_size", 1024)
    return pool


def try_instantiate(cls, pool: Dict[str, Any]) -> Tuple[bool, Optional[nn.Module], str]:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    kwargs = {}
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    aliases_map = {
        "num_classes": ["C", "n_classes", "num_class", "classes", "seg_classes"],
        "n_classes": ["C", "num_classes", "num_class", "classes", "seg_classes"],
        "classes": ["C", "num_classes", "n_classes", "seg_classes"],
        "c": ["C", "num_classes"],
        "dim_ff": ["dim_feedforward"],
        "dim_feedforward": ["dim_ff"],
        "emb_dim": ["emb_dims", "d_model"],
        "embed_dim": ["emb_dims", "d_model"],
    }
    for name, p in params.items():
        if name == "self":
            continue
        if name in pool:
            kwargs[name] = pool[name]
            continue
        found = False
        for a in aliases_map.get(name, []):
            if a in pool:
                kwargs[name] = pool[a]
                found = True
                break
        if found:
            continue
        if p.default is inspect.Parameter.empty and not accepts_var_kw:
            return False, None, f"falta parámetro requerido '{name}'"
    try:
        return True, cls(**kwargs), f"OK kwargs={kwargs}"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e} | kwargs={kwargs}"


def instantiate_model(module, model_type: str, class_name: str, C: int, cfg: Dict[str, Any], run_meta: Dict[str, Any]):
    classes = module_nn_classes(module)
    pool = build_param_pool(C=C, cfg=cfg, run_meta=run_meta)
    ordered = []
    if class_name:
        ordered.append(class_name)
    for name in candidate_class_names(model_type):
        if name not in ordered:
            ordered.append(name)
    mt = str(model_type).lower()
    for name in classes:
        low = name.lower()
        if mt == "pointnetpp" and ("pointnet" in low and ("pp" in low or "2" in low or "plus" in low)):
            if name not in ordered: ordered.append(name)
        elif mt == "dgcnn" and "dgcnn" in low:
            if name not in ordered: ordered.append(name)
        elif mt in ("transformer", "pointnettransformer", "pointtransformer") and "transform" in low:
            if name not in ordered: ordered.append(name)
        elif mt == "pointnet" and low == "pointnetseg":
            if name not in ordered: ordered.append(name)
    for name in classes:
        if name not in ordered:
            ordered.append(name)
    errors = []
    for name in ordered:
        if name not in classes:
            errors.append(f"{name}: no existe en módulo")
            continue
        ok, model, msg = try_instantiate(classes[name], pool)
        if ok and model is not None:
            return model, name, msg
        errors.append(f"{name}: {msg}")
    available = ", ".join(classes.keys())
    raise RuntimeError("No pude instanciar modelo.\n"
                       f"model_type={model_type}, class_name={class_name}\n"
                       f"Clases disponibles: {available}\n"
                       "Intentos:\n  - " + "\n  - ".join(errors))


def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        nk = str(k)
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            new_state[nk[len("model."):]] = v
        else:
            new_state[nk] = v
    try:
        model.load_state_dict(new_state, strict=True)
        return "strict=True"
    except Exception as e1:
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        return f"strict=False | missing={len(missing)} unexpected={len(unexpected)} | strict_error={type(e1).__name__}: {e1}"


def logits_to_pred(logits: torch.Tensor, N: int, C: int) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits debe ser 3D, llegó {tuple(logits.shape)}")
    if logits.shape[1] == N and logits.shape[2] == C:
        return logits.argmax(dim=-1)
    if logits.shape[1] == C and logits.shape[2] == N:
        return logits.argmax(dim=1)
    if logits.shape[-1] <= 512:
        return logits.argmax(dim=-1)
    raise ValueError(f"No reconozco forma logits={tuple(logits.shape)} para N={N}, C={C}")


def export_one_model(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    name = cfg["name"]
    out_dir = Path(cfg["out_dir"]).resolve()
    data_dir = Path(cfg["data_dir"]).resolve()
    script_path = Path(cfg["script_path"]).resolve()
    ckpt_path = Path(cfg.get("ckpt_path") or (out_dir / "best.pt")).resolve()
    index_csv = Path(cfg["index_csv"]).resolve() if cfg.get("index_csv") else None
    model_type = cfg.get("model_type", "")
    class_name = cfg.get("class_name", "")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"[{name}] No existe checkpoint: {ckpt_path}")

    C = infer_num_classes(data_dir)
    run_meta = load_run_meta(out_dir)
    normalize = bool(cfg.get("normalize", run_meta.get("normalize_unit_sphere", True)))
    if args.no_normalize:
        normalize = False
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"\n{'='*90}")
    print(f"[MODEL] {name}")
    print(f" script_path={script_path}")
    print(f" out_dir={out_dir}")
    print(f" ckpt={ckpt_path}")
    print(f" data_dir={data_dir}")
    print(f" C={C} normalize={normalize} device={device}")
    print(f"{'='*90}", flush=True)

    module = import_module_from_path(script_path)
    model, used_class, inst_msg = instantiate_model(module, model_type, class_name, C, cfg, run_meta)
    print(f"[{name}] clase usada: {used_class}")
    print(f"[{name}] instantiate: {inst_msg}")

    model = model.to(device)
    load_msg = load_checkpoint_into_model(model, ckpt_path, device)
    print(f"[{name}] checkpoint load: {load_msg}")

    loader = make_loader(data_dir=data_dir, split=args.split, batch_size=args.batch_size,
                         num_workers=args.num_workers, normalize=normalize)
    index_map = read_index_csv(index_csv)
    inf_root = ensure_dir(out_dir / "inference")
    pred_dir = ensure_dir(inf_root / "predictions")
    manifest_path = inf_root / "inference_manifest.csv"
    existing_manifest = read_existing_manifest(manifest_path)

    ignored_rows, exported = [], []
    max_examples = int(args.max_examples)
    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for xyz, y, row_i in loader:
            xyz = xyz.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            row_i_np = row_i.cpu().numpy().astype(int).tolist()

            logits = model(xyz)
            pred = logits_to_pred(logits, N=xyz.shape[1], C=C)

            for b, ri in enumerate(row_i_np):
                y_np = y[b].detach().cpu().numpy().astype(np.int64)
                if args.skip_only_bg:
                    vals = np.unique(y_np)
                    if len(vals) == 1 and int(vals[0]) == int(args.bg_index):
                        ignored_rows.append(int(ri))
                        continue

                if max_examples >= 0 and len(exported) >= max_examples:
                    break

                xyz_np = xyz[b].detach().cpu().numpy().astype(np.float32)
                pr_np = pred[b].detach().cpu().numpy().astype(np.int64)
                meta = index_map.get(int(ri), {})
                sample = meta.get("sample_name", "")
                jaw = meta.get("jaw", "")
                src_path = meta.get("path", "")

                tag = f"{args.split}_row{int(ri)}"
                if sample:
                    tag += "_" + sanitize_tag(sample)

                p_pred = pred_dir / f"{tag}_pred.npy"
                p_gt = pred_dir / f"{tag}_gt.npy"
                p_xyz = pred_dir / f"{tag}_xyz.npy"
                np.save(p_pred, pr_np)
                np.save(p_gt, y_np)
                np.save(p_xyz, xyz_np)

                row = existing_manifest.get(int(ri), {})
                row.update({
                    "row_i": int(ri),
                    "tag": row.get("tag", tag) or tag,
                    "sample_name": row.get("sample_name", sample) or sample,
                    "jaw": row.get("jaw", jaw) or jaw,
                    "path": row.get("path", src_path) or src_path,
                    "pred_npy": str(p_pred.relative_to(inf_root)),
                    "gt_npy": str(p_gt.relative_to(inf_root)),
                    "xyz_npy": str(p_xyz.relative_to(inf_root)),
                })
                row.setdefault("png_all", "")
                row.setdefault("png_errors", "")
                row.setdefault("png_d21", "")
                existing_manifest[int(ri)] = row
                exported.append(int(ri))

            if max_examples >= 0 and len(exported) >= max_examples:
                break

    rows = [existing_manifest[k] for k in sorted(existing_manifest.keys())]
    write_manifest(manifest_path, rows)

    summary = {
        "name": name, "model_type": model_type, "used_class": used_class,
        "script_path": str(script_path), "out_dir": str(out_dir), "ckpt_path": str(ckpt_path),
        "data_dir": str(data_dir), "index_csv": str(index_csv) if index_csv else "",
        "split": args.split, "C": int(C), "normalize": bool(normalize), "device": str(device),
        "n_exported": int(len(exported)), "exported_rows": exported,
        "n_ignored_only_bg": int(len(ignored_rows)), "ignored_rows": ignored_rows,
        "predictions_dir": str(pred_dir), "manifest_path": str(manifest_path),
        "elapsed_sec": float(time.time() - t0),
    }
    save_json(summary, inf_root / "predictions_export_summary.json")
    print(f"[{name}] OK exportados={len(exported)} ignored_only_bg={len(ignored_rows)}")
    print(f"[{name}] predictions_dir={pred_dir}")
    return summary


def load_configs(args) -> List[Dict[str, Any]]:
    if args.config_json:
        p = Path(args.config_json)
        obj = load_json(p)
        if isinstance(obj, dict) and "models" in obj:
            return obj["models"]
        if isinstance(obj, list):
            return obj
        raise ValueError("config_json debe ser una lista o {'models': [...]}")
    cfgs = DEFAULT_MODELS
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cfgs = [c for c in cfgs if c["name"] in wanted]
    return cfgs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_json", type=str, default="", help="JSON opcional con lista de modelos.")
    ap.add_argument("--only", type=str, default="", help="pointnet,pointnetpp,dgcnn,pointnettransformer")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_examples", type=int, default=20, help="-1 exporta todos los casos válidos.")
    ap.add_argument("--bg_index", type=int, default=0)
    ap.add_argument("--skip_only_bg", action="store_true", default=True)
    ap.add_argument("--include_only_bg", action="store_true")
    ap.add_argument("--no_normalize", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.include_only_bg:
        args.skip_only_bg = False
    cfgs = load_configs(args)
    if not cfgs:
        raise RuntimeError("No hay modelos para exportar.")
    all_summaries, errors = [], []
    for cfg in cfgs:
        try:
            all_summaries.append(export_one_model(cfg, args))
        except Exception as e:
            msg = {"name": cfg.get("name", "unknown"), "error_type": type(e).__name__, "error": str(e)}
            errors.append(msg)
            print(f"\n[ERROR] {msg['name']}: {msg['error_type']}: {msg['error']}", flush=True)
    out_global = Path("/home/htaucare/Tesis_Amaro/outputs/predictions_export_summary_all_models.json")
    save_json({"summaries": all_summaries, "errors": errors}, out_global)
    print("\n" + "="*90)
    print(f"FINAL: ok={len(all_summaries)} errors={len(errors)}")
    print(f"summary={out_global}")
    print("="*90)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
