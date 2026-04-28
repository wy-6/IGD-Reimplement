from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from dataclasses import asdict, is_dataclass
from pathlib import PurePath
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_timestamped_dir(parent_dir: str, prefix: Optional[str] = None) -> str:
    ensure_dir(parent_dir)
    stem = timestamp_now()
    name = f"{prefix}_{stem}" if prefix else stem
    path = os.path.join(parent_dir, name)
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(parent_dir, f"{name}_{suffix}")
        suffix += 1
    ensure_dir(path)
    return path


def _extract_checkpoint_sort_key(name: str):
    match = re.fullmatch(r"ckpt_(\d{8}_\d{6})(?:_(\d+))?", name)
    if not match:
        return None
    ts_key = match.group(1)
    suffix = int(match.group(2) or 0)
    return (ts_key, suffix)


def _relative_path_parts_after_outputs(path: str) -> Optional[List[str]]:
    parts = list(PurePath(os.path.normpath(os.path.expanduser(path))).parts)
    try:
        idx = parts.index("outputs")
    except ValueError:
        return None
    rest = parts[idx + 1 :]
    return list(rest) if rest else None


def remap_legacy_output_path(stored_path: str, output_root: str) -> str:
    """
    将旧环境（如 Kaggle）下的绝对路径映射到当前 output_root 下对应子路径，
    便于仓库内 outputs/ 断点续跑。
    """
    expanded = os.path.normpath(os.path.expanduser(stored_path))
    if os.path.isfile(os.path.join(expanded, "pytorch_model.bin")):
        return expanded

    root = os.path.normpath(os.path.expanduser(output_root))
    rel_parts = _relative_path_parts_after_outputs(expanded)
    if rel_parts is not None:
        candidate = os.path.normpath(os.path.join(root, *rel_parts))
        if os.path.isfile(os.path.join(candidate, "pytorch_model.bin")):
            return candidate

    return expanded


def resolve_latest_artifact_dir(
    base_dir: str,
    artifact_name: str = "pytorch_model.bin",
    *,
    remap_output_root: Optional[str] = None,
) -> Optional[str]:
    direct_path = os.path.join(base_dir, artifact_name)
    if os.path.exists(direct_path):
        return base_dir

    if not os.path.isdir(base_dir):
        pass
    else:
        named_candidates = []
        fallback_candidates = []
        for entry in os.scandir(base_dir):
            if not entry.is_dir():
                continue
            artifact_path = os.path.join(entry.path, artifact_name)
            if os.path.exists(artifact_path):
                sort_key = _extract_checkpoint_sort_key(entry.name)
                if sort_key is not None:
                    named_candidates.append((sort_key, entry.path))
                else:
                    fallback_candidates.append((os.path.getmtime(artifact_path), entry.path))

        if named_candidates:
            named_candidates.sort()
            return named_candidates[-1][1]
        if fallback_candidates:
            fallback_candidates.sort()
            return fallback_candidates[-1][1]

    meta_path = os.path.join(base_dir, "latest_checkpoint.json")
    if remap_output_root and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
        stored = meta.get("latest_checkpoint_dir")
        if isinstance(stored, str):
            mapped = remap_legacy_output_path(stored, remap_output_root)
            if os.path.isfile(os.path.join(mapped, artifact_name)):
                return mapped

    return None


def to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=to_jsonable)


def batch_iter(it: Iterable[Any], batch_size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

