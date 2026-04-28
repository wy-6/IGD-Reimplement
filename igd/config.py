from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

# 若设置，则覆盖 config.yaml 中的 paths.output_dir（便于 Cloud Studio / 挂载数据盘等）
ENV_OUTPUT_DIR = "IGD_OUTPUT_DIR"


def load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = os.path.abspath(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return resolve_runtime_paths(cfg, config_path=cfg_path)


def resolve_runtime_paths(cfg: Dict[str, Any], *, config_path: Optional[str] = None) -> Dict[str, Any]:
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        return cfg

    output_dir = paths.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        return cfg

    env_out = os.environ.get(ENV_OUTPUT_DIR)
    if env_out is not None and str(env_out).strip():
        paths["output_dir"] = os.path.normpath(os.path.expanduser(str(env_out).strip()))
        return cfg

    config_dir = os.path.dirname(config_path) if config_path else os.getcwd()
    if os.path.isabs(output_dir):
        paths["output_dir"] = os.path.normpath(output_dir)
    else:
        paths["output_dir"] = os.path.normpath(os.path.join(config_dir, output_dir))
    return cfg


@dataclass
class RunOverrides:
    dataset_name: Optional[str] = None
    device: Optional[str] = None
    stage: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    ig_steps: Optional[int] = None


def deep_get(d: Dict[str, Any], keys: str, default=None):
    cur = d
    for k in keys.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_set(d: Dict[str, Any], keys: str, value: Any) -> None:
    parts = keys.split(".")
    cur = d
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], o: RunOverrides) -> Dict[str, Any]:
    if o.dataset_name:
        deep_set(cfg, "dataset.name", o.dataset_name)
    if o.ig_steps is not None:
        deep_set(cfg, "igd.ig_steps", int(o.ig_steps))
    # 这些会在入口脚本里直接使用（不一定写回 cfg）
    return cfg

