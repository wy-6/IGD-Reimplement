from __future__ import annotations

import argparse
import os

import datasets
import torch

from igd.config import RunOverrides, apply_overrides, load_yaml
from igd.data import format_label_distribution, get_dataset_splits, load_dataset_and_tokenizer, shuffle_and_select_subset
from igd.model import IGDModel
from igd.pseudo import generate_pseudo_lines
from igd.train import load_baseline_weights, train_baseline, train_igd
from igd.utils import resolve_device, resolve_latest_artifact_dir, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--stage", type=str, choices=["baseline", "eval_baseline", "igd"], required=True)
    p.add_argument("--dataset", type=str, default=None, help="ag_news | imdb")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--ig_steps", type=int, default=None, help="仅用于 CPU smoke 时覆盖配置")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, RunOverrides(dataset_name=args.dataset, ig_steps=args.ig_steps))

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)
    seed = int(cfg.get("seed", 42))

    loaded = load_dataset_and_tokenizer(cfg, max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples)
    train_ds, eval_ds = get_dataset_splits(loaded)

    model = IGDModel(cfg, num_labels=loaded.num_labels)

    if args.stage in {"baseline", "eval_baseline"}:
        saved_dir = train_baseline(
            cfg=cfg,
            model=model,
            tokenizer=loaded.tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            data_collator=loaded.data_collator,
            device=device,
            max_length=int(cfg["dataset"].get("max_length", 128)),
            checkpoint_stage=args.stage,
        )
        print(f"saved {args.stage} checkpoint: {saved_dir}")
        return

    baseline_stage_dir = os.path.join(cfg["paths"]["output_dir"], cfg["dataset"]["name"], "baseline")
    baseline_dir = resolve_latest_artifact_dir(
        baseline_stage_dir,
        remap_output_root=cfg["paths"]["output_dir"],
    )
    if baseline_dir is None:
        raise FileNotFoundError(f"未找到 baseline checkpoint：{baseline_stage_dir}，请先运行 run_train.py --stage baseline")

    load_baseline_weights(model, baseline_dir)
    model.to(device)

    raw = datasets.load_dataset(cfg["dataset"]["name"])
    max_train = args.max_train_samples
    if max_train is None:
        default_map = cfg["dataset"].get("default_max_train_samples", {}) or {}
        if isinstance(default_map, dict):
            v = default_map.get(cfg["dataset"]["name"])
            if v is not None:
                max_train = int(v)
    text_key = cfg["dataset"].get("text_key") or ("text" if "text" in raw["train"].column_names else raw["train"].column_names[0])
    label_key = cfg["dataset"].get("label_key", "label")
    raw["train"] = shuffle_and_select_subset(raw["train"], max_train, seed=seed)
    print(f"[data] igd pseudo subset label distribution: {format_label_distribution(raw['train'], label_key)}")

    ig_steps = int(cfg["igd"].get("ig_steps", 50)) if args.ig_steps is None else int(args.ig_steps)
    pseudo_lines = generate_pseudo_lines(
        cfg=cfg,
        model=model,
        tokenizer=loaded.tokenizer,
        train_dataset=raw["train"],
        text_key=text_key,
        label_key=label_key,
        device=device,
        ig_steps=ig_steps,
    )

    saved_dir = train_igd(
        cfg=cfg,
        model=model,
        tokenizer=loaded.tokenizer,
        pseudo_lines=pseudo_lines,
        eval_ds=eval_ds,
        data_collator=loaded.data_collator,
        device=device,
        max_length=int(cfg["dataset"].get("max_length", 128)),
    )
    print(f"loaded baseline checkpoint: {baseline_dir}")
    print(f"saved igd checkpoint: {saved_dir}")


if __name__ == "__main__":
    main()
