from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import torch

from igd.attack_eval import run_attack_eval
from igd.config import RunOverrides, apply_overrides, load_yaml
from igd.data import load_dataset_and_tokenizer
from igd.model import IGDModel
from igd.utils import (
    remap_legacy_output_path,
    resolve_device,
    resolve_latest_artifact_dir,
    set_seed,
    timestamp_now,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--attacks", nargs="+", default=["textfooler", "textbugger"], help="支持 textfooler | textbugger | bertattack")
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--checkpoint_stage", type=str, choices=["baseline", "igd"], default="igd", help="加载 baseline 或 igd checkpoint")
    p.add_argument("--eval_stage", type=str, choices=["baseline", "igd"], default=None, help="推理时使用 baseline 头或 igd 头；默认与 checkpoint_stage 一致")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="显式指定要加载的 checkpoint 目录")
    p.add_argument("--checkpoint_name", type=str, default=None, help="显式指定要加载的 checkpoint 目录名，如 ckpt_20260424_203928")
    p.add_argument("--enable_defense_infer", action="store_true", help="启用随机 mask + IG-guided mask 的鲁棒推理")
    p.add_argument("--disable_defense_infer", action="store_true", help="强制关闭鲁棒推理，使用原始 IGD 推理")
    p.add_argument("--random_mask_trials", type=int, default=None, help="覆盖 defense_infer.random_mask_trials")
    p.add_argument("--stability_margin", type=float, default=None, help="覆盖 defense_infer.stability_margin")
    p.add_argument("--guided_ig_steps", type=int, default=None, help="覆盖 defense_infer.guided_ig_steps")
    p.add_argument("--query_budget", type=int, default=None, help="限制每条样本的最大查询次数，适合慢速攻击 smoke/补充实验")
    p.add_argument("--eval_batch_size", type=int, default=4, help="模型评估时的分块 batch size；BERTAttack 显存紧张时可设为 1")
    p.add_argument("--num_examples_offset", type=int, default=0, help="从评估集偏移位置开始跑，便于分段续跑")
    return p.parse_args()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()


def _resolve_eval_output_root(cfg) -> str:
    dataset_name = cfg["dataset"]["name"]
    return os.path.join(cfg["paths"]["output_dir"], dataset_name)


def _resolve_checkpoint_dir(cfg, stage: str) -> Optional[str]:
    dataset_name = cfg["dataset"]["name"]
    out_root = cfg["paths"]["output_dir"]
    stage_dir = os.path.normpath(os.path.join(out_root, dataset_name, stage))
    return resolve_latest_artifact_dir(stage_dir, remap_output_root=out_root)


def _resolve_manual_checkpoint_dir(cfg, checkpoint_dir: Optional[str], checkpoint_name: Optional[str], stage: str) -> Optional[str]:
    dataset_name = cfg["dataset"]["name"]
    out_root = cfg["paths"]["output_dir"]
    if checkpoint_dir:
        ckpt_dir = remap_legacy_output_path(checkpoint_dir, out_root)
        ckpt = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(ckpt):
            return os.path.normpath(ckpt_dir)
        raise FileNotFoundError(f"指定的 checkpoint_dir 不存在模型文件：{checkpoint_dir}")

    if checkpoint_name:
        norm_path = os.path.normpath(os.path.join(out_root, dataset_name, stage, checkpoint_name))
        ckpt_dir = remap_legacy_output_path(norm_path, out_root)
        ckpt = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(ckpt):
            return os.path.normpath(ckpt_dir)
        raise FileNotFoundError(f"未找到指定的 checkpoint_name：{checkpoint_name}")

    return None


def _format_percent(value: float) -> str:
    return f"{value:.2f}%"


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _render_attack_summary(dataset_name: str, checkpoint_dir: str, summaries: dict) -> str:
    lines = [f"Dataset: {dataset_name}", f"Checkpoint: {checkpoint_dir}", ""]
    for attack_name, summary in summaries.items():
        lines.extend(
            [
                f"Attack: {attack_name}",
                "+-------------------------------+--------+",
                "| Attack Results                |        |",
                "+-------------------------------+--------+",
                f"| Number of successful attacks: | {summary['number_of_successful_attacks']:<6} |",
                f"| Number of failed attacks:     | {summary['number_of_failed_attacks']:<6} |",
                f"| Number of skipped attacks:    | {summary['number_of_skipped_attacks']:<6} |",
                f"| Original accuracy:            | {_format_percent(summary['original_accuracy']):<6} |",
                f"| Accuracy under attack:        | {_format_percent(summary['accuracy_under_attack']):<6} |",
                f"| Attack success rate:          | {_format_percent(summary['attack_success_rate']):<6} |",
                f"| Average perturbed word %:     | {_format_percent(summary['average_perturbed_word_percent']):<6} |",
                f"| Average num. words per input: | {_format_number(summary['average_num_words_per_input']):<6} |",
                f"| Avg num queries:              | {_format_number(summary['avg_num_queries']):<6} |",
                "+-------------------------------+--------+",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, RunOverrides(dataset_name=args.dataset))
    infer_cfg = cfg.setdefault("defense_infer", {})
    if args.enable_defense_infer:
        infer_cfg["enabled"] = True
    if args.disable_defense_infer:
        infer_cfg["enabled"] = False
    if args.random_mask_trials is not None:
        infer_cfg["random_mask_trials"] = int(args.random_mask_trials)
    if args.stability_margin is not None:
        infer_cfg["stability_margin"] = float(args.stability_margin)
    if args.guided_ig_steps is not None:
        infer_cfg["guided_ig_steps"] = int(args.guided_ig_steps)

    eval_stage = args.eval_stage or args.checkpoint_stage
    if bool(infer_cfg.get("enabled", False)) and eval_stage != "igd":
        raise ValueError("鲁棒推理防御只支持 eval_stage=igd；baseline 评估请使用 --disable_defense_infer")

    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)

    loaded = load_dataset_and_tokenizer(cfg, max_train_samples=1, max_eval_samples=1)
    model = IGDModel(cfg, num_labels=loaded.num_labels)

    checkpoint_dir = _resolve_manual_checkpoint_dir(cfg, args.checkpoint_dir, args.checkpoint_name, args.checkpoint_stage)
    if checkpoint_dir is None:
        checkpoint_dir = _resolve_checkpoint_dir(cfg, args.checkpoint_stage)
    if checkpoint_dir is None:
        raise FileNotFoundError(
            f"未找到 {args.checkpoint_stage} checkpoint：请确认 paths.output_dir（或环境变量 IGD_OUTPUT_DIR）下是否存在对应数据集模型"
        )
    ckpt = os.path.join(checkpoint_dir, "pytorch_model.bin")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)

    results = run_attack_eval(
        cfg=cfg,
        model=model,
        device=device,
        attacks=args.attacks,
        max_eval_samples=args.max_eval_samples,
        query_budget=args.query_budget,
        eval_batch_size=args.eval_batch_size,
        eval_stage=eval_stage,
        num_examples_offset=args.num_examples_offset,
    )
    run_ts = timestamp_now()
    output_root = _resolve_eval_output_root(cfg)
    out_dir = os.path.join(output_root, "attack_eval")
    os.makedirs(out_dir, exist_ok=True)

    attack_label = "_".join(_slugify(name) for name in args.attacks)
    out_path = os.path.join(out_dir, f"attack_summary_{attack_label}_{run_ts}.txt")
    content = _render_attack_summary(cfg["dataset"]["name"], checkpoint_dir, results)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(content, end="")
    print(f"saved: {out_path}")
    print(f"loaded {args.checkpoint_stage} checkpoint: {checkpoint_dir}")
    print(f"eval stage: {eval_stage}")


if __name__ == "__main__":
    main()

