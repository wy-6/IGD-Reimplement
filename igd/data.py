from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass
class LoadedData:
    raw: DatasetDict
    tokenizer: Any
    data_collator: Any
    num_labels: int
    text_key: str
    label_key: str


def _infer_text_key(ds: Dataset) -> str:
    for k in ["text", "sentence", "content", "review"]:
        if k in ds.column_names:
            return k
    raise ValueError(f"无法推断 text 字段，columns={ds.column_names}")


def shuffle_and_select_subset(ds: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None:
        return ds
    n = min(int(max_samples), len(ds))
    if n >= len(ds):
        return ds
    return ds.shuffle(seed=int(seed)).select(range(n))


def format_label_distribution(ds: Dataset, label_key: str) -> str:
    counts = Counter(int(x) for x in ds[label_key])
    return ", ".join(f"{label}: {counts[label]}" for label in sorted(counts))


def load_dataset_and_tokenizer(
    cfg: Dict[str, Any],
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> LoadedData:
    name = cfg["dataset"]["name"]
    max_length = int(cfg["dataset"].get("max_length", 128))
    label_key = cfg["dataset"].get("label_key", "label")
    seed = int(cfg.get("seed", 42))

    raw = datasets.load_dataset(name)
    train_split = "train"
    eval_split = "test" if "test" in raw else ("validation" if "validation" in raw else None)
    if eval_split is None:
        raise ValueError(f"{name} 没有 test/validation split，splits={list(raw.keys())}")

    if max_train_samples is None:
        default_map = cfg["dataset"].get("default_max_train_samples", {}) or {}
        if isinstance(default_map, dict) and name in default_map and default_map[name] is not None:
            max_train_samples = int(default_map[name])
    raw[train_split] = shuffle_and_select_subset(raw[train_split], max_train_samples, seed=seed)
    if max_eval_samples is None:
        default_map = cfg["dataset"].get("default_max_eval_samples", {}) or {}
        if isinstance(default_map, dict) and name in default_map and default_map[name] is not None:
            max_eval_samples = int(default_map[name])
    raw[eval_split] = shuffle_and_select_subset(raw[eval_split], max_eval_samples, seed=seed + 1)

    print(f"[data] train subset label distribution: {format_label_distribution(raw[train_split], label_key)}")
    print(f"[data] {eval_split} subset label distribution: {format_label_distribution(raw[eval_split], label_key)}")

    text_key = cfg["dataset"].get("text_key") or _infer_text_key(raw[train_split])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"], use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch[text_key],
            truncation=True,
            max_length=max_length,
        )

    tokenized = DatasetDict()
    for split in [train_split, eval_split]:
        # 训练阶段不需要原始 text 列；保留标签即可，避免 data_collator 试图把字符串列转 tensor
        ds = raw[split].map(
            tokenize,
            batched=True,
            remove_columns=[c for c in raw[split].column_names if c != label_key],
        )
        if label_key not in ds.column_names:
            raise ValueError(f"label 字段缺失：expected {label_key}, columns={ds.column_names}")
        if label_key != "labels":
            ds = ds.rename_column(label_key, "labels")
        tokenized[split] = ds

    num_labels = cfg["dataset"].get("num_labels")
    if num_labels is None:
        # datasets 通常提供 features
        try:
            num_labels = int(raw[train_split].features[label_key].num_classes)
        except Exception:
            num_labels = int(max(raw[train_split][label_key]) + 1)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return LoadedData(raw=tokenized, tokenizer=tokenizer, data_collator=data_collator, num_labels=num_labels, text_key=text_key, label_key="labels")


def get_dataset_splits(loaded: LoadedData) -> Tuple[Dataset, Dataset]:
    train = loaded.raw["train"]
    eval_split = "test" if "test" in loaded.raw else "validation"
    return train, loaded.raw[eval_split]

