from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .pseudo import PseudoLine
from .utils import ensure_dir, make_timestamped_dir, save_json


class PairedTextDataset(Dataset):
    def __init__(self, lines, tokenizer, max_length: int):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ln = self.lines[idx]
        return {"text": ln.text, "pseudo_text": ln.pseudo_text, "label": int(ln.label)}


def _paired_collate(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        pseudos = [b["pseudo_text"] for b in batch]
        labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        enc = tokenizer(texts, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")
        enc_p = tokenizer(pseudos, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "pseudo_input_ids": enc_p["input_ids"],
            "pseudo_attention_mask": enc_p["attention_mask"],
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]
        if "token_type_ids" in enc_p:
            out["pseudo_token_type_ids"] = enc_p["token_type_ids"]
        return out

    return collate


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device, stage: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        forward_labels = batch.get("labels") if stage == "baseline" else None
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=forward_labels,
            token_type_ids=batch.get("token_type_ids"),
            pseudo_input_ids=batch.get("pseudo_input_ids"),
            pseudo_attention_mask=batch.get("pseudo_attention_mask"),
            pseudo_token_type_ids=batch.get("pseudo_token_type_ids"),
            stage=stage,
        )
        logits = out.logits
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    return {"acc": correct / max(1, total), "loss": total_loss / max(1, total)}


def _save_checkpoint(cfg: Dict[str, Any], model, tokenizer, stage: str, dataset_name: str) -> str:
    stage_dir = os.path.join(cfg["paths"]["output_dir"], dataset_name, stage)
    ensure_dir(stage_dir)
    out_dir = make_timestamped_dir(stage_dir, prefix="ckpt")
    torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(out_dir)
    save_json(os.path.join(out_dir, "run_config.json"), cfg)
    save_json(os.path.join(stage_dir, "latest_checkpoint.json"), {"latest_checkpoint_dir": out_dir})
    return out_dir


def train_baseline(
    *,
    cfg: Dict[str, Any],
    model,
    tokenizer,
    train_ds,
    eval_ds,
    data_collator,
    device: torch.device,
    max_length: int,
) -> str:
    model.train()
    batch_size = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"].get("epochs_baseline", 1))
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.01))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = epochs * len(train_loader)
    warmup = int(total_steps * float(cfg["training"].get("warmup_ratio", 0.1)))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)

    model.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"baseline train ep{ep}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch["labels"] if "labels" in batch else batch["label"]
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
                token_type_ids=batch.get("token_type_ids"),
                stage="baseline",
            )
            loss = out.loss_total
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            pbar.set_postfix(loss=float(loss.item()))

        metrics = evaluate(model, eval_loader, device, stage="baseline")
        tqdm.write(f"[baseline] ep{ep} eval acc={metrics['acc']:.4f} loss={metrics['loss']:.4f}")

    return _save_checkpoint(cfg, model, tokenizer, stage="baseline", dataset_name=cfg["dataset"]["name"])


def load_baseline_weights(model, baseline_dir: str) -> None:
    path = os.path.join(baseline_dir, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd, strict=False)


def train_igd(
    *,
    cfg: Dict[str, Any],
    model,
    tokenizer,
    pseudo_lines: List[PseudoLine],
    eval_ds,
    data_collator,
    device: torch.device,
    max_length: int,
) -> str:
    train_ds = PairedTextDataset(pseudo_lines, tokenizer=tokenizer, max_length=max_length)
    collate = _paired_collate(tokenizer, max_length=max_length)

    batch_size = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"].get("epochs_igd", 3))
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"].get("weight_decay", 0.01))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = epochs * len(train_loader)
    warmup = int(total_steps * float(cfg["training"].get("warmup_ratio", 0.1)))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)

    model.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"igd train ep{ep}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                token_type_ids=batch.get("token_type_ids"),
                pseudo_input_ids=batch["pseudo_input_ids"],
                pseudo_attention_mask=batch["pseudo_attention_mask"],
                pseudo_token_type_ids=batch.get("pseudo_token_type_ids"),
                stage="igd",
            )
            loss = out.loss_total
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            pbar.set_postfix(loss=float(loss.item()), loss_src=float(out.loss_src.item()) if out.loss_src is not None else 0.0)

        metrics = evaluate(model, eval_loader, device, stage="igd")
        tqdm.write(f"[igd] ep{ep} eval acc={metrics['acc']:.4f} loss={metrics['loss']:.4f}")

    return _save_checkpoint(cfg, model, tokenizer, stage="igd", dataset_name=cfg["dataset"]["name"])

