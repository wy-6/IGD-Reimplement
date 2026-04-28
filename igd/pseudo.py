from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from .ig import integrated_gradients_token_importance, select_important_token_indices
from .synonyms import GloveSynonyms, SynonymConfig, is_simple_word


def _dataset_max_mod_ratio(cfg: Dict[str, Any]) -> float:
    name = cfg["dataset"]["name"]
    if name == "ag_news":
        return float(cfg["igd"].get("max_mod_ratio_agnews", 0.3))
    if name == "imdb":
        return float(cfg["igd"].get("max_mod_ratio_imdb", 0.1))
    return 0.3


def _is_replaceable_token(tok: str) -> bool:
    if tok in ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]:
        return False
    if tok.startswith("##"):
        return False
    if tok in ["[UNK]"]:
        return False
    return is_simple_word(tok)


def _apply_token_replacements(tokens: List[str], repl: Dict[int, str]) -> List[str]:
    out = list(tokens)
    for idx, new_tok in repl.items():
        if 0 <= idx < len(out):
            out[idx] = new_tok
    return out


@dataclass
class PseudoLine:
    text: str
    pseudo_text: str
    label: int


def generate_pseudo_for_text(
    *,
    cfg: Dict[str, Any],
    model,
    tokenizer,
    synonyms: GloveSynonyms,
    text: str,
    label: int,
    device: torch.device,
    ig_steps: int,
) -> str:
    enc = tokenizer(text, truncation=True, max_length=int(cfg["dataset"].get("max_length", 128)), return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    scores = integrated_gradients_token_importance(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        target_label=int(label),
        steps=int(ig_steps),
    )
    important = select_important_token_indices(scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    replaceable = [i for i, tok in enumerate(tokens) if _is_replaceable_token(tok)]
    if not replaceable:
        return text

    imp_set = set(int(i) for i in important.tolist())
    ranked = sorted([i for i in replaceable if i in imp_set], key=lambda i: float(scores[i].item()), reverse=True)
    if not ranked:
        ranked = sorted(replaceable, key=lambda i: float(scores[i].item()), reverse=True)

    max_ratio = _dataset_max_mod_ratio(cfg)
    max_mod = max(1, int(math.floor(max_ratio * len(replaceable))))

    repl: Dict[int, str] = {}
    for idx in ranked:
        if len(repl) >= max_mod:
            break
        w = tokens[idx]
        cands = synonyms.candidates(w)
        if not cands:
            continue
        new_w = random.choice(cands)[0]
        if new_w.lower() == w.lower():
            continue
        repl[idx] = new_w

    if not repl:
        return text

    new_tokens = _apply_token_replacements(tokens, repl)
    pseudo_text = tokenizer.convert_tokens_to_string(new_tokens)
    pseudo_text = " ".join(pseudo_text.split())
    return pseudo_text


def generate_pseudo_lines(
    *,
    cfg: Dict[str, Any],
    model,
    tokenizer,
    train_dataset,
    text_key: str,
    label_key: str,
    device: torch.device,
    ig_steps: int,
) -> List[PseudoLine]:
    """按训练集在内存中生成伪样本，不写磁盘。"""
    syn_cfg = SynonymConfig(k=int(cfg["igd"].get("synonym_k", 50)), min_sim=float(cfg["igd"].get("min_semantic_sim", 0.84)))
    synonyms = GloveSynonyms(syn_cfg)
    synonyms.load()

    out: List[PseudoLine] = []
    for ex in tqdm(train_dataset, desc="generate pseudo (in-memory)"):
        text = ex[text_key]
        label = int(ex[label_key])
        pseudo_text = generate_pseudo_for_text(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            synonyms=synonyms,
            text=text,
            label=label,
            device=device,
            ig_steps=ig_steps,
        )
        out.append(PseudoLine(text=text, pseudo_text=pseudo_text, label=label))
    return out
