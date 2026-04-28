from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .ig import integrated_gradients_token_importance


_WORDLIKE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'._-]*$")


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _mask_budget(num_positions: int, ratio: float, min_tokens: int, max_tokens: Optional[int]) -> int:
    if num_positions <= 0:
        return 0
    k = max(int(round(num_positions * float(ratio))), int(min_tokens))
    if max_tokens is not None:
        k = min(k, int(max_tokens))
    return max(1, min(num_positions, k))


def _is_maskable_token(token: str) -> bool:
    if token in {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}:
        return False
    if token.startswith("##"):
        return False
    return bool(_WORDLIKE_RE.match(token))


def _candidate_positions(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[int]:
    token_ids = input_ids.squeeze(0).tolist()
    attn = attention_mask.squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    out: List[int] = []
    for idx, (tok, keep) in enumerate(zip(tokens, attn)):
        if int(keep) != 1:
            continue
        if _is_maskable_token(tok):
            out.append(int(idx))
    return out


def _masked_copy(input_ids: torch.Tensor, positions: Sequence[int], mask_token_id: int) -> torch.Tensor:
    out = input_ids.clone()
    for pos in positions:
        out[0, int(pos)] = int(mask_token_id)
    return out


@torch.no_grad()
def _predict_logits_from_encoded(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        stage="igd",
    )
    return out.logits.squeeze(0)


def predict_logits_for_text(
    text: str,
    *,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    enc = tokenizer(text, truncation=True, max_length=int(max_length), return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    return _predict_logits_from_encoded(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )


def _margin_from_logits(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    if probs.numel() <= 1:
        return 1.0
    top2 = torch.topk(probs, k=min(2, probs.numel()), dim=-1).values
    if top2.numel() < 2:
        return 1.0
    return float((top2[0] - top2[1]).item())


def _encode_text(text: str, tokenizer, device: torch.device, max_length: int):
    enc = tokenizer(text, truncation=True, max_length=int(max_length), return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    return input_ids, attention_mask, token_type_ids


def random_mask_logits(
    text: str,
    *,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    trials: int,
    mask_ratio: float,
    min_tokens_to_mask: int,
    max_tokens_to_mask: Optional[int],
) -> torch.Tensor:
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        return predict_logits_for_text(text, model=model, tokenizer=tokenizer, device=device, max_length=max_length)

    input_ids, attention_mask, token_type_ids = _encode_text(text, tokenizer, device, max_length)
    candidates = _candidate_positions(tokenizer, input_ids, attention_mask)
    if not candidates:
        return _predict_logits_from_encoded(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    k = _mask_budget(len(candidates), mask_ratio, min_tokens_to_mask, max_tokens_to_mask)
    rng = random.Random(_stable_seed(text))
    logits_list: List[torch.Tensor] = []
    for trial_idx in range(max(1, int(trials))):
        trial_rng = random.Random(rng.randint(0, 2**31 - 1) + int(trial_idx))
        sampled = trial_rng.sample(candidates, k=min(k, len(candidates)))
        masked_ids = _masked_copy(input_ids, sampled, mask_token_id=int(mask_token_id))
        logits = _predict_logits_from_encoded(
            model,
            input_ids=masked_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits_list.append(logits)
    return torch.stack(logits_list, dim=0).mean(dim=0)


def guided_mask_logits(
    text: str,
    *,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    ig_steps: int,
    guided_mask_ratio: float,
    min_tokens_to_mask: int,
    max_tokens_to_mask: Optional[int],
) -> torch.Tensor:
    mask_token_id = tokenizer.mask_token_id
    input_ids, attention_mask, token_type_ids = _encode_text(text, tokenizer, device, max_length)
    clean_logits = _predict_logits_from_encoded(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    if mask_token_id is None:
        return clean_logits

    candidates = _candidate_positions(tokenizer, input_ids, attention_mask)
    if not candidates:
        return clean_logits

    target_label = int(clean_logits.argmax(dim=-1).item())
    scores = integrated_gradients_token_importance(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        target_label=target_label,
        steps=max(1, int(ig_steps)),
    )
    ranked = sorted(candidates, key=lambda pos: float(scores[pos].item()), reverse=True)
    k = _mask_budget(len(ranked), guided_mask_ratio, min_tokens_to_mask, max_tokens_to_mask)
    chosen = ranked[:k]
    if not chosen:
        return clean_logits

    masked_ids = _masked_copy(input_ids, chosen, mask_token_id=int(mask_token_id))
    return _predict_logits_from_encoded(
        model,
        input_ids=masked_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )


def robust_predict_logits(
    text: str,
    *,
    cfg: Dict[str, Any],
    model,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    infer_cfg = cfg.get("defense_infer", {}) or {}
    max_length = int(cfg["dataset"].get("max_length", 128))
    enabled = bool(infer_cfg.get("enabled", False))
    if not enabled:
        return predict_logits_for_text(text, model=model, tokenizer=tokenizer, device=device, max_length=max_length)

    min_tokens_to_mask = int(infer_cfg.get("min_tokens_to_mask", 1))
    max_tokens_cfg = infer_cfg.get("max_tokens_to_mask")
    max_tokens_to_mask = int(max_tokens_cfg) if max_tokens_cfg is not None else None

    rand_logits = random_mask_logits(
        text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        trials=int(infer_cfg.get("random_mask_trials", 3)),
        mask_ratio=float(infer_cfg.get("random_mask_ratio", 0.15)),
        min_tokens_to_mask=min_tokens_to_mask,
        max_tokens_to_mask=max_tokens_to_mask,
    )
    if _margin_from_logits(rand_logits) >= float(infer_cfg.get("stability_margin", 0.15)):
        return rand_logits

    return guided_mask_logits(
        text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        ig_steps=int(infer_cfg.get("guided_ig_steps", 8)),
        guided_mask_ratio=float(infer_cfg.get("guided_mask_ratio", 0.10)),
        min_tokens_to_mask=min_tokens_to_mask,
        max_tokens_to_mask=max_tokens_to_mask,
    )
