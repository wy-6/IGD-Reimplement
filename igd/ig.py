from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _pooled_output(encoder_outputs) -> torch.Tensor:
    if hasattr(encoder_outputs, "pooler_output") and encoder_outputs.pooler_output is not None:
        return encoder_outputs.pooler_output
    return encoder_outputs.last_hidden_state[:, 0]


def integrated_gradients_token_importance(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor],
    target_label: int,
    steps: int = 50,
) -> torch.Tensor:
    """
    返回每个 token 的重要性分数（非负），shape=[seq_len]
    说明：用 inputs_embeds 做 IG 近似，baseline 为 0 向量（word embedding 部分）。
    """
    model.eval()
    device = input_ids.device
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps 必须 > 0")

    # 仅支持单样本（生成伪样本时足够；后续可再做 batch 化）
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("当前实现仅支持 batch=1 的 IG 计算")

    emb_layer = model.encoder.get_input_embeddings()
    x = emb_layer(input_ids)  # [1, L, H]
    x0 = torch.zeros_like(x)  # baseline
    dx = x - x0

    total_grad = torch.zeros_like(x)

    for k in range(1, steps + 1):
        alpha = float(k) / float(steps)
        xk = (x0 + alpha * dx).detach().requires_grad_(True)

        outputs = model.encoder(
            inputs_embeds=xk,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        v = _pooled_output(outputs)
        v = model.dropout(v)
        logits = model.baseline_head(v)  # baseline 头用于打分
        logit = logits[0, int(target_label)]

        grads = torch.autograd.grad(logit, xk, retain_graph=False, create_graph=False)[0]
        total_grad += grads.detach()

    avg_grad = total_grad / float(steps)
    ig = dx * avg_grad  # [1, L, H]
    # token importance：对 hidden 维度取 L1 范数
    scores = ig.abs().sum(dim=-1).squeeze(0)  # [L]
    # mask 掉 padding
    scores = scores * attention_mask.squeeze(0).float()
    return scores


def select_important_token_indices(scores: torch.Tensor) -> torch.Tensor:
    """
    按技术路线：选择大于均值的 token 作为候选替换词。
    """
    if scores.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=scores.device)
    mean = scores[scores > 0].mean() if (scores > 0).any() else scores.mean()
    idx = torch.nonzero(scores > mean, as_tuple=False).squeeze(-1)
    return idx

