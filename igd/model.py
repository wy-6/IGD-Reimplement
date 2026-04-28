from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .losses import AMSoftmaxConfig, AMSoftmaxHead, GRL, am_softmax_loss


@dataclass
class IGDOutputs:
    logits: torch.Tensor
    loss_cls: Optional[torch.Tensor] = None
    loss_src: Optional[torch.Tensor] = None
    loss_total: Optional[torch.Tensor] = None


class SourceClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IGDModel(nn.Module):
    """
    - baseline: 只用 v 做分类（普通线性头）
    - igd: 主分类用 [v, v']（AM-Softmax）；源分类器用 GRL 约束 v/v' 不可区分
    推理阶段：不提供 pseudo_* 时，v' 用 0 填充（与技术路线一致）
    """

    def __init__(self, cfg: Dict[str, Any], num_labels: int):
        super().__init__()
        backbone_name = cfg["model"]["backbone"]
        self.num_labels = int(num_labels)

        hf_cfg = AutoConfig.from_pretrained(backbone_name, num_labels=self.num_labels)
        self.encoder = AutoModel.from_pretrained(backbone_name, config=hf_cfg)
        self.hidden = int(hf_cfg.hidden_size)
        dropout_p = float(cfg["model"].get("dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        # baseline head
        self.baseline_head = nn.Linear(self.hidden, self.num_labels)

        # IGD head: AM-Softmax on concatenated vector
        am_cfg = AMSoftmaxConfig(
            s=float(cfg["igd"].get("am_s", 1.0)),
            m=float(cfg["igd"].get("am_m", 0.3)),
        )
        self.am_head = AMSoftmaxHead(self.hidden * 2, self.num_labels, am_cfg)

        # GRL + source classifier
        self.grl = GRL(lambd=float(cfg["igd"].get("lambda_grl", 0.2)))
        self.src_clf = SourceClassifier(self.hidden)

        self.use_zero_pseudo_at_infer = bool(cfg["igd"].get("use_zero_pseudo_at_infer", True))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 对 BERT 系列：pooler_output 可用；对 RoBERTa 没有 pooler，退化为 CLS
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            v = out.pooler_output
        else:
            v = out.last_hidden_state[:, 0]
        return self.dropout(v)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        stage: str = "baseline",
        token_type_ids: Optional[torch.Tensor] = None,
        pseudo_input_ids: Optional[torch.Tensor] = None,
        pseudo_attention_mask: Optional[torch.Tensor] = None,
        pseudo_token_type_ids: Optional[torch.Tensor] = None,
    ) -> IGDOutputs:
        v = self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if stage == "baseline":
            logits = self.baseline_head(v)
            loss = F.cross_entropy(logits, labels) if labels is not None else None
            return IGDOutputs(logits=logits, loss_cls=loss, loss_total=loss)

        if stage != "igd":
            raise ValueError(f"未知 stage={stage}，应为 baseline/igd")

        if pseudo_input_ids is None or pseudo_attention_mask is None:
            if not self.use_zero_pseudo_at_infer:
                raise ValueError("igd 推理未提供 pseudo_*，且 use_zero_pseudo_at_infer=false")
            v_p = torch.zeros_like(v)
        else:
            v_p = self.encode(
                input_ids=pseudo_input_ids,
                attention_mask=pseudo_attention_mask,
                token_type_ids=pseudo_token_type_ids,
            )

        z = torch.cat([v, v_p], dim=-1)
        logits = self.am_head(z, labels=labels if labels is not None else None)

        loss_cls = am_softmax_loss(logits, labels) if labels is not None else None

        loss_src = None
        if pseudo_input_ids is not None and labels is not None:
            # 源域判别：v=0(干净)，v'=1(伪样本)
            src_x = torch.cat([v, v_p], dim=0)
            src_y = torch.cat(
                [torch.zeros(v.size(0), dtype=torch.long, device=v.device), torch.ones(v_p.size(0), dtype=torch.long, device=v.device)],
                dim=0,
            )
            src_logits = self.src_clf(self.grl(src_x))
            loss_src = F.cross_entropy(src_logits, src_y)

        loss_total = None
        if loss_cls is not None:
            loss_total = loss_cls + (loss_src if loss_src is not None else 0.0)

        return IGDOutputs(logits=logits, loss_cls=loss_cls, loss_src=loss_src, loss_total=loss_total)

