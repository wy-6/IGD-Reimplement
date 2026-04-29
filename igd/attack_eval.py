from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from .data import format_label_distribution, shuffle_and_select_subset
from .masked_infer import robust_predict_logits


def _dataset_max_mod_ratio(cfg: Dict[str, Any]) -> float:
    name = cfg["dataset"]["name"]
    if name == "ag_news":
        return float(cfg["attack"].get("max_mod_ratio_agnews", 0.3))
    if name == "imdb":
        return float(cfg["attack"].get("max_mod_ratio_imdb", 0.1))
    return 0.3


def as_textattack_model_wrapper(wrapper):
    """
    兼容不同 TextAttack 版本的最小适配：
    - 如果可用，继承/包装成 textattack.models.wrappers.ModelWrapper
    - 否则直接返回可调用对象（部分版本可用）
    """
    try:
        from textattack.models.wrappers import ModelWrapper

        class _Wrapped(ModelWrapper):
            def __init__(self, inner):
                self.inner = inner
                # TextAttack 某些版本会访问 model_wrapper.model 用于日志/类型展示
                self.model = getattr(inner, "model", None)

            def __call__(self, text_input_list):
                return self.inner(text_input_list)

        return _Wrapped(wrapper)
    except Exception:
        return wrapper


class IGDTextAttackWrapper:
    """
    最小 TextAttack wrapper：输入 list[str]，输出 logits(np.ndarray-like)。
    推理按技术路线：不提供 pseudo_*，由模型内部用 0 填充 v'。
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        max_length: int,
        cfg: Optional[Dict[str, Any]] = None,
        eval_batch_size: int = 4,
        eval_stage: str = "igd",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = int(max_length)
        self.cfg = cfg or {}
        self.eval_batch_size = max(1, int(eval_batch_size))
        if eval_stage not in {"baseline", "igd"}:
            raise ValueError(f"未知 eval_stage={eval_stage}，应为 baseline/igd")
        self.eval_stage = eval_stage
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text_list: List[str]):
        infer_cfg = self.cfg.get("defense_infer", {}) or {}
        if bool(infer_cfg.get("enabled", False)) and self.eval_stage != "igd":
            raise ValueError("鲁棒推理防御只支持 eval_stage=igd；baseline 评估请关闭 defense_infer")
        if not bool(infer_cfg.get("enabled", False)):
            return self._predict_batch(text_list)

        logits_list: List[torch.Tensor] = []
        for text in text_list:
            logits = robust_predict_logits(
                text,
                cfg=self.cfg,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            logits_list.append(logits)
        if not logits_list:
            return []
        return torch.stack(logits_list, dim=0).detach().cpu().numpy()

    @torch.no_grad()
    def _predict_batch(self, text_list: List[str]):
        if not text_list:
            return []
        logits_chunks: List[torch.Tensor] = []
        for start in range(0, len(text_list), self.eval_batch_size):
            chunk = text_list[start : start + self.eval_batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                stage=self.eval_stage,
            )
            logits_chunks.append(out.logits.detach().cpu())
        return torch.cat(logits_chunks, dim=0).numpy()


def _align_textattack_device(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("命令行指定了 --device cuda，但当前环境未检测到可用 CUDA/GPU")

    device_str = str(device)
    os.environ["TA_DEVICE"] = device_str

    try:
        import textattack

        textattack.shared.utils.device = device_str
        misc = getattr(textattack.shared.utils, "misc", None)
        if misc is not None:
            misc.device = device_str
    except Exception:
        pass


def build_attack(cfg: Dict[str, Any], attack_name: str, model_wrapper, device: torch.device, eval_stage: str = "igd"):
    from textattack.attack_recipes import BERTAttackLi2020, TextBuggerLi2018, TextFoolerJin2019
    from textattack.constraints.overlap import MaxWordsPerturbed
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import textattack

    class SentenceTransformerSemanticConstraint(textattack.constraints.Constraint):
        def __init__(self, model_name: str, threshold: float):
            super().__init__(compare_against_original=True)
            self.model = SentenceTransformer(model_name, device=str(device))
            self.threshold = float(threshold)

        def _cos(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
            b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
            return (a * b).sum(axis=1)

        def _check_constraint(self, transformed_text, reference_text):
            # 保持接口：单条对比
            ref = reference_text.text
            cand = transformed_text.text
            emb = self.model.encode([ref, cand], convert_to_numpy=True, normalize_embeddings=True)
            sim = float((emb[0] * emb[1]).sum())
            return sim >= self.threshold

        def _check_constraint_many(self, transformed_texts, reference_text):
            if not transformed_texts:
                return []
            ref = reference_text.text
            cands = [t.text for t in transformed_texts]
            emb = self.model.encode([ref] + cands, convert_to_numpy=True, normalize_embeddings=True)
            ref_emb = emb[0:1]
            cand_emb = emb[1:]
            sims = self._cos(np.repeat(ref_emb, repeats=cand_emb.shape[0], axis=0), cand_emb)
            return [t for t, ok in zip(transformed_texts, sims >= self.threshold) if bool(ok)]

    attack_name = attack_name.lower()
    if attack_name == "textfooler":
        attack = TextFoolerJin2019.build(model_wrapper)
    elif attack_name == "textbugger":
        attack = TextBuggerLi2018.build(model_wrapper)
    elif attack_name in {"bertattack", "bert-attack"}:
        attack = BERTAttackLi2020.build(model_wrapper)
    else:
        raise ValueError(f"未知 attack={attack_name}")

    # 一些 TextAttack recipe 默认带 UniversalSentenceEncoder(TFHub) 语义约束，
    # 在 Windows/无网络/TFHub 缓存异常时会直接失败；这里移除它，统一使用 sentence-transformers
    attack.constraints = [c for c in attack.constraints if c.__class__.__name__ != "UniversalSentenceEncoder"]

    # baseline 用作普通 BERT 脆弱性对照时，TextFooler 不再叠加强语义约束和额外修改比例。
    # 这样保留 recipe 自带的基础约束，避免 baseline 攻击被过度限制。
    if eval_stage == "baseline" and attack_name == "textfooler":
        return attack

    # 约束对齐（尽量接近技术路线）
    max_ratio = _dataset_max_mod_ratio(cfg)
    attack.constraints.append(MaxWordsPerturbed(max_percent=max_ratio))

    min_sim = float(cfg["attack"].get("min_semantic_sim", 0.84))
    # 自定义 SentenceTransformer 语义约束（避免 TextAttack 内置 USE/抽象类差异）
    attack.constraints.append(SentenceTransformerSemanticConstraint("all-MiniLM-L6-v2", threshold=min_sim))

    return attack


def _safe_num_queries(result) -> int:
    for obj in (result, getattr(result, "perturbed_result", None), getattr(result, "original_result", None)):
        value = getattr(obj, "num_queries", None)
        if value is not None:
            return int(value)
    return 0


def _num_words_from_result(result) -> int:
    original = getattr(getattr(result, "original_result", None), "attacked_text", None)
    if original is None:
        return 0
    words = getattr(original, "words", None)
    if words is not None:
        return len(words)
    text = getattr(original, "text", "") or ""
    return len(text.split())


def _perturbed_word_percent(result) -> float:
    original = getattr(getattr(result, "original_result", None), "attacked_text", None)
    perturbed = getattr(getattr(result, "perturbed_result", None), "attacked_text", None)
    if original is None or perturbed is None:
        return 0.0

    diff_words = []
    diff_fn = getattr(original, "all_words_diff", None)
    if callable(diff_fn):
        try:
            diff_words = diff_fn(perturbed)
        except Exception:
            diff_words = []

    num_words = _num_words_from_result(result)
    if num_words <= 0:
        return 0.0
    return 100.0 * len(diff_words) / float(num_words)


def _summarize_attack_results(attack_name: str, results: List[Any], total_examples: int) -> Dict[str, Any]:
    success_results = [r for r in results if r.__class__.__name__ == "SuccessfulAttackResult"]
    failed_results = [r for r in results if r.__class__.__name__ == "FailedAttackResult"]
    skipped_results = [r for r in results if r.__class__.__name__ == "SkippedAttackResult"]
    evaluated_results = success_results + failed_results

    successful = len(success_results)
    failed = len(failed_results)
    skipped = len(skipped_results)

    original_accuracy = 100.0 * (successful + failed) / max(1, total_examples)
    accuracy_under_attack = 100.0 * failed / max(1, total_examples)
    attack_success_rate = 100.0 * successful / max(1, successful + failed)

    perturbed_values = [_perturbed_word_percent(r) for r in success_results]
    word_counts = [_num_words_from_result(r) for r in evaluated_results]
    query_counts = [_safe_num_queries(r) for r in evaluated_results]

    return {
        "attack": attack_name,
        "number_of_successful_attacks": successful,
        "number_of_failed_attacks": failed,
        "number_of_skipped_attacks": skipped,
        "original_accuracy": round(original_accuracy, 2),
        "accuracy_under_attack": round(accuracy_under_attack, 2),
        "attack_success_rate": round(attack_success_rate, 2),
        "average_perturbed_word_percent": round(sum(perturbed_values) / max(1, len(perturbed_values)), 2),
        "average_num_words_per_input": round(sum(word_counts) / max(1, len(word_counts)), 2),
        "avg_num_queries": round(sum(query_counts) / max(1, len(query_counts)), 2),
        "total_examples": total_examples,
    }


def run_attack_eval(
    *,
    cfg: Dict[str, Any],
    model,
    device: torch.device,
    attacks: List[str],
    max_eval_samples: Optional[int] = None,
    query_budget: Optional[int] = None,
    eval_batch_size: int = 4,
    eval_stage: str = "igd",
    num_examples_offset: int = 0,
) -> Dict[str, Any]:
    import datasets
    import textattack

    _align_textattack_device(device)

    ds_name = cfg["dataset"]["name"]
    seed = int(cfg.get("seed", 42))
    raw = datasets.load_dataset(ds_name)
    split = "test" if "test" in raw else "validation"
    raw[split] = shuffle_and_select_subset(raw[split], max_eval_samples, seed=seed + 1)

    text_key = "text" if "text" in raw[split].column_names else raw[split].column_names[0]
    label_key = cfg["dataset"].get("label_key", "label")
    print(f"[data] attack {split} subset label distribution: {format_label_distribution(raw[split], label_key)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"], use_fast=True)
    wrapper = IGDTextAttackWrapper(
        model,
        tokenizer,
        device=device,
        max_length=int(cfg["dataset"].get("max_length", 128)),
        cfg=cfg,
        eval_batch_size=eval_batch_size,
        eval_stage=eval_stage,
    )
    wrapper = as_textattack_model_wrapper(wrapper)

    num_examples_offset = max(0, int(num_examples_offset))
    dataset = [(ex[text_key], int(ex[label_key])) for ex in raw[split]]
    if num_examples_offset:
        dataset = dataset[num_examples_offset:]
    try:
        from textattack.datasets import Dataset as TA_Dataset

        dataset = TA_Dataset(dataset)
    except Exception:
        # 部分旧版 textattack 仍接受 list[(text,label)]
        pass

    results: Dict[str, Any] = {}
    for a in attacks:
        attack = build_attack(cfg, a, wrapper, device=device, eval_stage=eval_stage)
        num_examples = len(dataset)
        attack_args_kwargs = {
            "num_examples": num_examples,
            "disable_stdout": True,
        }
        if query_budget is not None:
            attack_args_kwargs["query_budget"] = int(query_budget)
        try:
            attack_args = textattack.AttackArgs(**attack_args_kwargs)
        except TypeError:
            # 兼容旧版 TextAttack：不支持 query_budget 时退回基础参数。
            attack_args = textattack.AttackArgs(num_examples=num_examples, disable_stdout=True)
        attacker = textattack.Attacker(attack, dataset, attack_args=attack_args)
        raw_results = attacker.attack_dataset()

        if not isinstance(raw_results, list):
            raw_results = list(raw_results)
        results[a] = _summarize_attack_results(a, raw_results, total_examples=max(1, num_examples))
    return results

