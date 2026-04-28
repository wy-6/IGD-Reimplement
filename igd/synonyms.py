from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


def is_simple_word(w: str) -> bool:
    return bool(_WORD_RE.match(w))


@dataclass
class SynonymConfig:
    k: int = 50
    min_sim: float = 0.84


class GloveSynonyms:
    """
    基于 GloVe 的同义词候选（按余弦相似度取 top-k），用于伪样本生成和攻击约束对齐。

    说明：
    - 默认使用 gensim downloader 的 `glove-wiki-gigaword-100`
    - 如果在 Kaggle 关闭网络，需提前在 Notebook 打开 Internet 或把向量文件作为 dataset 上传
    """

    def __init__(self, cfg: SynonymConfig, glove_name: str = "glove-wiki-gigaword-100"):
        self.cfg = cfg
        self.glove_name = glove_name
        self._kv = None

    def load(self):
        if self._kv is not None:
            return
        import gensim.downloader as api

        self._kv = api.load(self.glove_name)

    def _vec(self, w: str) -> Optional[np.ndarray]:
        if self._kv is None:
            self.load()
        if w in self._kv:
            return self._kv[w]
        wl = w.lower()
        if wl in self._kv:
            return self._kv[wl]
        return None

    def candidates(self, word: str) -> List[Tuple[str, float]]:
        if not is_simple_word(word):
            return []
        v = self._vec(word)
        if v is None:
            return []

        # gensim KV 提供 most_similar
        try:
            raw = self._kv.most_similar(word if word in self._kv else word.lower(), topn=int(self.cfg.k))
        except KeyError:
            return []

        out: List[Tuple[str, float]] = []
        for w, sim in raw:
            sim = float(sim)
            if sim < float(self.cfg.min_sim):
                continue
            if not is_simple_word(w):
                continue
            if w.lower() == word.lower():
                continue
            out.append((w, sim))
        return out

