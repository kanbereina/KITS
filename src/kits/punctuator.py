"""标点恢复：给无标点的转录 chunk 补上日语句读，让断句规则生效。

kotoba-whisper 等蒸馏模型只能产出 chunk（短语）级时间戳且**不带句末标点**，
导致 kits.subtitle.segment_sentences 的「标点断句」「停顿断句」全失效，整段被
max_duration 硬切。本模块复用 kotoba 官方同款标点模型（punctuators 库的
PunctCapSegModelONNX）给每个 chunk 文本补标点，**时间戳原样保留**，使 segment_sentences
能在带句末标点（。！？）的 chunk 处自然断句。

依赖 punctuators（ONNX 推理），故在 CLI 中走延迟导入。逐 chunk 批量推理，效率高。
"""

from __future__ import annotations

from kits.subtitle import Word

__all__ = ["DEFAULT_PUNCT_MODEL", "Punctuator"]

# kotoba 官方同款标点模型
DEFAULT_PUNCT_MODEL = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"

# 已含这些标点的文本视为「已断句」，跳过推理（与官方 kotoba_whisper.py 一致）
_JA_PUNCTUATIONS = ("!", "?", "、", "。", "！", "？")


class Punctuator:
    """日语标点恢复器。延迟加载 ONNX 模型，复用实例可处理多段。"""

    def __init__(self, model: str = DEFAULT_PUNCT_MODEL):
        self.model_id = model
        self._model = None

    def load(self) -> None:
        """加载标点模型。延迟导入 punctuators。"""
        try:
            from punctuators.models import PunctCapSegModelONNX
        except ImportError as e:
            raise RuntimeError(
                "未安装 punctuators，请先安装：uv add punctuators"
            ) from e
        print(f"\n✒️  加载标点恢复模型: {self.model_id}")
        self._model = PunctCapSegModelONNX.from_pretrained(self.model_id)
        print("✅ 标点模型加载完成")

    @staticmethod
    def _needs_punctuation(text: str) -> bool:
        """文本非空且不含任何句读时才需要补标点。"""
        return bool(text.strip()) and not any(p in text for p in _JA_PUNCTUATIONS)

    def restore(self, words: list[Word]) -> list[Word]:
        """给一批 chunk 文本批量补标点，返回时间戳不变、文本带标点的新列表。

        只对「无标点」的 chunk 送模型推理（批量），已含标点或空白的原样保留。
        推理结果含 'unk' 时回退原文（与官方一致），避免乱码污染字幕。
        """
        if not words:
            return []
        if self._model is None:
            self.load()

        # 收集需要推理的下标与文本，批量送一次 infer
        idx_to_infer = [i for i, w in enumerate(words) if self._needs_punctuation(w["text"])]
        if idx_to_infer:
            batch = [words[i]["text"] for i in idx_to_infer]
            results = self._model.infer(batch)
            punctuated: dict[int, str] = {}
            for i, res in zip(idx_to_infer, results):
                text = "".join(res) if isinstance(res, list) else str(res)
                # 含 unk 视为不可靠，回退原文
                if "unk" in text.lower():
                    text = words[i]["text"]
                punctuated[i] = text
        else:
            punctuated = {}

        restored: list[Word] = []
        for i, w in enumerate(words):
            new_text = punctuated.get(i, w["text"])
            restored.append({"text": new_text, "timestamp": w["timestamp"]})
        return restored
