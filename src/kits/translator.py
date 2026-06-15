"""字幕翻译：调用 DeepSeek API 把日语 SRT 翻译成中文 SRT。

仅依赖 httpx（经由 kits.deepseek 公共客户端），不引入 torch / transformers。
消费 kits.subtitle 解析出的句子列表，逐批翻译后保持时间戳不变写回 SRT。
"""

from __future__ import annotations

import httpx

from kits.deepseek import DEFAULT_MODEL, DeepSeekClient, DeepSeekError
from kits.subtitle import Sentence

# 翻译系统提示：要求逐条对应、只输出译文、保持顺序与条数一致
_SYSTEM_PROMPT = (
    "你是一个专业的字幕翻译。将用户给出的日语字幕逐条翻译成简体中文。\n"
    "要求：\n"
    "1. 输入是带编号的多条字幕，每条以「序号|||原文」的形式给出。\n"
    "2. 必须逐条翻译，输出同样数量的行，每行格式为「序号|||译文」。\n"
    "3. 保持序号与原文一一对应，不要合并、拆分、增删任何一条。\n"
    "4. 只输出译文行，不要任何解释、前后缀或代码块标记。\n"
    "5. 翻译要自然口语化，符合中文表达习惯，保留语气。"
)


# 向后兼容：旧代码 / 测试可能仍 import TranslationError
class TranslationError(DeepSeekError):
    """翻译过程中的错误（API 失败、响应解析失败等）。"""


class DeepSeekTranslator:
    """DeepSeek 字幕翻译器。按批调用公共客户端的 chat 接口。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = 20,
        timeout: float = 120.0,
    ):
        try:
            self._client = DeepSeekClient(api_key=api_key, model=model, timeout=timeout)
        except DeepSeekError as e:
            # 保持对外抛 TranslationError 的历史契约
            raise TranslationError(str(e)) from e
        self.batch_size = batch_size

    @property
    def api_key(self) -> str:
        return self._client.api_key

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def timeout(self) -> float:
        return self._client.timeout

    @staticmethod
    def _build_user_content(batch: list[Sentence]) -> str:
        """把一批字幕拼成「序号|||原文」的多行文本。"""
        return "\n".join(f"{i}|||{s['text']}" for i, s in enumerate(batch))

    @staticmethod
    def _parse_response(content: str, count: int) -> list[str]:
        """解析「序号|||译文」多行响应，按序号回填，缺失项留空。"""
        translations: list[str | None] = [None] * count
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or "|||" not in line:
                continue
            idx_str, _, text = line.partition("|||")
            try:
                idx = int(idx_str.strip())
            except ValueError:
                continue
            if 0 <= idx < count:
                translations[idx] = text.strip()
        # 缺失的条目用占位，避免时间轴错位
        return [t if t is not None else "" for t in translations]

    def translate(self, sentences: list[Sentence]) -> list[Sentence]:
        """翻译整个句子列表，返回文本替换为中文、时间戳不变的新列表。"""
        if not sentences:
            return []

        total = len(sentences)
        result: list[Sentence] = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, total, self.batch_size):
                batch = sentences[start : start + self.batch_size]
                batch_no = start // self.batch_size + 1
                batch_total = (total + self.batch_size - 1) // self.batch_size
                print(
                    f"🌐 翻译批次 {batch_no}/{batch_total} "
                    f"（{start + 1}-{start + len(batch)}/{total} 条）..."
                )
                content = self._client.chat(
                    _SYSTEM_PROMPT,
                    self._build_user_content(batch),
                    temperature=1.3,
                    client=client,
                )
                texts = self._parse_response(content, len(batch))
                for sent, text in zip(batch, texts):
                    # 译文为空时回退到原文，宁可保留日语也不丢字幕
                    result.append(
                        {
                            "start": sent["start"],
                            "end": sent["end"],
                            "text": text or sent["text"],
                        }
                    )
        print(f"✅ 翻译完成，共 {len(result)} 条字幕")
        return result
