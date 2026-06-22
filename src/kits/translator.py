# KITS - 鹿乃 Twitch 直播工具
# Copyright (C) 2026 KanbeReina
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""字幕翻译：调用 DeepSeek API 把日语 SRT 翻译成中文 SRT。

仅依赖 httpx（经由 kits.deepseek 公共客户端），不引入 torch / transformers。
消费 kits.subtitle 解析出的句子列表，逐批翻译后保持时间戳不变写回 SRT。
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx

from kits.llm import DEFAULT_MODEL, LLMClient, LLMError
from kits.subtitle import Sentence

__all__ = ["DeepSeekTranslator", "TranslationError"]

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
# LLMError 即 DeepSeekError（同一对象），故继承契约不破
class TranslationError(LLMError):
    """翻译过程中的错误（API 失败、响应解析失败等）。"""


class DeepSeekTranslator:
    """字幕翻译器。按批调用 OpenAI 兼容公共客户端的 chat 接口。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = 20,
        timeout: float = 120.0,
        base_url: str | None = None,
    ):
        try:
            self._client = LLMClient(
                api_key=api_key, model=model, base_url=base_url, timeout=timeout
            )
        except LLMError as e:
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

    def translate_iter(
        self, sentences: list[Sentence]
    ) -> Iterator[tuple[list[Sentence], int, int]]:
        """逐批翻译并流式产出，便于调用方边翻译边写盘、显示进度。

        每翻完一批 yield ``(本批译好的句子, 已完成条数, 总条数)``。批次严格按顺序
        串行处理（无并发），故产出顺序与原字幕顺序一致，可安全增量写盘。
        """
        if not sentences:
            return

        total = len(sentences)
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, total, self.batch_size):
                batch = sentences[start : start + self.batch_size]
                content = self._client.chat(
                    _SYSTEM_PROMPT,
                    self._build_user_content(batch),
                    temperature=1.3,
                    client=client,
                )
                texts = self._parse_response(content, len(batch))
                batch_result: list[Sentence] = [
                    {
                        "start": sent["start"],
                        "end": sent["end"],
                        # 译文为空时回退到原文，宁可保留日语也不丢字幕
                        "text": text or sent["text"],
                    }
                    for sent, text in zip(batch, texts)
                ]
                yield batch_result, start + len(batch), total

    def translate(self, sentences: list[Sentence]) -> list[Sentence]:
        """翻译整个句子列表，返回文本替换为中文、时间戳不变的新列表。"""
        result: list[Sentence] = []
        for batch_result, _done, _total in self.translate_iter(sentences):
            result.extend(batch_result)
        return result
