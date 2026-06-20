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

"""字幕总结：调用 DeepSeek 把已有 SRT 字幕总结成可读的回顾文本。

仅依赖 httpx（经由 kits.deepseek 公共客户端），不引入 torch / transformers。
消费 kits.subtitle.parse_srt 产出的句子列表。

提示词走 JSON 预设：包内置 prompts.json 提供多种预设（timeline / summary /
highlights / setlist），用户可用 --prompt-file 传入自定义 JSON 覆盖。长字幕用
map-reduce：先按字符预算分块、逐块总结，再把分段总结合并成最终结果。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from kits.deepseek import DEFAULT_MODEL, DeepSeekClient, DeepSeekError
from kits.subtitle import Sentence, seconds_to_srt_time

__all__ = [
    "PromptPreset",
    "PromptsConfig",
    "SummarizeError",
    "Summarizer",
    "available_presets",
    "chunk_sentences",
    "format_sentences",
    "load_presets",
    "resolve_preset",
]

# 包内置预设文件（随包发布的包数据，置于 data/ 子目录与 .py 源码分离）
_BUILTIN_PROMPTS = Path(__file__).with_name("data") / "prompts.json"


class SummarizeError(RuntimeError):
    """总结过程中的错误（预设缺失、API 失败等）。"""


class PromptPreset(BaseModel):
    """单个总结预设：描述 + system 提示词。"""

    description: str = ""
    system: str = Field(min_length=1)


class PromptsConfig(BaseModel):
    """提示词配置整体结构。

    加载时即校验：presets 至少一项、每项含非空 system，default（若给）必须存在于
    presets。坏配置在此处即抛清晰错误，而非延迟到取 presets[name]["system"] 时才 KeyError。
    """

    presets: dict[str, PromptPreset] = Field(min_length=1)
    default: str | None = None
    reduce_system: str = ""

    def resolve(self, name: str | None) -> tuple[str, str]:
        """解析预设名，返回 (预设名, system 提示词)。

        name 为 None 时用 default。未知预设抛 SummarizeError 并列出可用项。
        """
        chosen = name or self.default
        if chosen not in self.presets:
            raise SummarizeError(
                f"不支持的总结预设: {chosen!r}。当前可用: {', '.join(sorted(self.presets))}"
            )
        # noinspection PyTypeChecker
        return chosen, self.presets[chosen].system

    @property
    def preset_names(self) -> list[str]:
        """所有可用预设名（已排序）。"""
        return sorted(self.presets)


def _hms(seconds: float) -> str:
    """秒 -> HH:MM:SS（丢掉毫秒，总结里用不到那么细）。"""
    return seconds_to_srt_time(seconds).split(",")[0]


def load_presets(prompt_file: str | None = None) -> PromptsConfig:
    """加载并校验提示词预设，返回 PromptsConfig。

    先读包内置 prompts.json；若传入 prompt_file，则用其内容覆盖（浅合并 presets、
    顶层键如 default/reduce_system 也可覆盖）。结构错误（缺 system、presets 为空等）
    在 pydantic 校验阶段即抛 SummarizeError。
    """
    config: dict = json.loads(_BUILTIN_PROMPTS.read_text(encoding="utf-8"))

    if prompt_file:
        path = Path(prompt_file)
        if not path.is_file():
            raise SummarizeError(f"找不到提示词文件: {path}")
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SummarizeError(f"提示词文件不是合法 JSON: {e}") from e
        # 合并：用户 presets 并入内置，其余顶层键直接覆盖
        merged_presets = {**config.get("presets", {}), **user.get("presets", {})}
        config.update(user)
        config["presets"] = merged_presets

    try:
        return PromptsConfig.model_validate(config)
    except ValidationError as e:
        raise SummarizeError(f"提示词配置结构不合法: {e}") from e


def available_presets(prompt_file: str | None = None) -> list[str]:
    """返回所有可用预设名（已排序），用于 CLI 帮助 / 报错提示。"""
    return load_presets(prompt_file).preset_names


def resolve_preset(name: str | None, prompt_file: str | None = None) -> tuple[str, str]:
    """解析预设名，返回 (预设名, system 提示词)。

    name 为 None 时用配置里的 default。未知预设抛 SummarizeError 并列出可用项。
    """
    return load_presets(prompt_file).resolve(name)


def format_sentences(sentences: list[Sentence]) -> str:
    """把句子列表渲染成「[HH:MM:SS-HH:MM:SS] 文本」多行，喂给模型。"""
    return "\n".join(
        f"[{_hms(s['start'])}-{_hms(s['end'])}] {s['text']}" for s in sentences
    )


def chunk_sentences(
    sentences: list[Sentence], max_chars: int = 8000
) -> list[list[Sentence]]:
    """按渲染后字符预算把句子切成若干块，长字幕分块总结用。

    以每条渲染文本的长度累加估算，超过 max_chars 即开新块。单条超长也自成一块。
    """
    if not sentences:
        return []

    chunks: list[list[Sentence]] = []
    current: list[Sentence] = []
    size = 0
    for s in sentences:
        line_len = len(f"[{_hms(s['start'])}-{_hms(s['end'])}] {s['text']}") + 1
        if current and size + line_len > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(s)
        size += line_len
    if current:
        chunks.append(current)
    return chunks


class Summarizer:
    """DeepSeek 字幕总结器。按预设提示词总结字幕，长字幕走 map-reduce。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        preset: str | None = None,
        prompt_file: str | None = None,
        max_chars: int = 8000,
        timeout: float = 180.0,
    ):
        # 先解析预设（可能抛 SummarizeError），再建客户端
        config = load_presets(prompt_file)
        self.preset_name, self._system = config.resolve(preset)
        self._reduce_system = config.reduce_system
        try:
            self._client = DeepSeekClient(api_key=api_key, model=model, timeout=timeout)
        except DeepSeekError as e:
            raise SummarizeError(str(e)) from e
        self.max_chars = max_chars

    def summarize(self, sentences: list[Sentence]) -> str:
        """总结整个句子列表，返回总结文本（Markdown）。"""
        if not sentences:
            raise SummarizeError("没有可总结的字幕内容")

        chunks = chunk_sentences(sentences, self.max_chars)
        print(f"📝 使用预设「{self.preset_name}」，分 {len(chunks)} 块总结")

        partials: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            print(f"🤖 总结第 {i}/{len(chunks)} 块（{len(chunk)} 条字幕）...")
            text = self._client.chat(
                self._system, format_sentences(chunk), temperature=1.0
            )
            partials.append(text.strip())

        # 单块直接返回；多块再做一次合并（reduce）
        if len(partials) == 1:
            print("✅ 总结完成")
            return partials[0]

        print("🔗 合并分段总结...")
        combined = "\n\n".join(
            f"【第{i}段】\n{p}" for i, p in enumerate(partials, 1)
        )
        reduce_system = self._reduce_system or self._system
        final = self._client.chat(reduce_system, combined, temperature=1.0)
        print("✅ 总结完成")
        return final.strip()
