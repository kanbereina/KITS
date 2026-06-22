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

"""OpenAI 兼容 LLM 公共客户端：封装 base_url / 鉴权 / HTTP 请求与错误处理。

仅依赖 httpx，不引入 torch / transformers。translator 与 summarizer 共用此客户端，
各自只负责领域逻辑（如翻译的分批拼装、总结的提示词组织），网络细节集中在这里。

请求体 / 响应解析为 OpenAI 标准格式（/chat/completions、Bearer 鉴权、
choices[0].message.content），故任意 OpenAI 兼容端点（DeepSeek / OpenAI /
Ollama / vLLM / LM Studio 等）都能接入。默认 base_url 指向 DeepSeek，零配置时
行为与历史一致。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMClient",
    "LLMError",
]

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

_CHAT_COMPLETIONS_PATH = "/chat/completions"


class LLMError(RuntimeError):
    """LLM 调用过程中的错误（缺 Key、API 失败、响应解析失败等）。"""


def _resolve_endpoint(base_url: str) -> str:
    """把 base_url 规整成完整的 chat/completions endpoint。

    去掉结尾的 `/`；若已以 `/chat/completions` 结尾则视为完整 endpoint 直接用，
    否则拼上该路径。兼容用户传完整 URL 或仅传 base（如 OpenAI 的 .../v1）。
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith(_CHAT_COMPLETIONS_PATH):
        return trimmed
    return trimmed + _CHAT_COMPLETIONS_PATH


_DEFAULT_ENDPOINT = _resolve_endpoint(DEFAULT_BASE_URL)


class LLMClient:
    """OpenAI 兼容 chat/completions 客户端。

    负责：解析 base_url（参数 > 环境变量 KITS_LLM_BASE_URL > 默认 DeepSeek）、
    解析 API Key（参数 > KITS_LLM_API_KEY > OPENAI_API_KEY > DEEPSEEK_API_KEY）、
    发起单次请求、解析出回复文本。批处理 / 重试等领域策略由调用方实现。

    空 Key 策略：默认端点（DeepSeek）仍强制要求 Key；用户显式改了 base_url
    （如本地 Ollama）则允许空 Key——本地端点常不需要鉴权。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        resolved_base = base_url or os.environ.get("KITS_LLM_BASE_URL") or DEFAULT_BASE_URL
        self.endpoint = _resolve_endpoint(resolved_base)

        key = (
            api_key
            or os.environ.get("KITS_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
        )
        # 仅默认端点（DeepSeek）强制要求 Key；自定义端点允许空 Key（本地模型无需鉴权）
        if not key and self.endpoint == _DEFAULT_ENDPOINT:
            raise LLMError(
                "缺少 API Key，请用 --api-key 传入或设置环境变量 "
                "KITS_LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY"
            )
        self.api_key = key or ""
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 1.0,
        client: httpx.Client | None = None,
    ) -> str:
        """发一次 system+user 对话，返回助手回复文本。

        client 非 None 时复用该连接（批量调用场景），否则临时开一个。
        """
        if client is not None:
            return self._post(client, system_prompt, user_content, temperature)
        with httpx.Client(timeout=self.timeout) as own:
            return self._post(own, system_prompt, user_content, temperature)

    @staticmethod
    def _build_payload(
        model: str, system_prompt: str, user_content: str, temperature: float
    ) -> dict[str, Any]:
        """组装 chat/completions 请求体。"""
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "stream": False,
        }

    def _post(
        self,
        client: httpx.Client,
        system_prompt: str,
        user_content: str,
        temperature: float,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        # 空 Key（本地端点）不强加 Authorization 头
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = client.post(
            self.endpoint,
            headers=headers,
            json=self._build_payload(
                self.model, system_prompt, user_content, temperature
            ),
        )
        if resp.status_code != 200:
            raise LLMError(
                f"LLM API 返回 HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise LLMError(f"解析 LLM 响应失败: {e}") from e
