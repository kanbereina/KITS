"""DeepSeek API 公共客户端：封装鉴权、HTTP 请求与错误处理。

仅依赖 httpx，不引入 torch / transformers。translator 与 summarizer 共用此客户端，
各自只负责领域逻辑（如翻译的分批拼装、总结的提示词组织），网络细节集中在这里。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

__all__ = [
    "DEEPSEEK_API_URL",
    "DEFAULT_MODEL",
    "DeepSeekClient",
    "DeepSeekError",
]

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekError(RuntimeError):
    """DeepSeek 调用过程中的错误（缺 Key、API 失败、响应解析失败等）。"""


class DeepSeekClient:
    """DeepSeek chat/completions 客户端。

    负责：解析 API Key（参数优先于环境变量 DEEPSEEK_API_KEY）、发起单次请求、
    解析出回复文本。批处理 / 重试等领域策略由调用方实现。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise DeepSeekError(
                "缺少 DeepSeek API Key，请用 --api-key 传入或设置环境变量 DEEPSEEK_API_KEY"
            )
        self.api_key = key
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
        resp = client.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self._build_payload(
                self.model, system_prompt, user_content, temperature
            ),
        )
        if resp.status_code != 200:
            raise DeepSeekError(
                f"DeepSeek API 返回 HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise DeepSeekError(f"解析 DeepSeek 响应失败: {e}") from e
