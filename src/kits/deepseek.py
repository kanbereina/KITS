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

"""DeepSeek 公共客户端的向后兼容转发层。

实际实现已泛化为 OpenAI 兼容客户端，迁移到 kits.llm。本模块仅 re-export 并保留
DeepSeekClient / DeepSeekError 别名，使历史引用（测试 / 文档示例）继续可用。
新代码请直接用 kits.llm.LLMClient。
"""

from __future__ import annotations

from kits.llm import (
    DEEPSEEK_API_URL,
    DEFAULT_MODEL,
    LLMClient,
    LLMError,
)

# 向后兼容别名：旧代码 / 测试仍 import 这两个名字
DeepSeekClient = LLMClient
DeepSeekError = LLMError

__all__ = [
    "DEEPSEEK_API_URL",
    "DEFAULT_MODEL",
    "DeepSeekClient",
    "DeepSeekError",
]
