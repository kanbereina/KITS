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

"""KITS：鹿乃 Twitch 直播工具。

子命令：download（yt-dlp 下载音频）、subtitle（音频转 SRT）、translate（日译中）、
separate（人声分离）、summarize（DeepSeek 总结）。

此处仅导出纯逻辑的字幕数据契约与函数（无 torch / httpx 依赖）；
重依赖模块（transcriber/translator/separator/summarizer）按需从各自模块导入，
避免 `import kits` 时无谓加载 GPU 栈或网络栈。
"""

from kits.subtitle import (
    Sentence,
    SrtWriter,
    Word,
    clean_text,
    parse_srt,
    seconds_to_srt_time,
    segment_sentences,
    sentences_to_srt,
    srt_time_to_seconds,
    write_srt,
)

__all__ = [
    "Sentence",
    "SrtWriter",
    "Word",
    "clean_text",
    "parse_srt",
    "seconds_to_srt_time",
    "segment_sentences",
    "sentences_to_srt",
    "srt_time_to_seconds",
    "write_srt",
]
