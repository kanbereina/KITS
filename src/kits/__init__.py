"""KITS：鹿乃 Twitch 直播工具。

子命令：download（下载合并 MP4）、subtitle（音频转 SRT）、translate（日译中）、
separate（人声分离）、sum（DeepSeek 总结）。

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
