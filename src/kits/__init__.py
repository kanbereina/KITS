"""KITS：直播音频转 SRT 字幕工具。

后续将接入 Twitch 音频下载（downloader）与 DeepSeek 总结分析（summarizer）。
"""

from kits.subtitle import (
    Sentence,
    Word,
    clean_text,
    seconds_to_srt_time,
    segment_sentences,
    sentences_to_srt,
    write_srt,
)

__all__ = [
    "Sentence",
    "Word",
    "clean_text",
    "seconds_to_srt_time",
    "segment_sentences",
    "sentences_to_srt",
    "write_srt",
]
