"""字幕纯逻辑：单词级时间戳 -> 完整句子 -> SRT。

本模块不依赖 torch / transformers，可独立单元测试。
转录（transcriber）产出的单词时间戳在这里被切分成完整句子并生成 SRT。
"""

from __future__ import annotations

import re
from typing import TypedDict

# 判定为句子结尾的标点（日语 + 通用）
SENTENCE_ENDINGS = ("。", "．", ".", "！", "!", "？", "?", "…", "」", "』", "】", "〉", "》")
# 强制断句时优先切分的逗号 / 读点
SOFT_BREAKS = ("、", "，", ",")


class Word(TypedDict):
    """转录器输出的单词级时间戳。

    text: 单词文本
    timestamp: (开始秒, 结束秒)，任一端可能为 None
    """

    text: str
    timestamp: tuple[float | None, float | None]


class Sentence(TypedDict):
    """断句后的一条字幕。"""

    start: float
    end: float
    text: str


def seconds_to_srt_time(seconds: float | None) -> str:
    """把秒数转成 SRT 时间格式 (HH:MM:SS,mmm)，对超 24h、None、负值都做安全处理。"""
    if seconds is None or seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def srt_time_to_seconds(srt_time: str) -> float:
    """把 SRT 时间格式 (HH:MM:SS,mmm) 解析回秒数，是 seconds_to_srt_time 的逆操作。"""
    hms, _, millis = srt_time.strip().partition(",")
    hours, minutes, secs = (int(part) for part in hms.split(":"))
    return hours * 3600 + minutes * 60 + secs + int(millis or 0) / 1000


def clean_text(text: str) -> str:
    """清理重复字符和乱码。"""
    # 同一个小写假名连续出现 3 次以上时只保留 1 个
    text = re.sub(r"([ぁぃぅぇぉっゃゅょ])\1{2,}", r"\1", text)
    # 任意字符连续 4 次以上压缩成 2 次（抑制幻觉式重复）
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    # 去除乱码
    text = re.sub(r"[�]", "", text)
    # 多个空白合并成一个
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _flush(words: list[Word]) -> Sentence | None:
    """把单词缓冲区合并成一条字幕（一个句子）。"""
    if not words:
        return None
    start = words[0]["timestamp"][0]
    end = words[-1]["timestamp"][1]
    text = clean_text("".join(w["text"] for w in words))
    if not text:
        return None
    if start is None:
        start = 0.0
    if end is None or end <= start:
        end = start + 0.5
    return {"start": start, "end": end, "text": text}


def _last_soft_break(words: list[Word]) -> int | None:
    """返回缓冲区中最后一个以逗号 / 读点结尾的单词下标。"""
    for i in range(len(words) - 1, -1, -1):
        if words[i]["text"].strip().endswith(SOFT_BREAKS):
            return i
    return None


def segment_sentences(
    words: list[Word],
    max_gap: float = 0.7,
    max_chars: int = 60,
    max_duration: float = 15.0,
) -> list[Sentence]:
    """把单词级时间戳切分成完整句子。

    断句条件（按优先级）：
      1. 以句子结尾标点结束
      2. 与上一个单词的停顿(无声)超过 max_gap
      3. 超过字符上限 max_chars / 时长上限 max_duration（防止句子失控变长）
    """
    sentences: list[Sentence] = []
    buf: list[Word] = []

    for w in words:
        ts = w.get("timestamp") or (None, None)
        start = ts[0]
        # 停顿过长时，在当前单词之前断句
        if buf:
            prev_end = buf[-1]["timestamp"][1]
            if start is not None and prev_end is not None and (start - prev_end) > max_gap:
                if (sent := _flush(buf)) is not None:
                    sentences.append(sent)
                buf = []

        buf.append(w)
        joined = "".join(x["text"] for x in buf).strip()

        # 遇到句末标点，句子完结
        if joined.endswith(SENTENCE_ENDINGS):
            if (sent := _flush(buf)) is not None:
                sentences.append(sent)
            buf = []
            continue

        # 超过上限：若有逗号则在逗号处切开，剩余部分留到下一句
        duration = 0.0
        first_start = buf[0]["timestamp"][0]
        if start is not None and first_start is not None:
            duration = (ts[1] or start) - first_start
        if len(joined) >= max_chars or duration >= max_duration:
            split_at = _last_soft_break(buf)
            head = buf[: split_at + 1] if split_at is not None else buf
            tail = buf[split_at + 1 :] if split_at is not None else []
            if (sent := _flush(head)) is not None:
                sentences.append(sent)
            buf = tail

    if (sent := _flush(buf)) is not None:
        sentences.append(sent)
    return sentences


def sentences_to_srt(sentences: list[Sentence]) -> str:
    """把句子列表渲染成 SRT 字符串。"""
    blocks = []
    for i, sent in enumerate(sentences, 1):
        start = seconds_to_srt_time(sent["start"])
        end = seconds_to_srt_time(sent["end"])
        blocks.append(f"{i}\n{start} --> {end}\n{sent['text']}\n")
    return "\n".join(blocks)


def write_srt(sentences: list[Sentence], output_file: str) -> None:
    """把句子列表写入 SRT 文件（UTF-8）。"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(sentences_to_srt(sentences))


# 匹配 SRT 时间轴行： 00:00:00,000 --> 00:00:02,600
_SRT_TIME_LINE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)


def parse_srt(content: str) -> list[Sentence]:
    """把 SRT 文本解析成句子列表，是 sentences_to_srt 的逆操作。

    宽松解析：按空行分块，块内首个时间轴行之后的所有行合并为字幕文本。
    缺失序号或多行文本都能容错。
    """
    sentences: list[Sentence] = []
    # 统一换行，按一个或多个空行分块
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        time_idx = next(
            (i for i, ln in enumerate(lines) if _SRT_TIME_LINE.search(ln)), None
        )
        if time_idx is None:
            continue
        match = _SRT_TIME_LINE.search(lines[time_idx])
        text = "\n".join(lines[time_idx + 1 :]).strip()
        if not text:
            continue
        sentences.append(
            {
                "start": srt_time_to_seconds(match.group(1)),
                "end": srt_time_to_seconds(match.group(2)),
                "text": text,
            }
        )
    return sentences
