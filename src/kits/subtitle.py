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
    # 2~8 字的短语连续重复 3 次以上压成 2 次（抑制「はっはっはっ」「ここだここだ」式幻觉）
    text = re.sub(r"(.{2,8}?)\1{2,}", r"\1\1", text)
    # 去除乱码
    text = re.sub(r"[�]", "", text)
    # 多个空白合并成一个
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _flush(words: list[Word], max_duration: float | None = None) -> Sentence | None:
    """把单词缓冲区合并成一条字幕（一个句子）。

    max_duration 非 None 时，对超过该时长的句子做硬钳制（end = start + max_duration），
    兜底那些词级时间戳缺失、导致前面断句规则失效而拉得过长的句子。
    """
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
    if max_duration is not None and (end - start) > max_duration:
        end = start + max_duration
    return {"start": start, "end": end, "text": text}


def _last_soft_break(words: list[Word]) -> int | None:
    """返回缓冲区中最后一个以逗号 / 读点结尾的单词下标。"""
    for i in range(len(words) - 1, -1, -1):
        if words[i]["text"].strip().endswith(SOFT_BREAKS):
            return i
    return None


# 句末标点集合（用于在 chunk 文本内部查找切点）。注意排除 SENTENCE_ENDINGS 里的引号类
# （」』】〉》）——它们只在「整段以其结尾」时才算句末，出现在文本中间不宜就地切。
_INNER_ENDINGS = ("。", "．", ".", "！", "!", "？", "?", "…")


def _split_internal_punctuation(words: list[Word]) -> list[Word]:
    """把「文本内部含句末标点」的 chunk 拆成多个 Word，时间戳按字符长度比例分配。

    kotoba 等蒸馏模型的标点恢复是逐 chunk 补的，一个 chunk 内可能含多句（如
    「そう?どうだろう。こんにちは」），其句末标点在文本中间，segment_sentences 的
    「chunk 结尾判标点」断不开，只能等 max_duration 硬钳。这里预处理：在每个句末标点
    之后切开成独立 Word，使后续断句能在标点处自然分段。时间戳缺失则原样透传不拆。
    """
    result: list[Word] = []
    for w in words:
        text = w["text"]
        ts = w.get("timestamp") or (None, None)
        start, end = ts
        # 找出所有「句末标点」在文本中的结束位置（标点之后即为切点）
        cuts = [
            i + 1
            for i, ch in enumerate(text)
            if ch in _INNER_ENDINGS
        ]
        # 无内部标点、或标点恰在末尾（只有一个切点且等于文本长度）、或时间戳缺失 → 不拆
        meaningful = [c for c in cuts if 0 < c < len(text)]
        if not meaningful or start is None or end is None or end <= start:
            result.append(w)
            continue
        # 按字符比例把 [start, end] 切成若干片段
        total = len(text)
        span = end - start
        prev = 0
        bounds = [*meaningful, total] if meaningful[-1] != total else meaningful
        for b in bounds:
            piece = text[prev:b]
            if piece:
                seg_start = start + span * (prev / total)
                seg_end = start + span * (b / total)
                result.append({"text": piece, "timestamp": (seg_start, seg_end)})
            prev = b
    return result


def segment_sentences(
    words: list[Word],
    max_gap: float = 0.7,
    max_chars: int = 60,
    max_duration: float = 15.0,
) -> list[Sentence]:
    """把单词级时间戳切分成完整句子。

    断句条件（按优先级）：
      1. 以句子结尾标点结束（含 chunk 内部的句末标点——先经 _split_internal_punctuation
         把含内部标点的 chunk 拆开，使句中的 。？！ 也能触发断句）
      2. 与上一个单词的停顿(无声)超过 max_gap
      3. 超过字符上限 max_chars / 时长上限 max_duration（防止句子失控变长）
    """
    words = _split_internal_punctuation(words)
    sentences: list[Sentence] = []
    buf: list[Word] = []
    # 缓冲区内最后一个已知的 end（应对词级时间戳缺失：Whisper 常吐 None）
    last_known_end: float | None = None
    # 缓冲区内第一个已知的 start
    first_known_start: float | None = None

    for w in words:
        ts = w.get("timestamp") or (None, None)
        start = ts[0]
        # 停顿过长时，在当前单词之前断句
        if buf:
            prev_end = last_known_end
            if start is not None and prev_end is not None and (start - prev_end) > max_gap:
                if (sent := _flush(buf, max_duration)) is not None:
                    sentences.append(sent)
                buf = []
                last_known_end = None
                first_known_start = None

        buf.append(w)
        if start is not None and first_known_start is None:
            first_known_start = start
        if ts[1] is not None:
            last_known_end = ts[1]
        joined = "".join(x["text"] for x in buf).strip()

        # 遇到句末标点，句子完结
        if joined.endswith(SENTENCE_ENDINGS):
            if (sent := _flush(buf, max_duration)) is not None:
                sentences.append(sent)
            buf = []
            last_known_end = None
            first_known_start = None
            continue

        # 超过上限：若有逗号则在逗号处切开，剩余部分留到下一句
        # 用「最后已知 end - 第一个已知 start」估算时长，不依赖当前词的时间戳
        duration = 0.0
        if last_known_end is not None and first_known_start is not None:
            duration = last_known_end - first_known_start
        if len(joined) >= max_chars or duration >= max_duration:
            split_at = _last_soft_break(buf)
            if split_at is not None:
                head = buf[: split_at + 1]
                tail = buf[split_at + 1 :]
            else:
                # 没有逗号可切：按词边界硬切，避免整段被 flush 成一条超长字幕
                head = buf
                tail = []
            if (sent := _flush(head, max_duration)) is not None:
                sentences.append(sent)
            buf = tail
            # 重算尾部的时间戳基准
            last_known_end = next(
                (t[1] for tw in reversed(tail) if (t := tw["timestamp"])[1] is not None),
                None,
            )
            first_known_start = next(
                (t[0] for tw in tail if (t := tw["timestamp"])[0] is not None),
                None,
            )

    if (sent := _flush(buf, max_duration)) is not None:
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


class SrtWriter:
    """增量 SRT 写入器：维护全局递增序号，分批追加写盘。

    用于分段转录场景——每段转完即把该段句子追加进同一个 SRT 文件，
    序号跨段连续，中途中断时已写入的部分仍是合法 SRT。
    """

    def __init__(self, output_file: str):
        self.output_file = output_file
        self._index = 0
        # 开始时清空/创建文件。显式 truncate：若上次运行被强杀留下残留文件，
        # "w" 在个别平台/文件系统上不保证把旧内容截断到新长度，可能残留上次更长的尾部
        # （表现为文件中段出现 NUL 空洞 + 旧字幕尾巴）。truncate(0) 强制清零长度兜底。
        self._fh = open(output_file, "w", encoding="utf-8")
        self._fh.truncate(0)

    def append(self, sentences: list[Sentence]) -> int:
        """追加一批句子，返回累计已写入条数。"""
        for sent in sentences:
            self._index += 1
            start = seconds_to_srt_time(sent["start"])
            end = seconds_to_srt_time(sent["end"])
            # 每块之间用空行分隔；第一块前不需要
            prefix = "" if self._index == 1 else "\n"
            self._fh.write(f"{prefix}{self._index}\n{start} --> {end}\n{sent['text']}\n")
        self._fh.flush()
        return self._index

    @property
    def count(self) -> int:
        """已写入的字幕条数。"""
        return self._index

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> SrtWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


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
