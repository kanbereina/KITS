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

"""字幕纯逻辑：单词级时间戳 -> 完整句子 -> SRT。

本模块不依赖 torch / transformers，可独立单元测试。
转录（transcriber）产出的单词时间戳在这里被切分成完整句子并生成 SRT。
"""

from __future__ import annotations

import bisect
import re
from typing import TypedDict

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
    text = re.sub(r"�", "", text)
    # 多个空白合并成一个
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _vad_voice_end(
    speech: list[tuple[float, float]], speech_starts: list[float], start: float
) -> float | None:
    """给定句子起点 start，返回「人声实际结束处」——从 start 起第一个 VAD 人声区间的 end。

    用于把 kotoba 虚高的 chunk end 夹回真实人声边界（见 _flush）。speech 为 VAD 探出的
    人声区间 [(s,e),...]（按起点升序），speech_starts 是各区间起点的预提取列表（bisect 用）。
      - start 落在某人声区间内 → 返回该区间 end
      - 否则取 start 之后第一个人声区间 → 返回其 end
      - start 在所有人声区间之后（无后继）→ None（VAD 帮不上，调用方回退字符速率）
    """
    if not speech:
        return None
    # 第一个起点 > start 的区间下标；其前一个（若存在）起点 <= start
    i = bisect.bisect_right(speech_starts, start)
    if i > 0 and speech[i - 1][1] > start:
        return speech[i - 1][1]  # start 落在 speech[i-1] 区间内
    if i < len(speech):
        return speech[i][1]  # start 之后第一个人声区间
    return None


def _flush(
    words: list[Word],
    max_duration: float | None = None,
    max_seconds_per_char: float | None = None,
    min_duration: float = 0.5,
    speech: list[tuple[float, float]] | None = None,
    speech_starts: list[float] | None = None,
    vad_ratio_threshold: float = 1.0,
) -> Sentence | None:
    """把单词缓冲区合并成一条字幕（一个句子）。

    max_duration 非 None 时，对超过该时长的句子做硬钳制（end = start + max_duration），
    兜底那些词级时间戳缺失、导致前面断句规则失效而拉得过长的句子。

    虚高收缩（治 kotoba chunk 时间戳跨度虚高：十来个字的短句标成十几秒）分两路，
    **VAD 优先、字符速率回退**：
      - speech（VAD 人声区间）可用、且本句明显虚高（时长 / 字符 > vad_ratio_threshold）时，
        优先用 _vad_voice_end 找「人声实际结束处」夹 end——真实测量，比字符速率猜测准
        （实测能修正约 82% 的虚高条目，VAD 边界偏差 0.1~0.3s vs 字符估计 2.5~3.5s）。
      - VAD 对不上（start 在所有人声区间之后、或夹出的 end 不落在 (start, end) 内）时，
        回退按 max_seconds_per_char「每字符最多 N 秒」收缩。二者单点决策、互斥，避免叠加
        误判；都只缩短显示时长、不改文本与起点，故不会与下一条重叠。

    speech 为 None / 空（短音频跳过 VAD、或调用方不传）时退化为纯字符速率收缩，与引入
    VAD 前行为完全一致。日语正常语速 ≥2 字/秒（每字 ≤0.5s），max_seconds_per_char 取
    0.5 偏宽松、只砍明显虚高者；vad_ratio_threshold 取 1.0 更严，只对极虚高者动 VAD。

    min_duration：显示时长下界。收缩后过短的字幕（kotoba 偶把 start/end 挤在一点，
    如 0.04s）撑到此值，保证播放器里可读。
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
    # 硬钳：超 max_duration 先夹（兜底词级时间戳缺失导致的失控长句）
    if max_duration is not None and (end - start) > max_duration:
        end = start + max_duration
    # 虚高收缩：VAD 优先、字符速率回退（互斥单点决策）
    clamped_by_vad = False
    if (
        speech
        and speech_starts is not None
        and (end - start) / len(text) > vad_ratio_threshold
    ):
        ve = _vad_voice_end(speech, speech_starts, start)
        if ve is not None and start < ve < end:
            end = ve
            clamped_by_vad = True
    if not clamped_by_vad and max_seconds_per_char is not None:
        char_limit = len(text) * max_seconds_per_char
        if (end - start) > char_limit:
            end = start + char_limit
    # 最后兜底下界：经上面各步收缩后仍可能过短（原始时间戳就挤在一点），撑到 min_duration
    if (end - start) < min_duration:
        end = start + min_duration
    # noinspection PyTypeChecker
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
        # 按字符比例把 [start, end] 切成若干片段。meaningful 已滤掉末尾切点
        # （c < len(text)），故末尾必有残余文本，补上 total 作为最后一段右界。
        total = len(text)
        span = end - start
        prev = 0
        bounds = [*meaningful, total]
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
    max_seconds_per_char: float = 0.5,
    speech: list[tuple[float, float]] | None = None,
    vad_ratio_threshold: float = 1.0,
) -> list[Sentence]:
    """把单词级时间戳切分成完整句子。

    断句条件（按优先级）：
      1. 以句子结尾标点结束（含 chunk 内部的句末标点——先经 _split_internal_punctuation
         把含内部标点的 chunk 拆开，使句中的 。？！ 也能触发断句）
      2. 与上一个单词的停顿(无声)超过 max_gap
      3. 超过字符上限 max_chars / 时长上限 max_duration（防止句子失控变长）

    虚高 end 收缩（治 kotoba chunk 时间戳跨度虚高，短句标成 max_duration）：**VAD 优先、
    字符速率回退**（见 _flush）。speech 为 VAD 探出的人声区间 [(s,e),...]（全局时间轴、
    与 words 时间戳同轴），传入则对明显虚高（时长/字符 > vad_ratio_threshold）的句子按
    人声实际结束处夹 end；VAD 对不上或未传 speech 时，回退 max_seconds_per_char 按字符
    速率收缩（<=0 关闭该回退）。speech 为 None 时行为与引入 VAD 前完全一致。
    """
    spc = max_seconds_per_char if max_seconds_per_char and max_seconds_per_char > 0 else None
    # 预提取人声区间起点供 _vad_voice_end 二分；speech 须按起点升序（silero 输出即如此）
    speech_starts = [s for s, _ in speech] if speech else None

    def flush(buf: list[Word]) -> Sentence | None:
        return _flush(buf, max_duration, spc, speech=speech,
                      speech_starts=speech_starts, vad_ratio_threshold=vad_ratio_threshold)

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
                if (sent := flush(buf)) is not None:
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
            if (sent := flush(buf)) is not None:
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
            if (sent := flush(head)) is not None:
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

    if (sent := flush(buf)) is not None:
        sentences.append(sent)
    return _clamp_overlaps(sentences)


def _clamp_overlaps(sentences: list[Sentence]) -> list[Sentence]:
    """把每条 end 夹到不超过下一条 start，消除相邻字幕的时间重叠。

    重叠来源有二：min_duration 把近零时长条目撑长后越过了紧邻的下一条；分段转录
    接缝处 overlap 垫料的边界效应。两者都表现为「前条 end 略大于后条 start」。
    就地夹紧 end=min(end, 下一条 start)即可消除。仅缩短显示时长、不动文本与起点。

    若后一条 start <= 本条 start（模型把两句标在同一时刻），夹紧会使本条零时长——
    此时优先「不重叠」、容忍零时长，因为这种同刻并发本就无法既不重叠又非零时长。
    """
    for i in range(len(sentences) - 1):
        nxt_start = sentences[i + 1]["start"]
        if sentences[i]["end"] > nxt_start:
            sentences[i]["end"] = max(sentences[i]["start"], nxt_start)
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
        # noinspection PyUnresolvedReferences
        sentences.append(
            {
                "start": srt_time_to_seconds(match.group(1)),
                "end": srt_time_to_seconds(match.group(2)),
                "text": text,
            }
        )
    return sentences
