"""音频转录：封装 Whisper pipeline 的加载与转录（默认 kotoba-whisper-v2.2）。

产出单词级时间戳列表（list[Word]），交给 kits.subtitle 断句生成 SRT。
模型转录文本同样可直接喂给后续的 DeepSeek 总结模块。

长音频支持「按静音切分 + 分段转录」：先用 ffmpeg silencedetect 探测静音区间，
在静音中点设切点把音频切成若干段，逐段转录后追加产出（边转边出，可显示进度、
分批写盘）。切点都落在无人说话处，故不会把句子拦腰截断，精度不受影响。
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import pipeline

from kits.subtitle import Word

MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2"


def _silence_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    warnings.filterwarnings("ignore", message=".*custom logits processor.*")


def require_cuda() -> str:
    """检测 CUDA 是否可用，不可用则抛错。返回设备名 "cuda"。"""
    print(f"\n📦 PyTorch 版本: {torch.__version__}")
    print(f"💻 CUDA 可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("请安装 GPU 版本的 CUDA！")
    print(f"🔧 CUDA 版本: {torch.version.cuda}")
    print(f"🎮 GPU 数量: {torch.cuda.device_count()}")
    print(f"🖥️  当前 GPU: {torch.cuda.get_device_name(0)}")
    return "cuda"


# silencedetect 输出形如：[silencedetect @ ...] silence_start: 12.34
_SILENCE_START = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)")


def probe_duration(audio_file: str) -> float:
    """用 ffprobe 取音频总时长（秒）。失败抛 RuntimeError。"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 获取时长失败: {result.stderr[:300]}")
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise RuntimeError(f"无法解析音频时长: {result.stdout[:100]}") from e


def detect_silences(
    audio_file: str, noise_db: float = -45.0, min_silence: float = 0.5
) -> list[tuple[float, float]]:
    """用 ffmpeg silencedetect 探测静音区间，返回 [(start, end), ...]（秒）。

    noise_db: 音量低于该响度视为静音（越负越严格、检出静音越少，越接近 0 越宽松）。
    min_silence: 最短静音时长，短于此不计入。
    """
    cmd = [
        "ffmpeg", "-i", audio_file,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    # silencedetect 把结果打到 stderr
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log = result.stderr

    silences: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in log.splitlines():
        if (m := _SILENCE_START.search(line)) is not None:
            pending_start = float(m.group(1))
        elif (m := _SILENCE_END.search(line)) is not None and pending_start is not None:
            silences.append((pending_start, float(m.group(1))))
            pending_start = None
    return silences


def plan_segments(
    duration: float,
    silences: list[tuple[float, float]],
    target_chunk: float = 300.0,
    max_chunk: float = 600.0,
    fallback_silences: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """根据静音区间规划分段切点，返回 [(start, end), ...] 覆盖 [0, duration]。

    策略：从当前段起点出发，在 [起点+target_chunk 之后的第一个静音中点] 处切；
    若到 起点+max_chunk 仍无（严格阈值的）静音可切（如长时间唱歌），先在
    fallback_silences（更宽松阈值探到的静音）里找一个落在 (soft_limit, hard_limit)
    的中点退而求其次；仍无才在 max_chunk 处强切兜底。切点优先落在静音中点，不打断语句。

    fallback_silences 为 None 时退化为原行为（无二次探测），保证既有契约不变。
    """
    if duration <= max_chunk:
        return [(0.0, duration)]

    # 静音中点列表（升序），作为候选切点
    midpoints = sorted((s + e) / 2 for s, e in silences)
    # 宽松阈值探到的备选中点（升序），仅在严格候选缺位时启用
    fb_midpoints = sorted((s + e) / 2 for s, e in (fallback_silences or []))

    segments: list[tuple[float, float]] = []
    start = 0.0
    idx = 0
    while start < duration:
        soft_limit = start + target_chunk
        hard_limit = start + max_chunk
        # 跳过落在软下限之前的候选切点
        while idx < len(midpoints) and midpoints[idx] <= soft_limit:
            idx += 1
        cut = None
        if idx < len(midpoints) and midpoints[idx] < hard_limit:
            cut = midpoints[idx]
            idx += 1
        else:
            # 严格阈值无候选：在宽松候选里找落在 (soft_limit, hard_limit) 的第一个
            cut = next(
                (m for m in fb_midpoints if soft_limit < m < hard_limit),
                None,
            )
            if cut is None:
                # 宽松候选也没有：强切在硬上限兜底
                cut = hard_limit
        end = min(cut, duration)
        segments.append((start, end))
        start = end
    return segments


def slice_audio(audio_file: str, start: float, end: float, out_path: Path) -> None:
    """用 ffmpeg 切出 [start, end] 区间到 out_path（重编码为 16k 单声道 wav）。

    用 `-ss {start} -i ... -t {时长}`（seek 到起点后读固定时长）而非 `-to {绝对结束}`：
    `-to` 作为输入选项时，不同 ffmpeg 版本对「绝对时间轴 vs 相对 seek 点」解释不一，
    可能切错段长；`-t` 取时长则语义稳定，跨版本一致。
    """
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg", "-ss", f"{start:.3f}", "-i", audio_file,
        "-t", f"{duration:.3f}",
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_path), "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"切分音频失败 [{start:.1f}-{end:.1f}]: {result.stderr[:300]}")


def _shift_words(words: list[Word], offset: float) -> list[Word]:
    """把一段内的词级时间戳整体加上段起始偏移，回到全局时间轴。"""
    shifted: list[Word] = []
    for w in words:
        ts = w.get("timestamp") or (None, None)
        start = None if ts[0] is None else ts[0] + offset
        end = None if ts[1] is None else ts[1] + offset
        shifted.append({"text": w["text"], "timestamp": (start, end)})
    return shifted


def _word_center(word: Word) -> float | None:
    """估算一个词（chunk）的中心时间，用于判定它归属哪一段。

    时间戳两端齐全取中点；只有一端则用已知端（kotoba chunk 常见尾部 end=None、
    首部 start=None）；两端皆缺返回 None（交由调用方保守保留）。
    """
    ts = word.get("timestamp") or (None, None)
    start, end = ts
    if start is not None and end is not None:
        return (start + end) / 2
    if end is not None:
        return end
    if start is not None:
        return start
    return None


def _keep_core_words(
    words: list[Word], lo: float | None, hi: float | None
) -> list[Word]:
    """只保留中心时间落在 [lo, hi) 的词，丢弃 overlap 垫料区里本属于相邻段的词。

    分段转录时给每段取数窗口在逻辑区间两侧各 pad 若干秒（给模型留上下文，避免接缝
    处把词/乐句切碎），转录后用本函数按「词中心落在本段逻辑区间」去重：垫料区的词
    会落在相邻段的逻辑区间里、由相邻段保留，从而每个时间点恰好归属一段、接缝无缝。

    lo / hi 为 None 表示该侧不设限（首段左侧、末段右侧），避免丢掉音频极端处的词。
    无法定位中心（时间戳全缺）的词保守保留，避免丢内容。
    """
    kept: list[Word] = []
    for w in words:
        center = _word_center(w)
        if center is None:
            kept.append(w)
            continue
        if lo is not None and center < lo:
            continue
        if hi is not None and center >= hi:
            continue
        kept.append(w)
    return kept


class Transcriber:
    """Whisper 转录器。延迟加载模型，复用同一实例可转录多个文件。"""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def load(self) -> None:
        """检查/下载模型并加载 pipeline。"""
        _silence_warnings()
        if self.device is None:
            self.device = require_cuda()
        print(f"🤖 使用模型: {self.model_id}")

        print("\n📥 检查/下载模型...")
        try:
            snapshot_download(repo_id=self.model_id, local_files_only=False)
            print("✅ 模型准备完成")
        except Exception as e:
            print(f"⚠️  模型下载警告（尝试使用本地缓存）: {e}")

        print("\n🚀 加载模型...")
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 精度（GPU float16 / CPU float32）。transformers 5.x 用 dtype，旧名 torch_dtype 已弃用
            device=self.device,
            chunk_length_s=15,
            model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {},  # 传入注意力实现配置（GPU 时启用 sdpa）
            batch_size=8,  # 批处理大小=8，表示一次性处理 8 个音频文件/片段（提高吞吐量）
            stride_length_s=(5, 3),  # 处理长音频的参数，表示滑动窗口的步长策略
        )
        print("✅ 模型加载完成")

    def _transcribe_file(
        self, audio_file: str, language: str, beams: int
    ) -> list[Word]:
        """转录单个音频文件，返回（相对该文件的）词级时间戳。内部方法。

        注意时间戳粒度：kotoba-whisper 等蒸馏模型解码器只有 2 层，但其
        generation_config 的 alignment_heads 继承自原版 large-v3（引用到第 25 层），
        用 return_timestamps="word" 抽词级时间戳会在 cross_attentions[l] 越界
        （IndexError）。故这里用 return_timestamps=True 取 chunk（短语）级时间戳——
        其结构同为 {"text", "timestamp": (start, end)}，兼容 Word 契约与 segment_sentences。
        """
        if self._pipe is None:
            self.load()
        # noinspection PyTypeChecker
        result: dict = self._pipe(
            audio_file,
            return_timestamps=True,
            generate_kwargs={
                # 固定日语转录：kotoba-whisper 的 generation_config 是 is_multilingual=true 且
                # forced_decoder_ids 的语言槽为 null（不固定语言），不传则每段先做语种自动检测——
                # 对含唱歌/BGM/日英混杂的直播易误判语种、那一段质量骤降。官方 kotoba_whisper.py
                # 同样硬编码 language="ja"/task="transcribe"，这里对齐之，消除误判与相关弃用告警。
                "language": language,
                "task": "transcribe",
                "num_beams": beams,
                # 抑制相同 n-gram 的无限循环（幻觉）
                "no_repeat_ngram_size": 3,
            },
        )
        return result.get("chunks", [])

    def transcribe(
        self,
        audio_file: str,
        language: str = "japanese",
        beams: int = 3,
    ) -> list[Word]:
        """转录整个音频，一次性返回词级时间戳列表（不分段）。"""
        if self._pipe is None:
            self.load()

        print(f"\n🎵 正在转录: {audio_file}")
        print("⏳ 这可能需要一些时间...")
        words = self._transcribe_file(audio_file, language, beams)
        print("✅ 转录完成！")

        if not words:
            raise RuntimeError("未获取到单词级时间戳，无法生成字幕")
        return words

    def transcribe_segmented(
        self,
        audio_file: str,
        language: str = "japanese",
        beams: int = 3,
        target_chunk: float = 300.0,
        max_chunk: float = 600.0,
        noise_db: float = -45.0,
        min_silence: float = 0.5,
        overlap: float = 2.0,
        fallback_db: float = -35.0,
    ) -> Iterator[list[Word]]:
        """按静音切分长音频，逐段转录并产出（已对齐全局时间轴的）词列表。

        每段转完即 yield，调用方可据此显示进度、分批写盘。切点落在静音中点，
        不打断语句，故精度与整段转录一致。短音频（<= max_chunk）退化为一次整段转录。

        overlap: 取数窗口在每段逻辑区间两侧各外扩的秒数（垫料），给模型在接缝处留
        上下文、避免把词/乐句切碎（鹿乃长时间唱歌时硬切尤甚）。转录后按「词中心落在
        本段逻辑区间」过滤，垫料区的词归相邻段，接缝无缝去重。逻辑区间本身仍无缝相接。

        fallback_db: 二次「宽松」静音阈值（应 > noise_db，越接近 0 越宽松）。当严格阈值
        noise_db 在某段 [target,max] 区间探不到静音、本会在 max_chunk 处硬切时，改用这批
        宽松候选找次优切点，把硬切降为真正的最后兜底。设为 <= noise_db 则关闭二次探测。
        """
        if self._pipe is None:
            self.load()

        duration = probe_duration(audio_file)
        print(f"\n🎵 音频总时长: {duration:.1f}s（约 {duration / 60:.1f} 分钟）")

        if duration <= max_chunk:
            print("ℹ️  音频较短，整段转录。")
            words = self._transcribe_file(audio_file, language, beams)
            if words:
                yield words
            return

        print(f"🔍 探测静音区间（noise={noise_db}dB, d={min_silence}s）...")
        silences = detect_silences(audio_file, noise_db, min_silence)
        print(f"  找到 {len(silences)} 段静音")
        # 宽松阈值二次探测（仅当 fallback_db 比严格阈值更宽松时才跑），给硬切段兜次优切点
        fallback_silences: list[tuple[float, float]] | None = None
        if fallback_db > noise_db:
            print(f"🔍 二次宽松探测（noise={fallback_db}dB, d={min_silence}s）...")
            fallback_silences = detect_silences(audio_file, fallback_db, min_silence)
            print(f"  找到 {len(fallback_silences)} 段（宽松）")
        segments = plan_segments(
            duration, silences, target_chunk, max_chunk, fallback_silences
        )
        print(f"✂️  规划为 {len(segments)} 段，开始分段转录（重叠垫料 {overlap:.1f}s）\n")

        total = len(segments)
        with tempfile.TemporaryDirectory(prefix="kits_seg_") as tmp:
            tmp_dir = Path(tmp)
            for i, (start, end) in enumerate(segments, 1):
                # 取数窗口在逻辑区间两侧外扩 overlap（夹在 [0, duration] 内），转录后再裁回
                win_start = max(0.0, start - overlap)
                win_end = min(duration, end + overlap)
                print(
                    f"🎤 转录第 {i}/{total} 段 "
                    f"[{start:.1f}s -> {end:.1f}s, 时长 {end - start:.1f}s, "
                    f"取数 {win_start:.1f}~{win_end:.1f}s]..."
                )
                seg_path = tmp_dir / f"seg_{i:04d}.wav"
                slice_audio(audio_file, win_start, win_end, seg_path)
                words = self._transcribe_file(str(seg_path), language, beams)
                seg_path.unlink(missing_ok=True)
                if not words:
                    continue
                # 词级时间戳先加回窗口起始偏移，对齐全局时间轴
                words = _shift_words(words, win_start)
                # 再按「词中心落在本段逻辑区间 [start, end)」裁掉垫料区的词（首尾两端不设限）
                lo = None if i == 1 else start
                hi = None if i == total else end
                words = _keep_core_words(words, lo, hi)
                if words:
                    yield words
        print("\n✅ 全部分段转录完成！")
