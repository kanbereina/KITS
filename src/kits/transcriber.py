"""音频转录：封装 Whisper large-v3-turbo 的加载与转录。

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
    audio_file: str, noise_db: float = -30.0, min_silence: float = 0.5
) -> list[tuple[float, float]]:
    """用 ffmpeg silencedetect 探测静音区间，返回 [(start, end), ...]（秒）。

    noise_db: 低于该响度视为静音（越接近 0 越严格，越负越宽松）。
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
) -> list[tuple[float, float]]:
    """根据静音区间规划分段切点，返回 [(start, end), ...] 覆盖 [0, duration]。

    策略：从当前段起点出发，在 [起点+target_chunk 之后的第一个静音中点] 处切；
    若到 起点+max_chunk 仍无静音可切（如长时间唱歌），则在 max_chunk 处强切兜底。
    切点优先落在静音中点，保证不打断语句。
    """
    if duration <= max_chunk:
        return [(0.0, duration)]

    # 静音中点列表（升序），作为候选切点
    midpoints = sorted((s + e) / 2 for s, e in silences)

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
            # 软下限到硬上限之间没有静音可切：强切在硬上限
            cut = hard_limit
        end = min(cut, duration)
        segments.append((start, end))
        start = end
    return segments


def slice_audio(audio_file: str, start: float, end: float, out_path: Path) -> None:
    """用 ffmpeg 切出 [start, end] 区间到 out_path（重编码为 16k 单声道 wav）。"""
    cmd = [
        "ffmpeg", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", audio_file,
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
            device=self.device,
            chunk_length_s=30,
            stride_length_s=(5, 3),
        )
        print("✅ 模型加载完成")

    def _transcribe_file(
        self, audio_file: str, language: str, beams: int
    ) -> list[Word]:
        """转录单个音频文件，返回（相对该文件的）词级时间戳。内部方法。"""
        if self._pipe is None:
            self.load()
        # noinspection PyTypeChecker
        result: dict = self._pipe(
            audio_file,
            return_timestamps="word",
            generate_kwargs={
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
        beams: int = 1,
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
        beams: int = 1,
        target_chunk: float = 300.0,
        max_chunk: float = 600.0,
        noise_db: float = -30.0,
        min_silence: float = 0.5,
    ) -> Iterator[list[Word]]:
        """按静音切分长音频，逐段转录并产出（已对齐全局时间轴的）词列表。

        每段转完即 yield，调用方可据此显示进度、分批写盘。切点落在静音中点，
        不打断语句，故精度与整段转录一致。短音频（<= max_chunk）退化为一次整段转录。
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
        segments = plan_segments(duration, silences, target_chunk, max_chunk)
        print(f"✂️  规划为 {len(segments)} 段，开始分段转录\n")

        total = len(segments)
        with tempfile.TemporaryDirectory(prefix="kits_seg_") as tmp:
            tmp_dir = Path(tmp)
            for i, (start, end) in enumerate(segments, 1):
                print(
                    f"🎤 转录第 {i}/{total} 段 "
                    f"[{start:.1f}s -> {end:.1f}s, 时长 {end - start:.1f}s]..."
                )
                seg_path = tmp_dir / f"seg_{i:04d}.wav"
                slice_audio(audio_file, start, end, seg_path)
                words = self._transcribe_file(str(seg_path), language, beams)
                seg_path.unlink(missing_ok=True)
                if words:
                    yield _shift_words(words, start)
        print("\n✅ 全部分段转录完成！")
