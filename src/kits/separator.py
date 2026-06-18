"""人声分离：封装 audio-separator（UVR/MDX 模型），从音频中分离出人声。

依赖 audio-separator（内含 torch/onnxruntime），故在 CLI 中走延迟导入，避免无谓
加载重依赖栈。默认只产出人声（Vocals）轨，供后续转录字幕时降低 BGM / 唱歌的干扰。

用法：
    sep = VocalSeparator(output_dir="downloads")
    vocals_path = sep.separate("live.mp3")        # 返回人声音频路径
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

__all__ = ["DEFAULT_MODEL", "SeparationError", "VocalSeparator"]


def _expose_torch_cuda_dlls() -> None:
    """把 torch 自带的 CUDA 12 / cuDNN 9 dll 目录加入 DLL 搜索路径。

    onnxruntime-gpu 需要 cublasLt64_12.dll / cudnn64_9.dll 等运行时库才能启用
    CUDAExecutionProvider，否则静默回落到 CPU（人声分离慢数倍）。本机没装独立
    CUDA Toolkit，但 torch(cu128) 在 torch/lib 下自带这些 dll，这里直接复用，
    免去额外安装。必须在 import onnxruntime 之前调用。仅 Windows 需要。
    """
    if os.name != "nt":
        return
    try:
        import torch
    except ImportError:
        return
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.is_dir():
        os.add_dll_directory(str(torch_lib))


def _probe_duration(audio_file: str) -> float:
    """用 ffprobe 取音频总时长（秒）。失败返回 0（按短音频整段处理兜底）。

    与 transcriber.probe_duration 同款逻辑，但本模块不依赖 torch 栈，故不复用
    那边（transcriber 顶层会拉起 transformers/torch，违反分层约定）。
    """
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _probe_bitrate(audio_file: str) -> int:
    """用 ffprobe 取原音频比特率（bps）。失败返回 0。

    优先取音频流比特率（精确，如 128000），它不含容器开销；取不到再回退到
    format 整体比特率（MP3 等会比标称略高，如 129192，靠取整时的容差吸收）。
    """
    for args in (
        ["-select_streams", "a:0", "-show_entries", "stream=bit_rate"],
        ["-show_entries", "format=bit_rate"],
    ):
        cmd = ["ffprobe", "-v", "error", *args,
               "-of", "default=noprint_wrappers=1:nokey=1", audio_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        try:
            value = int(result.stdout.strip())
            if value > 0:
                return value
        except ValueError:
            continue
    return 0


def _slice_audio(audio_file: str, start: float, dur: float, out_path: Path) -> None:
    """切出 [start, start+dur] 区间到 out_path。

    保留原始采样率/声道（人声分离需要全频带立体声，不能像转录那样降采样到
    16k 单声道，否则分离音质下降）。重编码为 WAV 避免 MP3 帧边界对不齐。
    """
    cmd = [
        "ffmpeg", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", audio_file, "-vn", "-c:a", "pcm_s16le",
        str(out_path), "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SeparationError(
            f"切分音频失败 [{start:.1f}s +{dur:.1f}s]: {result.stderr[:300]}"
        )


def _encode_final(parts: list[Path], out_path: Path, bitrate: str | None) -> None:
    """把一个或多个无损 WAV 分段编码为最终输出文件（格式由 out_path 扩展名决定）。

    单段直接转码；多段用 concat demuxer 先拼接再转码。中间分段始终是无损 WAV，
    只在这里做一次有损编码，避免分段方案重复压缩叠加损质。bitrate 仅对有损格式
    有效（None 表示无损或交给 ffmpeg 默认）。
    """
    if len(parts) == 1:
        cmd = ["ffmpeg", "-i", str(parts[0]), "-vn"]
        list_file = None
    else:
        list_file = out_path.parent / "_concat_list.txt"
        # concat demuxer 要求每行 file '绝对路径'
        lines = [f"file '{p.resolve().as_posix()}'" for p in parts]
        list_file.write_text("\n".join(lines), encoding="utf-8")
        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(list_file), "-vn"]
    if bitrate:
        cmd += ["-b:a", bitrate]
    cmd += [str(out_path), "-y"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SeparationError(f"编码最终人声文件失败: {result.stderr[:300]}")
    finally:
        if list_file is not None:
            list_file.unlink(missing_ok=True)


# 无损音频格式：输出这些格式时不套用比特率（由位深决定，无比特率概念）。
_LOSSLESS_FORMATS = {"wav", "flac", "aiff", "aif", "alac"}


def _round_up_pow2_kbps(bitrate_bps: int) -> int:
    """把原始比特率(bps)向上取整到最小的 2 的幂 kbps，夹到 [32, 320]。

    例：63kbps→64，118kbps→128，128kbps→128。人声分离后内容比原混音简单，输出
    不超过原始比特率即可保真，故取 >= 原值的最近 2 的幂；上限 320（MP3 常见最高
    档），下限 32。留 5% 容差吸收 MP3 标称偏差（如 129k 仍归 128 而非跳到 256）。
    """
    kbps = bitrate_bps / 1000 * 0.95
    p = 32
    while p < kbps and p < 320:
        p *= 2
    return min(p, 320)


# audio-separator 模型（当前使用的是 MDXC，轻量模型，人声 SDR 10.2，伴奏 15.5。VIP 模型，综合表现优秀，速度与干净度兼顾）。
# 可用 model_filename 覆盖。
DEFAULT_MODEL = "UVR-MDX-NET_Main_427.onnx"


class SeparationError(RuntimeError):
    """人声分离过程中的错误（模型加载失败、未产出人声轨等）。"""


class VocalSeparator:
    """audio-separator 封装。延迟加载模型，复用同一实例可分离多个文件。

    默认 output_single_stem='Vocals'，只输出人声轨，省去无用的伴奏文件。
    """

    def __init__(
        self,
        output_dir: str = "output",
        model_filename: str = DEFAULT_MODEL,
        output_format: str = "MP3",
        model_file_dir: str | None = None,
        segment_size: int = 512,
        overlap: float = 0.1,
        segment_minutes: float = 15.0,
        output_bitrate: str | None = None,
        cache_dir: str = ".cache",
    ):
        self.output_dir = output_dir
        # 所有中间产物（切片、底层分离出的裸 WAV）统一落在此目录下的临时工作目录，
        # 与最终输出位置解耦，避免裸文件漏进当前目录 / 输出目录。默认 cwd 下的 .cache。
        self.cache_dir = cache_dir
        self.model_filename = model_filename
        self.output_format = output_format
        self.model_file_dir = model_file_dir
        self.segment_size = segment_size
        self.overlap = overlap
        # 超过该时长的音频按此长度切段、逐段分离再合并，避免一次性出整轨爆内存。
        # <=0 表示禁用分段、始终整段处理。
        self.segment_minutes = segment_minutes
        # 最终输出比特率（如 "128k"）。None=自动对齐原音频（向上取整到 2 的幂 kbps）。
        # 仅对有损输出格式（MP3/AAC 等）生效；无损格式忽略。
        self.output_bitrate = output_bitrate
        self._sep = None

    def load(self) -> None:
        """构造底层 Separator 并加载模型。延迟导入 audio-separator。"""
        # audio-separator 会 import onnxruntime；先暴露 torch 自带的 CUDA dll，
        # 否则 onnxruntime-gpu 找不到 cublasLt/cudnn，CUDAExecutionProvider 失效。
        _expose_torch_cuda_dlls()
        try:
            from audio_separator.separator import Separator
        except ImportError as e:
            raise SeparationError(
                "未安装 audio-separator，请先安装：uv add 'audio-separator[gpu]'"
            ) from e

        print(f"\n🎚️  加载人声分离模型: {self.model_filename}")
        # 底层统一输出无损 WAV：作为中间产物，最终格式/比特率由 _encode_final 一次性
        # 套用，避免分段方案里反复有损压缩叠加损质，也绕开 output_bitrate 在 load()
        # 固化、运行时改不动的限制。
        # output_dir 此处只是占位：真正写盘前会在每次分离时重定向到临时工作目录
        # （见 _redirect_output），故底层产物不会落到 self.output_dir / 当前目录。
        kwargs: dict = {
            "output_dir": str(Path(self.cache_dir)),
            "output_format": "WAV",
            "output_single_stem": "Vocals",
        }
        if self.model_file_dir is not None:
            kwargs["model_file_dir"] = self.model_file_dir

        # segment_size 越大 / overlap 越小，分块数越少、迭代越快（GPU 也更吃满）。
        # 对“压 BGM 的人声预处理”而言不需要高 overlap，默认偏快档。
        # 注意：MDX(.onnx) 的 overlap 是 0~1 小数；MDXC(roformer .ckpt) 的 overlap
        # 是整数步数，语义不同，故两边分开传，避免给 roformer 塞错类型。
        self._sep = Separator(
            mdx_params={
                "hop_length": 1024,
                "enable_denoise": False,
                "batch_size": 1,
                "segment_size": self.segment_size,
                "overlap": self.overlap,
            },
            mdxc_params={
                "override_model_segment_size": True,
                "pitch_shift": 0,
                "batch_size": 1,
                "segment_size": self.segment_size,
                "overlap": 8,
            },
            **kwargs,
        )
        self._sep.load_model(model_filename=self.model_filename)
        print("✅ 模型加载完成")

    def _resolve_bitrate(self, in_path: Path, ext: str) -> str | None:
        """决定最终输出比特率。无损格式（按最终扩展名判断）返回 None；有损格式：
        用户指定优先，否则探测原音频比特率并向上取整到 2 的幂 kbps。探测失败则
        返回 None（交给 ffmpeg 默认）。
        """
        if ext.lower() in _LOSSLESS_FORMATS:
            return None
        if self.output_bitrate is not None:
            return self.output_bitrate
        src_bps = _probe_bitrate(str(in_path))
        if src_bps <= 0:
            return None
        kbps = _round_up_pow2_kbps(src_bps)
        print(f"🎚️  原音频比特率 ~{src_bps // 1000}k，输出对齐为 {kbps}k")
        return f"{kbps}k"

    def separate(self, audio_file: str, output_path: str | None = None) -> str:
        """分离出人声轨，返回人声音频文件路径。

        output_path 指定时直接用作最终输出文件（输出格式按其扩展名），否则在
        output_dir 下派生 `{输入名}_(Vocals).{output_format}`。

        长音频（超过 segment_minutes）按固定时长切段、逐段分离再用 ffmpeg 合并，
        避免 audio-separator 一次性把整轨结果拉进内存导致 MemoryError。短音频直接
        整段处理。底层统一产出无损 WAV，最终再编码为目标格式/比特率。
        """
        in_path = Path(audio_file)
        if not in_path.is_file():
            raise FileNotFoundError(f"找不到输入音频文件: {in_path}")

        if output_path is not None:
            final_path = Path(output_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            ext = final_path.suffix.lstrip(".").lower() or self.output_format.lower()
        else:
            ext = self.output_format.lower()
            final_path = Path(self.output_dir) / f"{in_path.stem}_(Vocals).{ext}"

        bitrate = self._resolve_bitrate(in_path, ext)

        seg_seconds = self.segment_minutes * 60
        duration = _probe_duration(str(in_path)) if seg_seconds > 0 else 0.0

        # 切片与底层裸 WAV 统一落在 cache_dir 下的临时工作目录，与最终输出位置解耦，
        # 不再漏进当前目录 / output_dir。整个目录用完即删，无需逐个清中间 WAV。
        cache_root = Path(self.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="kits_sep_", dir=cache_root))
        try:
            if seg_seconds <= 0 or duration <= seg_seconds:
                # 短音频或禁用分段：整段分离出一份 WAV
                wav_parts = [Path(self._separate_one(in_path, tmp_dir))]
            else:
                wav_parts = self._separate_segments(in_path, duration, seg_seconds, tmp_dir)

            print(f"\n🔗 输出人声 → {final_path.name}")
            _encode_final(wav_parts, final_path, bitrate)
            print(f"✅ 人声已保存: {final_path}")
            return str(final_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _separate_segments(
        self, in_path: Path, duration: float, seg_seconds: float, tmp_dir: Path
    ) -> list[Path]:
        """按固定时长切段、逐段分离，返回各段人声 WAV 路径列表（供合并）。"""
        num_segments = math.ceil(duration / seg_seconds)
        print(
            f"\n🎬 音频较长（{duration / 60:.1f} 分钟），按 {self.segment_minutes:g} "
            f"分钟切成 {num_segments} 段逐段分离，避免内存溢出"
        )
        vocal_parts: list[Path] = []
        for idx in range(num_segments):
            start = idx * seg_seconds
            dur = min(seg_seconds, duration - start)
            print(f"\n── 第 {idx + 1}/{num_segments} 段（{start / 60:.1f}~"
                  f"{(start + dur) / 60:.1f} 分钟）──")

            seg_in = tmp_dir / f"seg_{idx:03d}.wav"
            _slice_audio(str(in_path), start, dur, seg_in)
            vocal = self._separate_one(seg_in, tmp_dir)
            vocal_parts.append(Path(vocal))
            seg_in.unlink(missing_ok=True)  # 原始分段切片用完即删，省空间
        return vocal_parts

    def _separate_one(self, in_path: Path, work_dir: Path) -> str:
        """对单个（已确保不超长的）音频文件做一次分离，返回人声 WAV 路径。

        分离前把底层 Separator（及其 model_instance）的 output_dir 重定向到 work_dir，
        故裸 WAV 落在临时工作目录、不污染当前目录或 self.output_dir。
        """
        if self._sep is None:
            self.load()

        work_dir.mkdir(parents=True, exist_ok=True)
        # audio-separator 在 separate() 时才读 output_dir（见 common_separator 写盘逻辑），
        # 故运行时重定向有效；model_instance 已加载时需一并同步。
        work_str = str(work_dir)
        self._sep.output_dir = work_str
        if getattr(self._sep, "model_instance", None) is not None:
            self._sep.model_instance.output_dir = work_str

        print(f"🎤 正在分离人声: {in_path.name}")
        outputs = self._sep.separate(str(in_path))
        if not outputs:
            raise SeparationError(f"人声分离未产出任何文件: {in_path.name}")

        # 只取人声轨（output_single_stem='Vocals' 时通常只有一个输出）
        vocal = next((p for p in outputs if "vocal" in str(p).lower()), outputs[0])
        vocal_path = Path(vocal)
        # audio-separator 通常只返回裸文件名，落在重定向后的 work_dir 下
        if not vocal_path.is_absolute() and not vocal_path.exists():
            vocal_path = work_dir / vocal_path.name
        return str(vocal_path)
