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


def _concat_audio(parts: list[Path], out_path: Path) -> None:
    """用 ffmpeg concat demuxer 把多段同格式人声音频按序无缝拼接成一个文件。"""
    list_file = out_path.parent / "_concat_list.txt"
    # concat demuxer 要求每行 file '绝对路径'，单引号内的单引号需转义
    lines = [f"file '{p.resolve().as_posix()}'" for p in parts]
    list_file.write_text("\n".join(lines), encoding="utf-8")
    try:
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(out_path), "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SeparationError(f"合并人声分段失败: {result.stderr[:300]}")
    finally:
        list_file.unlink(missing_ok=True)


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
        output_dir: str = "downloads",
        model_filename: str = DEFAULT_MODEL,
        output_format: str = "MP3",
        model_file_dir: str | None = None,
        segment_size: int = 512,
        overlap: float = 0.1,
        segment_minutes: float = 15.0,
    ):
        self.output_dir = output_dir
        self.model_filename = model_filename
        self.output_format = output_format
        self.model_file_dir = model_file_dir
        self.segment_size = segment_size
        self.overlap = overlap
        # 超过该时长的音频按此长度切段、逐段分离再合并，避免一次性出整轨爆内存。
        # <=0 表示禁用分段、始终整段处理。
        self.segment_minutes = segment_minutes
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
        kwargs: dict = {
            "output_dir": self.output_dir,
            "output_format": self.output_format,
            "output_single_stem": "Vocals",
        }
        if self.model_file_dir is not None:
            kwargs["model_file_dir"] = self.model_file_dir

        # segment_size 越大 / overlap 越小，分块数越少、迭代越快（GPU 也更吃满）。
        # 对“压 BGM 的人声预处理”而言不需要高 overlap，默认偏快档。
        # 注意：MDX(.onnx) 的 overlap 是 0~1 小数；MDXC(roformer .ckpt) 的 overlap
        # 是整数步数，语义不同，故两边分开传，避免给 roformer 塞错类型。
        self._sep = Separator(
            output_bitrate="128k",
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

    def separate(self, audio_file: str) -> str:
        """分离出人声轨，返回人声音频文件路径。

        长音频（超过 segment_minutes）按固定时长切段、逐段分离再用 ffmpeg 合并，
        避免 audio-separator 一次性把整轨结果拉进内存导致 MemoryError。短音频直接
        整段处理。
        """
        in_path = Path(audio_file)
        if not in_path.is_file():
            raise FileNotFoundError(f"找不到输入音频文件: {in_path}")

        seg_seconds = self.segment_minutes * 60
        duration = _probe_duration(str(in_path)) if seg_seconds > 0 else 0.0

        if seg_seconds <= 0 or duration <= seg_seconds:
            # 短音频或禁用分段：整段处理
            return self._separate_one(in_path)

        return self._separate_segmented(in_path, duration, seg_seconds)

    def _separate_segmented(
        self, in_path: Path, duration: float, seg_seconds: float
    ) -> str:
        """按固定时长切段、逐段分离、ffmpeg 合并。

        分段切片放临时目录；人声产物受 audio-separator 固化的 output_dir 限制只能落
        在 self.output_dir，故合并完成后再清理这些中间人声文件，避免污染输出目录。
        """
        num_segments = math.ceil(duration / seg_seconds)
        print(
            f"\n🎬 音频较长（{duration / 60:.1f} 分钟），按 {self.segment_minutes:g} "
            f"分钟切成 {num_segments} 段逐段分离，避免内存溢出"
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="kits_sep_", dir=self.output_dir))
        vocal_parts: list[Path] = []
        try:
            for idx in range(num_segments):
                start = idx * seg_seconds
                dur = min(seg_seconds, duration - start)
                print(f"\n── 第 {idx + 1}/{num_segments} 段（{start / 60:.1f}~"
                      f"{(start + dur) / 60:.1f} 分钟）──")

                seg_in = tmp_dir / f"seg_{idx:03d}.wav"
                _slice_audio(str(in_path), start, dur, seg_in)
                vocal = self._separate_one(seg_in)
                vocal_parts.append(Path(vocal))
                seg_in.unlink(missing_ok=True)  # 原始分段切片用完即删，省空间

            out_name = f"{in_path.stem}_(Vocals).{self.output_format.lower()}"
            final_path = Path(self.output_dir) / out_name
            print(f"\n🔗 合并 {len(vocal_parts)} 段人声 → {final_path.name}")
            _concat_audio(vocal_parts, final_path)
            print(f"✅ 人声已保存: {final_path}")
            return str(final_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            for p in vocal_parts:  # 清理落在 output_dir 的中间人声分段
                p.unlink(missing_ok=True)

    def _separate_one(self, in_path: Path) -> str:
        """对单个（已确保不超长的）音频文件做一次分离，返回人声轨路径。

        audio-separator 的输出目录在 load() 时已固化为 self.output_dir，运行时改
        属性无效，故产物统一落在 self.output_dir，返回值据此拼绝对路径。
        """
        if self._sep is None:
            self.load()

        print(f"🎤 正在分离人声: {in_path.name}")
        outputs = self._sep.separate(str(in_path))
        if not outputs:
            raise SeparationError(f"人声分离未产出任何文件: {in_path.name}")

        # 只取人声轨（output_single_stem='Vocals' 时通常只有一个输出）
        vocal = next((p for p in outputs if "vocal" in str(p).lower()), outputs[0])
        vocal_path = Path(vocal)
        # audio-separator 通常只返回裸文件名，落在 output_dir 下
        if not vocal_path.is_absolute() and not vocal_path.exists():
            vocal_path = Path(self.output_dir) / vocal_path.name
        return str(vocal_path)
