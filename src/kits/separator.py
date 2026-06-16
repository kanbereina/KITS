"""人声分离：封装 audio-separator（UVR/MDX 模型），从音频中分离出人声。

依赖 audio-separator（内含 torch/onnxruntime），故在 CLI 中走延迟导入，避免无谓
加载重依赖栈。默认只产出人声（Vocals）轨，供后续转录字幕时降低 BGM / 唱歌的干扰。

用法：
    sep = VocalSeparator(output_dir="downloads")
    vocals_path = sep.separate("live.mp3")        # 返回人声音频路径
"""

from __future__ import annotations

import os
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
    ):
        self.output_dir = output_dir
        self.model_filename = model_filename
        self.output_format = output_format
        self.model_file_dir = model_file_dir
        self.segment_size = segment_size
        self.overlap = overlap
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

        audio-separator 可能返回相对 output_dir 的文件名，这里统一拼成可用路径。
        """
        if self._sep is None:
            self.load()

        in_path = Path(audio_file)
        if not in_path.is_file():
            raise FileNotFoundError(f"找不到输入音频文件: {in_path}")

        print(f"\n🎤 正在分离人声: {in_path.name}")
        print("⏳ 这可能需要一些时间...")
        outputs = self._sep.separate(str(in_path))
        if not outputs:
            raise SeparationError("人声分离未产出任何文件")

        # 只取人声轨（output_single_stem='Vocals' 时通常只有一个输出）
        vocal = next((p for p in outputs if "vocal" in str(p).lower()), outputs[0])
        vocal_path = Path(vocal)
        # audio-separator 返回的可能是相对 output_dir 的文件名
        if not vocal_path.is_absolute() and not vocal_path.exists():
            vocal_path = Path(self.output_dir) / vocal_path.name

        print(f"✅ 人声已保存: {vocal_path}")
        return str(vocal_path)
