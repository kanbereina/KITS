"""音频转录：封装 Whisper large-v3-turbo 的加载与转录。

产出单词级时间戳列表（list[Word]），交给 kits.subtitle 断句生成 SRT。
模型转录文本同样可直接喂给后续的 DeepSeek 总结模块。
"""

from __future__ import annotations

import warnings

import torch
from huggingface_hub import snapshot_download
from transformers import pipeline

from kits.subtitle import Word

MODEL_ID = "openai/whisper-large-v3-turbo"


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

    def transcribe(
        self,
        audio_file: str,
        language: str = "japanese",
        beams: int = 1,
    ) -> list[Word]:
        """转录音频，返回单词级时间戳列表。"""
        if self._pipe is None:
            self.load()

        print(f"\n🎵 正在转录: {audio_file}")
        print("⏳ 这可能需要一些时间...")
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
        print("✅ 转录完成！")

        words: list[Word] = result.get("chunks", [])
        if not words:
            raise RuntimeError("未获取到单词级时间戳，无法生成字幕")
        return words
