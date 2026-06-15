"""separator 单元测试：只覆盖不触发重依赖的逻辑（延迟导入、入参校验）。

不安装 audio-separator 时，仅构造 VocalSeparator 不应失败（重依赖在 load() 时才导入）。
"""

from __future__ import annotations

import sys

import pytest

from kits.separator import DEFAULT_MODEL, VocalSeparator


class TestConstruction:
    def test_construct_does_not_import_audio_separator(self):
        # 构造实例不应触发 audio_separator 导入（延迟到 load()）
        sys.modules.pop("audio_separator", None)
        sep = VocalSeparator(output_dir="out")
        assert sep.output_dir == "out"
        assert sep.model_filename == DEFAULT_MODEL
        assert "audio_separator" not in sys.modules

    def test_custom_params(self):
        sep = VocalSeparator(
            output_dir="d", model_filename="x.onnx", output_format="WAV"
        )
        assert sep.model_filename == "x.onnx"
        assert sep.output_format == "WAV"


class TestSeparateValidation:
    def test_missing_file_raises_before_model_load(self, monkeypatch):
        # 输入文件不存在时应抛 FileNotFoundError，而不是去加载模型
        sep = VocalSeparator(output_dir="out")

        def _boom():
            raise AssertionError("不应在文件校验前加载模型")

        monkeypatch.setattr(sep, "load", _boom)
        sep._sep = object()  # 假装已加载，跳过 load
        with pytest.raises(FileNotFoundError):
            sep.separate("绝对不存在的音频.mp3")
