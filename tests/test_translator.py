"""translator 模块单元测试：只覆盖不触网的纯逻辑（提示拼装 / 响应解析 / Key 校验）。"""

from __future__ import annotations

import pytest

from kits.translator import DeepSeekTranslator, TranslationError


class TestApiKeyValidation:
    def test_raises_without_key(self, monkeypatch):
        # 清掉环境变量且不传 key 时应报错
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(TranslationError):
            DeepSeekTranslator()

    def test_reads_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        assert DeepSeekTranslator().api_key == "env-key"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        assert DeepSeekTranslator(api_key="arg-key").api_key == "arg-key"


class TestBuildUserContent:
    def test_numbers_each_line(self):
        batch = [
            {"start": 0.0, "end": 1.0, "text": "こんにちは"},
            {"start": 1.0, "end": 2.0, "text": "ありがとう"},
        ]
        content = DeepSeekTranslator._build_user_content(batch)
        assert content == "0|||こんにちは\n1|||ありがとう"


class TestParseResponse:
    def test_parses_in_order(self):
        content = "0|||你好\n1|||谢谢"
        assert DeepSeekTranslator._parse_response(content, 2) == ["你好", "谢谢"]

    def test_reorders_by_index(self):
        # 即使响应行乱序，也按序号回填
        content = "1|||谢谢\n0|||你好"
        assert DeepSeekTranslator._parse_response(content, 2) == ["你好", "谢谢"]

    def test_missing_line_becomes_empty(self):
        content = "0|||你好"
        assert DeepSeekTranslator._parse_response(content, 2) == ["你好", ""]

    def test_ignores_garbage_lines(self):
        content = "这是说明\n0|||你好\n```\n1|||谢谢"
        assert DeepSeekTranslator._parse_response(content, 2) == ["你好", "谢谢"]

    def test_ignores_out_of_range_index(self):
        content = "0|||你好\n5|||越界"
        assert DeepSeekTranslator._parse_response(content, 2) == ["你好", ""]

    def test_ignores_non_integer_index(self):
        content = "x|||坏序号\n0|||你好"
        assert DeepSeekTranslator._parse_response(content, 1) == ["你好"]
