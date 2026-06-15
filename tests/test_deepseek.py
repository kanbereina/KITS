"""deepseek 公共客户端单元测试：只覆盖不触网的纯逻辑（Key 校验、默认值）。"""

from __future__ import annotations

import pytest

from kits.deepseek import DEFAULT_MODEL, DeepSeekClient, DeepSeekError


class TestApiKeyValidation:
    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(DeepSeekError):
            DeepSeekClient()

    def test_reads_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        assert DeepSeekClient().api_key == "env-key"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        assert DeepSeekClient(api_key="arg-key").api_key == "arg-key"


class TestDefaults:
    def test_default_model_and_timeout(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        c = DeepSeekClient()
        assert c.model == DEFAULT_MODEL
        assert c.timeout == 120.0

    def test_custom_model_and_timeout(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        c = DeepSeekClient(model="deepseek-reasoner", timeout=30.0)
        assert c.model == "deepseek-reasoner"
        assert c.timeout == 30.0
