"""llm 公共客户端单元测试：只覆盖不触网的纯逻辑（base_url 规整、Key 回退、空 Key 策略、别名）。"""

from __future__ import annotations

import pytest

from kits.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMClient,
    LLMError,
)


def _clear_env(monkeypatch):
    for name in (
        "KITS_LLM_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "KITS_LLM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


class TestApiKeyValidation:
    def test_default_endpoint_raises_without_key(self, monkeypatch):
        _clear_env(monkeypatch)
        with pytest.raises(LLMError):
            LLMClient()

    def test_explicit_key_overrides_env(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
        assert LLMClient(api_key="arg-key").api_key == "arg-key"


class TestApiKeyFallbackOrder:
    def test_reads_deepseek_key(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        assert LLMClient().api_key == "ds-key"

    def test_openai_key_over_deepseek(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        assert LLMClient().api_key == "oa-key"

    def test_kits_key_over_all(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        monkeypatch.setenv("KITS_LLM_API_KEY", "kits-key")
        assert LLMClient().api_key == "kits-key"


class TestBaseUrlResolution:
    def test_default_endpoint(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        assert LLMClient().endpoint == f"{DEFAULT_BASE_URL}/chat/completions"

    def test_arg_base_appends_path(self, monkeypatch):
        _clear_env(monkeypatch)
        c = LLMClient(api_key="k", base_url="http://localhost:11434/v1")
        assert c.endpoint == "http://localhost:11434/v1/chat/completions"

    def test_trailing_slash_trimmed(self, monkeypatch):
        _clear_env(monkeypatch)
        c = LLMClient(api_key="k", base_url="http://localhost:11434/v1/")
        assert c.endpoint == "http://localhost:11434/v1/chat/completions"

    def test_full_endpoint_kept_as_is(self, monkeypatch):
        _clear_env(monkeypatch)
        c = LLMClient(api_key="k", base_url="http://host/custom/chat/completions")
        assert c.endpoint == "http://host/custom/chat/completions"

    def test_env_base_url(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("KITS_LLM_BASE_URL", "http://localhost:8000/v1")
        c = LLMClient(api_key="k")
        assert c.endpoint == "http://localhost:8000/v1/chat/completions"

    def test_arg_base_url_over_env(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("KITS_LLM_BASE_URL", "http://env/v1")
        c = LLMClient(api_key="k", base_url="http://arg/v1")
        assert c.endpoint == "http://arg/v1/chat/completions"


class TestEmptyKeyPolicy:
    def test_custom_base_allows_empty_key(self, monkeypatch):
        _clear_env(monkeypatch)
        # 本地端点（非默认 base_url）允许空 Key，不抛错
        c = LLMClient(base_url="http://localhost:11434/v1")
        assert c.api_key == ""

    def test_default_base_still_requires_key(self, monkeypatch):
        _clear_env(monkeypatch)
        with pytest.raises(LLMError):
            LLMClient(base_url=DEFAULT_BASE_URL)


class TestDefaults:
    def test_default_model_and_timeout(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
        c = LLMClient()
        assert c.model == DEFAULT_MODEL
        assert c.timeout == 120.0


class TestBackwardCompatAliases:
    def test_deepseek_aliases_are_same_objects(self):
        from kits import deepseek

        assert deepseek.DeepSeekClient is LLMClient
        assert deepseek.DeepSeekError is LLMError
