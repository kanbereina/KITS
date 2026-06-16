"""summarizer 单元测试：只覆盖不触网的纯逻辑（预设加载 / 解析 / 格式化 / 分块）。"""

from __future__ import annotations

import json

import pytest

from kits.summarizer import (
    SummarizeError,
    available_presets,
    chunk_sentences,
    format_sentences,
    load_presets,
    resolve_preset,
)


class TestLoadPresets:
    def test_builtin_has_expected_presets(self):
        names = available_presets()
        assert {"timeline", "summary", "highlights", "setlist"} <= set(names)

    def test_builtin_default_is_timeline(self):
        assert load_presets()["default"] == "timeline"

    def test_prompt_file_overrides_and_merges(self, tmp_path):
        custom = tmp_path / "p.json"
        custom.write_text(
            json.dumps(
                {"default": "mine", "presets": {"mine": {"system": "自定义提示词"}}}
            ),
            encoding="utf-8",
        )
        cfg = load_presets(str(custom))
        # 自定义预设并入，且内置预设仍在（浅合并）
        assert "mine" in cfg["presets"]
        assert "timeline" in cfg["presets"]
        assert cfg["default"] == "mine"

    def test_missing_prompt_file_raises(self):
        with pytest.raises(SummarizeError):
            load_presets("不存在的文件.json")

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json", encoding="utf-8")
        with pytest.raises(SummarizeError):
            load_presets(str(bad))


class TestResolvePreset:
    def test_none_uses_default(self):
        name, system = resolve_preset(None)
        assert name == "timeline"
        assert system

    def test_explicit_preset(self):
        name, system = resolve_preset("setlist")
        assert name == "setlist"
        assert "歌单" in system

    def test_unknown_preset_raises(self):
        with pytest.raises(SummarizeError):
            resolve_preset("nope")


class TestFormatSentences:
    def test_renders_hms_timestamps(self):
        sents = [
            {"start": 0.0, "end": 2.0, "text": "こんにちは"},
            {"start": 65.5, "end": 70.0, "text": "歌います"},
        ]
        out = format_sentences(sents)
        assert out == "[00:00:00-00:00:02] こんにちは\n[00:01:05-00:01:10] 歌います"


class TestChunkSentences:
    def test_empty_returns_empty(self):
        assert chunk_sentences([]) == []

    def test_single_chunk_when_under_budget(self):
        sents = [{"start": 0.0, "end": 1.0, "text": "あ"}]
        assert len(chunk_sentences(sents, max_chars=8000)) == 1

    def test_splits_when_over_budget(self):
        sents = [{"start": float(i), "end": i + 1.0, "text": "テスト"} for i in range(10)]
        chunks = chunk_sentences(sents, max_chars=40)
        # 每条渲染约 24 字符，max_chars=40 应切成多块且不丢条
        assert len(chunks) > 1
        assert sum(len(c) for c in chunks) == 10

    def test_oversized_single_sentence_is_own_chunk(self):
        sents = [{"start": 0.0, "end": 1.0, "text": "長" * 100}]
        chunks = chunk_sentences(sents, max_chars=10)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1
