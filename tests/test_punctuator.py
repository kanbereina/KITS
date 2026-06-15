"""punctuator 单元测试：覆盖不触模型的纯逻辑（标点判定、restore 的回填/回退/时间戳保留）。

用假模型替身注入 _model，避免下载真实 ONNX 模型。
"""

from __future__ import annotations

from kits.punctuator import Punctuator


class _FakeModel:
    """假标点模型：把每条文本按出现的关键词补上句号，返回 list[list[str]]（与真模型同形）。"""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.calls: list[list[str]] = []

    def infer(self, batch: list[str]) -> list[list[str]]:
        self.calls.append(list(batch))
        return [list(self.mapping.get(t, t)) for t in batch]


class TestNeedsPunctuation:
    def test_empty_or_blank_skipped(self):
        assert Punctuator._needs_punctuation("") is False
        assert Punctuator._needs_punctuation("   ") is False

    def test_already_punctuated_skipped(self):
        assert Punctuator._needs_punctuation("こんにちは。") is False
        assert Punctuator._needs_punctuation("元気？") is False
        assert Punctuator._needs_punctuation("えっ、そう") is False

    def test_plain_text_needs(self):
        assert Punctuator._needs_punctuation("こんにちは") is True


class TestRestore:
    def test_empty_returns_empty(self):
        p = Punctuator()
        assert p.restore([]) == []

    def test_punctuates_and_preserves_timestamps(self):
        p = Punctuator()
        p._model = _FakeModel({"おはようございます": "おはよう。ございます。"})
        words = [{"text": "おはようございます", "timestamp": (1.0, 3.0)}]
        out = p.restore(words)
        assert out[0]["text"] == "おはよう。ございます。"
        assert out[0]["timestamp"] == (1.0, 3.0)

    def test_only_infers_unpunctuated(self):
        # 已含标点的 chunk 不送模型，原样保留
        p = Punctuator()
        fake = _FakeModel({"あいう": "あ、いう。"})
        p._model = fake
        words = [
            {"text": "もう終わり。", "timestamp": (0.0, 1.0)},
            {"text": "あいう", "timestamp": (1.0, 2.0)},
        ]
        out = p.restore(words)
        assert out[0]["text"] == "もう終わり。"  # 未改
        assert out[1]["text"] == "あ、いう。"  # 补标点
        # 只把无标点的那条送了模型
        assert fake.calls == [["あいう"]]

    def test_unk_result_falls_back_to_original(self):
        p = Punctuator()
        p._model = _FakeModel({"なぞ": "<unk>。"})
        words = [{"text": "なぞ", "timestamp": (0.0, 1.0)}]
        out = p.restore(words)
        assert out[0]["text"] == "なぞ"  # 含 unk，回退原文

    def test_no_unpunctuated_skips_infer(self):
        p = Punctuator()
        fake = _FakeModel({})
        p._model = fake
        words = [{"text": "終わり。", "timestamp": (0.0, 1.0)}]
        out = p.restore(words)
        assert out[0]["text"] == "終わり。"
        assert fake.calls == []  # 无需推理
