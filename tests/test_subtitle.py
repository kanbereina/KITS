"""subtitle 模块单元测试：纯逻辑，不依赖 torch / 网络。"""

from __future__ import annotations

from kits.subtitle import (
    Word,
    clean_text,
    parse_srt,
    seconds_to_srt_time,
    segment_sentences,
    sentences_to_srt,
    srt_time_to_seconds,
    write_srt,
)


def _word(text: str, start: float | None, end: float | None) -> Word:
    """构造 Word 的便捷函数。"""
    return {"text": text, "timestamp": (start, end)}


class TestSecondsToSrtTime:
    def test_zero(self):
        assert seconds_to_srt_time(0) == "00:00:00,000"

    def test_with_millis(self):
        assert seconds_to_srt_time(1.5) == "00:00:01,500"

    def test_hours_minutes_seconds(self):
        # 1 小时 2 分 3.456 秒
        assert seconds_to_srt_time(3723.456) == "01:02:03,456"

    def test_none_treated_as_zero(self):
        assert seconds_to_srt_time(None) == "00:00:00,000"

    def test_negative_treated_as_zero(self):
        assert seconds_to_srt_time(-5.0) == "00:00:00,000"

    def test_rounds_millis(self):
        # 0.0016 秒四舍五入到 2 毫秒（round 用银行家舍入，故取非 .5 边界值）
        assert seconds_to_srt_time(0.0016) == "00:00:00,002"


class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert clean_text("a    b") == "a b"

    def test_removes_garbage_char(self):
        assert clean_text("こんにちは�") == "こんにちは"

    def test_collapses_small_kana_repeats(self):
        # 小写假名连续 3 次以上压成 1 个
        assert clean_text("あっっっと") == "あっと"

    def test_collapses_long_char_repeats(self):
        # 同一字符连续 4 次以上压成 2 次
        assert clean_text("わーーーーい") == "わーーい"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestSegmentSentences:
    def test_empty_input(self):
        assert segment_sentences([]) == []

    def test_single_sentence_with_ending_punctuation(self):
        words = [
            _word("こんにちは", 0.0, 1.0),
            _word("。", 1.0, 1.2),
        ]
        result = segment_sentences(words)
        assert len(result) == 1
        assert result[0]["text"] == "こんにちは。"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.2

    def test_splits_on_sentence_ending(self):
        words = [
            _word("はい", 0.0, 0.5),
            _word("。", 0.5, 0.6),
            _word("いいえ", 0.7, 1.2),
            _word("。", 1.2, 1.3),
        ]
        result = segment_sentences(words)
        assert len(result) == 2
        assert result[0]["text"] == "はい。"
        assert result[1]["text"] == "いいえ。"

    def test_splits_on_long_gap(self):
        # 两个词之间停顿超过 max_gap=0.7，应断句
        words = [
            _word("あ", 0.0, 0.5),
            _word("い", 2.0, 2.5),
        ]
        result = segment_sentences(words, max_gap=0.7)
        assert len(result) == 2
        assert result[0]["text"] == "あ"
        assert result[1]["text"] == "い"

    def test_no_split_on_small_gap(self):
        words = [
            _word("あ", 0.0, 0.5),
            _word("い", 0.6, 1.0),
        ]
        result = segment_sentences(words, max_gap=0.7)
        assert len(result) == 1
        assert result[0]["text"] == "あい"

    def test_splits_at_soft_break_when_over_max_chars(self):
        # 超过字符上限时，优先在读点处切开
        words = [
            _word("あいうえお、", 0.0, 1.0),
            _word("かきくけこ", 1.0, 2.0),
        ]
        result = segment_sentences(words, max_chars=5)
        assert len(result) == 2
        assert result[0]["text"] == "あいうえお、"
        assert result[1]["text"] == "かきくけこ"

    def test_splits_on_max_duration(self):
        # 超过时长上限应断句
        words = [
            _word("あ、", 0.0, 1.0),
            _word("い", 20.0, 21.0),
        ]
        result = segment_sentences(words, max_gap=100.0, max_duration=15.0)
        assert len(result) == 2

    def test_trailing_buffer_is_flushed(self):
        # 没有结尾标点的剩余内容也要输出
        words = [_word("おわり", 0.0, 1.0)]
        result = segment_sentences(words)
        assert len(result) == 1
        assert result[0]["text"] == "おわり"

    def test_handles_none_timestamps(self):
        words = [_word("テスト", None, None)]
        result = segment_sentences(words)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        # end 在 start 之上有兜底
        assert result[0]["end"] > result[0]["start"]


class TestSentencesToSrt:
    def test_renders_numbered_blocks(self):
        sentences = [
            {"start": 0.0, "end": 2.6, "text": "こんにちは。"},
            {"start": 5.0, "end": 6.8, "text": "ありがとう"},
        ]
        srt = sentences_to_srt(sentences)
        expected = (
            "1\n00:00:00,000 --> 00:00:02,600\nこんにちは。\n"
            "\n"
            "2\n00:00:05,000 --> 00:00:06,800\nありがとう\n"
        )
        assert srt == expected

    def test_empty_list(self):
        assert sentences_to_srt([]) == ""


class TestWriteSrt:
    def test_writes_utf8_file(self, tmp_path):
        sentences = [{"start": 0.0, "end": 1.0, "text": "テスト。"}]
        out = tmp_path / "out.srt"
        write_srt(sentences, str(out))
        content = out.read_text(encoding="utf-8")
        assert "テスト。" in content
        assert "00:00:00,000 --> 00:00:01,000" in content


class TestSrtTimeToSeconds:
    def test_zero(self):
        assert srt_time_to_seconds("00:00:00,000") == 0.0

    def test_with_millis(self):
        assert srt_time_to_seconds("00:00:01,500") == 1.5

    def test_hours_minutes_seconds(self):
        assert srt_time_to_seconds("01:02:03,456") == 3723.456

    def test_roundtrip_with_seconds_to_srt_time(self):
        # 与 seconds_to_srt_time 互逆
        for sec in (0.0, 1.5, 3723.456, 59.999):
            assert srt_time_to_seconds(seconds_to_srt_time(sec)) == sec


class TestParseSrt:
    def test_parses_single_block(self):
        srt = "1\n00:00:00,000 --> 00:00:02,600\nこんにちは。\n"
        result = parse_srt(srt)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.6
        assert result[0]["text"] == "こんにちは。"

    def test_parses_multiple_blocks(self):
        srt = (
            "1\n00:00:00,000 --> 00:00:02,600\nこんにちは。\n"
            "\n"
            "2\n00:00:05,000 --> 00:00:06,800\nありがとう\n"
        )
        result = parse_srt(srt)
        assert len(result) == 2
        assert result[1]["text"] == "ありがとう"

    def test_roundtrip_with_sentences_to_srt(self):
        # parse_srt 是 sentences_to_srt 的逆操作
        sentences = [
            {"start": 0.0, "end": 2.6, "text": "こんにちは。"},
            {"start": 5.0, "end": 6.8, "text": "ありがとう"},
        ]
        assert parse_srt(sentences_to_srt(sentences)) == sentences

    def test_handles_crlf_line_endings(self):
        srt = "1\r\n00:00:00,000 --> 00:00:01,000\r\nテスト\r\n"
        result = parse_srt(srt)
        assert len(result) == 1
        assert result[0]["text"] == "テスト"

    def test_handles_multiline_text(self):
        srt = "1\n00:00:00,000 --> 00:00:02,000\n第一行\n第二行\n"
        result = parse_srt(srt)
        assert len(result) == 1
        assert result[0]["text"] == "第一行\n第二行"

    def test_skips_block_without_timestamp(self):
        srt = "这是一段没有时间轴的文字\n\n1\n00:00:00,000 --> 00:00:01,000\n有效\n"
        result = parse_srt(srt)
        assert len(result) == 1
        assert result[0]["text"] == "有效"

    def test_empty_input(self):
        assert parse_srt("") == []
