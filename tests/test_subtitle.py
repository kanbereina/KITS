"""subtitle 模块单元测试：纯逻辑，不依赖 torch / 网络。"""

from __future__ import annotations

from kits.subtitle import (
    SrtWriter,
    Word,
    _clamp_overlaps,
    _split_internal_punctuation,
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

    def test_collapses_phrase_repeats(self):
        # 2 字短语重复 3 次以上压成 2 次（抑制「はっはっはっ」式幻觉）
        assert clean_text("はっはっはっはっはっ") == "はっはっ"

    def test_collapses_longer_phrase_repeats(self):
        # 多字短语重复也折叠
        assert clean_text("ここだここだここだここだ") == "ここだここだ"

    def test_keeps_single_phrase_repeat(self):
        # 只重复一次（出现 2 次）属正常强调，不折叠
        assert clean_text("だめだめ") == "だめだめ"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestSplitInternalPunctuation:
    def test_splits_chunk_with_internal_ending(self):
        # 一个 chunk 内含句中句号，应拆成两个 Word，时间按字符比例分配
        words = [_word("そうだ。どう", 0.0, 10.0)]
        out = _split_internal_punctuation(words)
        assert [w["text"] for w in out] == ["そうだ。", "どう"]
        # "そうだ。" 占 4/6 字符 → end ≈ 0 + 10*4/6
        assert abs(out[0]["timestamp"][1] - 10.0 * 4 / 6) < 1e-6
        assert out[1]["timestamp"][0] == out[0]["timestamp"][1]
        assert out[1]["timestamp"][1] == 10.0

    def test_multiple_internal_endings(self):
        words = [_word("あ?い。う", 0.0, 9.0)]
        out = _split_internal_punctuation(words)
        assert [w["text"] for w in out] == ["あ?", "い。", "う"]

    def test_no_split_when_punct_at_end(self):
        # 标点恰在末尾不算内部切点，原样返回（交给 segment_sentences 正常处理）
        words = [_word("こんにちは。", 0.0, 2.0)]
        out = _split_internal_punctuation(words)
        assert out == words

    def test_no_split_without_punctuation(self):
        words = [_word("ただのテキスト", 0.0, 2.0)]
        out = _split_internal_punctuation(words)
        assert out == words

    def test_no_split_when_timestamps_missing(self):
        # 时间戳缺失无法按比例分配，原样透传不拆
        words = [_word("そう。どう", None, None)]
        out = _split_internal_punctuation(words)
        assert out == words

    def test_preserves_normal_words(self):
        words = [_word("ふつう", 0.0, 1.0), _word("の", 1.0, 2.0)]
        assert _split_internal_punctuation(words) == words


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

    def test_splits_on_internal_sentence_ending(self):
        # 单个 chunk 内含多句（句末标点在中间）→ 应在标点处断成多条，而非靠 max_duration 硬钳
        words = [_word("そうだ。どうだろう？こんにちは", 0.0, 12.0)]
        result = segment_sentences(words, max_duration=15.0)
        assert [s["text"] for s in result] == ["そうだ。", "どうだろう？", "こんにちは"]
        # 时间轴单调不重叠
        assert result[0]["start"] == 0.0
        assert result[0]["end"] <= result[1]["start"]
        assert result[1]["end"] <= result[2]["start"]

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

    def test_clamps_duration_when_timestamps_missing(self):
        # 中间词时间戳缺失，老逻辑会让句子无限拉长；现应被钳制在 max_duration 内
        words = [_word("あ", 0.0, 1.0)]
        words += [_word("ん", None, None) for _ in range(50)]
        words.append(_word("ん", 120.0, 121.0))
        result = segment_sentences(
            words, max_gap=1000.0, max_chars=10000, max_duration=15.0
        )
        for sent in result:
            assert sent["end"] - sent["start"] <= 15.0 + 1e-6

    def test_hard_splits_when_over_max_chars_without_soft_break(self):
        # 超过字符上限且无逗号可切：按词边界硬切，而非整段合成一条
        words = [_word("あいうえお", float(i), float(i) + 1.0) for i in range(5)]
        result = segment_sentences(
            words, max_gap=1000.0, max_chars=5, max_duration=1000.0
        )
        assert len(result) > 1

    def test_duration_split_uses_last_known_end(self):
        # 时长判定基于「最后已知 end - 第一已知 start」，即便末词时间戳缺失也能断
        words = [
            _word("あ、", 0.0, 1.0),
            _word("い", 20.0, 21.0),
            _word("う", None, None),
        ]
        result = segment_sentences(
            words, max_gap=1000.0, max_chars=10000, max_duration=15.0
        )
        assert len(result) >= 2

    def test_shrinks_inflated_short_sentence_by_char_rate(self):
        # kotoba 常把短句标成虚高时长：十几字短句却跨 14s。
        # 句末标点断句正确、max_duration 也没超，但应按字符速率收缩 end。
        text = "さくらぜサクラゼさん39か月。"
        words = [_word(text, 266.0, 280.0)]
        result = segment_sentences(words, max_seconds_per_char=0.5)
        assert len(result) == 1
        # 上限 = 字符数 * 0.5s，远小于原始 14s
        assert result[0]["end"] - result[0]["start"] <= len(text) * 0.5 + 1e-6
        assert result[0]["end"] - result[0]["start"] < 14.0  # 确实被收缩了
        assert result[0]["start"] == 266.0  # 起点不动
        assert result[0]["text"].endswith("。")  # 文本不动

    def test_char_rate_disabled_when_zero(self):
        # max_seconds_per_char<=0 时关闭二级收缩，时长仅受 max_duration 约束
        words = [_word("短い。", 0.0, 14.0)]
        result = segment_sentences(words, max_seconds_per_char=0.0, max_duration=15.0)
        assert abs(result[0]["end"] - result[0]["start"] - 14.0) < 1e-6

    def test_char_rate_keeps_normal_pace_sentences(self):
        # 正常语速的句子（时长与字符匹配）不应被误缩
        words = [_word("これはふつうのはやさです。", 0.0, 5.0)]  # 13 字 / 5s，约 0.38s/字
        result = segment_sentences(words, max_seconds_per_char=0.5)
        assert abs(result[0]["end"] - result[0]["start"] - 5.0) < 1e-6

    def test_min_duration_floor_for_near_zero_chunk(self):
        # kotoba 偶尔把短 chunk 的 start/end 挤在一点（end-start≈0.04s），
        # 应兜底撑到最小显示时长（0.5s），避免播放器一闪而过/不显示。
        words = [_word("レシテじゃん。", 17.48, 17.52)]
        result = segment_sentences(words)
        assert len(result) == 1
        assert result[0]["end"] - result[0]["start"] >= 0.5 - 1e-6
        assert result[0]["start"] == 17.48  # 起点不动

    def test_no_overlap_after_min_duration_stretch(self):
        # min_duration 撑长近零时长条目后，不得越过紧邻下一条的 start（消重叠）。
        # 第一条 end≈start，撑到 +0.5 会到 17.98，但下一条 17.5 开始 → 应夹回 17.5。
        words = [
            _word("レシテじゃん。", 17.48, 17.50),
            _word("つぎ。", 17.50, 21.0),
        ]
        result = segment_sentences(words)
        assert len(result) == 2
        assert result[0]["end"] <= result[1]["start"] + 1e-6  # 无重叠

    def test_clamp_overlaps_collapses_seam_overlap(self):
        # 接缝重叠：前条 end 大于后条 start，应夹到后条 start
        sents = [
            {"start": 509.0, "end": 513.5, "text": "前"},
            {"start": 512.8, "end": 527.8, "text": "后"},
        ]
        out = _clamp_overlaps(sents)
        assert out[0]["end"] == 512.8
        assert out[1]["start"] == 512.8


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


class TestSrtWriter:
    def test_single_batch_matches_write_srt(self, tmp_path):
        # 单批追加的结果应与一次性 write_srt 完全一致
        sentences = [
            {"start": 0.0, "end": 2.6, "text": "こんにちは。"},
            {"start": 5.0, "end": 6.8, "text": "ありがとう"},
        ]
        out = tmp_path / "incr.srt"
        with SrtWriter(str(out)) as w:
            w.append(sentences)
        assert out.read_text(encoding="utf-8") == sentences_to_srt(sentences)

    def test_multiple_batches_continuous_index(self, tmp_path):
        # 跨批序号连续，且可被 parse_srt 还原
        out = tmp_path / "incr.srt"
        with SrtWriter(str(out)) as w:
            n1 = w.append([{"start": 0.0, "end": 1.0, "text": "一"}])
            n2 = w.append([
                {"start": 2.0, "end": 3.0, "text": "二"},
                {"start": 4.0, "end": 5.0, "text": "三"},
            ])
        assert n1 == 1
        assert n2 == 3
        parsed = parse_srt(out.read_text(encoding="utf-8"))
        assert [s["text"] for s in parsed] == ["一", "二", "三"]

    def test_count_property(self, tmp_path):
        out = tmp_path / "incr.srt"
        with SrtWriter(str(out)) as w:
            assert w.count == 0
            w.append([{"start": 0.0, "end": 1.0, "text": "x"}])
            assert w.count == 1

    def test_partial_output_is_valid_srt(self, tmp_path):
        # 模拟中途中断：只写了一批就 close，剩下的文件仍可解析
        out = tmp_path / "incr.srt"
        w = SrtWriter(str(out))
        w.append([{"start": 0.0, "end": 1.0, "text": "已完成"}])
        w.close()
        parsed = parse_srt(out.read_text(encoding="utf-8"))
        assert len(parsed) == 1
        assert parsed[0]["text"] == "已完成"
