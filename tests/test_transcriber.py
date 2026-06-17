"""transcriber 模块单元测试：只覆盖不碰 GPU / 不触网的纯逻辑。

分段规划（plan_segments）、时间偏移（_shift_words）、静音日志解析。
真正的转录需要 CUDA + 模型，不在此单测范围内。
"""

from __future__ import annotations

from kits.transcriber import (
    _keep_core_words,
    _longest_silence_midpoint,
    _shift_words,
    _word_center,
    plan_segments,
)


class TestPlanSegments:
    def test_short_audio_single_segment(self):
        # 时长不超过 max_chunk，整段一刀不切
        segments = plan_segments(300.0, [], target_chunk=300.0, max_chunk=600.0)
        assert segments == [(0.0, 300.0)]

    def test_segments_cover_full_duration(self):
        # 分段必须无缝覆盖 [0, duration]
        silences = [(290.0, 295.0), (590.0, 596.0), (880.0, 885.0)]
        segments = plan_segments(1000.0, silences, target_chunk=300.0, max_chunk=600.0)
        assert segments[0][0] == 0.0
        assert segments[-1][1] == 1000.0
        for prev, nxt in zip(segments, segments[1:]):
            assert prev[1] == nxt[0]  # 首尾相接，无缝隙无重叠

    def test_cut_falls_on_silence_midpoint(self):
        # 软下限(300)之后的第一个静音中点是 (310+320)/2 = 315
        silences = [(310.0, 320.0)]
        segments = plan_segments(700.0, silences, target_chunk=300.0, max_chunk=600.0)
        assert segments[0] == (0.0, 315.0)

    def test_hard_limit_when_no_silence(self):
        # 软下限到硬上限间无静音（如长时间唱歌），在硬上限强切
        segments = plan_segments(1000.0, [], target_chunk=300.0, max_chunk=600.0)
        assert segments[0] == (0.0, 600.0)
        assert segments[1][0] == 600.0

    def test_ignores_silence_before_soft_limit(self):
        # 软下限(300)之前的静音不作为切点
        silences = [(100.0, 105.0), (400.0, 410.0)]
        segments = plan_segments(700.0, silences, target_chunk=300.0, max_chunk=600.0)
        # 应在 405 处切，而非 102.5
        assert segments[0] == (0.0, 405.0)

    def test_fallback_none_preserves_original_behavior(self):
        # fallback_silences 默认 None：无静音时仍在硬上限强切（原契约不变）
        segments = plan_segments(1000.0, [], target_chunk=300.0, max_chunk=600.0)
        assert segments[0] == (0.0, 600.0)

    def test_fallback_used_before_hard_cut(self):
        # 严格阈值无静音、但宽松候选在 (300,600) 有中点 (440+460)/2=450 → 用它而非硬切 600
        fb = [(440.0, 460.0)]
        segments = plan_segments(
            1000.0, [], target_chunk=300.0, max_chunk=600.0, fallback_silences=fb
        )
        assert segments[0] == (0.0, 450.0)

    def test_fallback_outside_window_falls_back_to_hard_cut(self):
        # 宽松候选都不在 (300,600) 区间内 → 仍硬切在 600
        fb = [(100.0, 110.0), (700.0, 710.0)]
        segments = plan_segments(
            1000.0, [], target_chunk=300.0, max_chunk=600.0, fallback_silences=fb
        )
        assert segments[0] == (0.0, 600.0)

    def test_strict_silence_preferred_over_fallback(self):
        # 严格阈值已有候选(315)时，不动用宽松候选
        strict = [(310.0, 320.0)]
        fb = [(340.0, 360.0)]
        segments = plan_segments(
            700.0, strict, target_chunk=300.0, max_chunk=600.0, fallback_silences=fb
        )
        assert segments[0] == (0.0, 315.0)

    def test_picks_longest_silence_not_first(self):
        # 窗口(300,600)内有 3 段静音：310(停4s) / 450(停20s) / 520(停2s)
        # 应切最长的 450 处，而非最早的 310
        silences = [(308.0, 312.0), (440.0, 460.0), (519.0, 521.0)]
        segments = plan_segments(900.0, silences, target_chunk=300.0, max_chunk=600.0)
        assert segments[0] == (0.0, 450.0)

    def test_longest_silence_tie_prefers_earlier(self):
        # 并列最长时取更靠前者：两段都停 10s，取 350 而非 500
        silences = [(345.0, 355.0), (495.0, 505.0)]
        segments = plan_segments(900.0, silences, target_chunk=300.0, max_chunk=600.0)
        assert segments[0] == (0.0, 350.0)


class TestLongestSilenceMidpoint:
    def test_returns_longest_in_window(self):
        sil = [(310.0, 314.0), (400.0, 420.0), (500.0, 505.0)]
        assert _longest_silence_midpoint(sil, 300.0, 600.0) == 410.0

    def test_excludes_out_of_window(self):
        # 窗口外的更长静音不算；窗口内只有 350 合格
        sil = [(100.0, 200.0), (348.0, 352.0)]
        assert _longest_silence_midpoint(sil, 300.0, 600.0) == 350.0

    def test_none_when_empty_window(self):
        sil = [(100.0, 150.0), (700.0, 760.0)]
        assert _longest_silence_midpoint(sil, 300.0, 600.0) is None

    def test_boundary_is_exclusive(self):
        # 中点正好落在 lo 或 hi 上不算（区间开）
        sil = [(295.0, 305.0)]  # 中点 300 == lo
        assert _longest_silence_midpoint(sil, 300.0, 600.0) is None


class TestShiftWords:
    def test_adds_offset_to_timestamps(self):
        words = [{"text": "あ", "timestamp": (1.0, 2.0)}]
        shifted = _shift_words(words, 100.0)
        assert shifted[0]["timestamp"] == (101.0, 102.0)
        assert shifted[0]["text"] == "あ"

    def test_preserves_none_timestamps(self):
        words = [{"text": "い", "timestamp": (None, 2.0)}]
        shifted = _shift_words(words, 50.0)
        assert shifted[0]["timestamp"] == (None, 52.0)

    def test_zero_offset_unchanged(self):
        words = [{"text": "う", "timestamp": (1.0, 2.0)}]
        assert _shift_words(words, 0.0)[0]["timestamp"] == (1.0, 2.0)

    def test_empty_list(self):
        assert _shift_words([], 10.0) == []


class TestWordCenter:
    def test_both_ends_uses_midpoint(self):
        assert _word_center({"text": "あ", "timestamp": (10.0, 20.0)}) == 15.0

    def test_only_end_uses_end(self):
        # kotoba chunk 常见尾部 end=None 之外，也有只有 end 的情形
        assert _word_center({"text": "い", "timestamp": (None, 20.0)}) == 20.0

    def test_only_start_uses_start(self):
        assert _word_center({"text": "う", "timestamp": (10.0, None)}) == 10.0

    def test_both_none_returns_none(self):
        assert _word_center({"text": "え", "timestamp": (None, None)}) is None


class TestKeepCoreWords:
    def _w(self, text: str, start: float, end: float):
        return {"text": text, "timestamp": (start, end)}

    def test_drops_words_in_left_padding(self):
        # 逻辑区间 [100, 200)，垫料区 [98,100) 的词中心=99 应被丢弃
        words = [self._w("pad", 98.0, 100.0), self._w("core", 150.0, 160.0)]
        kept = _keep_core_words(words, 100.0, 200.0)
        assert [w["text"] for w in kept] == ["core"]

    def test_drops_words_in_right_padding(self):
        # 右垫料区中心 >= hi 被丢弃；正好落在 hi 上的也丢（区间右开）
        words = [self._w("core", 150.0, 160.0), self._w("pad", 200.0, 210.0)]
        kept = _keep_core_words(words, 100.0, 200.0)
        assert [w["text"] for w in kept] == ["core"]

    def test_none_bounds_keep_everything_in_range(self):
        # 首段左侧 lo=None：不丢左边的词；末段右侧 hi=None：不丢右边的词
        words = [self._w("a", 0.0, 5.0), self._w("b", 500.0, 510.0)]
        assert _keep_core_words(words, None, None) == words

    def test_keeps_word_with_unknown_center(self):
        # 时间戳全缺无法定位中心，保守保留避免丢内容
        words = [{"text": "x", "timestamp": (None, None)}]
        assert _keep_core_words(words, 100.0, 200.0) == words

    def test_seam_no_gap_no_overlap(self):
        # 相邻两段接缝去重：中心正好落在边界 150 的词只归右段一次（区间左闭右开）
        prev = _keep_core_words([self._w("seam", 148.0, 152.0)], 100.0, 150.0)
        nxt = _keep_core_words([self._w("seam", 148.0, 152.0)], 150.0, 200.0)
        assert [w["text"] for w in prev] == []   # 中心 150 >= hi(150)，左段丢
        assert [w["text"] for w in nxt] == ["seam"]  # 中心 150 >= lo(150)，右段留
