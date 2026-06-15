"""transcriber 模块单元测试：只覆盖不碰 GPU / 不触网的纯逻辑。

分段规划（plan_segments）、时间偏移（_shift_words）、静音日志解析。
真正的转录需要 CUDA + 模型，不在此单测范围内。
"""

from __future__ import annotations

from kits.transcriber import _shift_words, plan_segments


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
