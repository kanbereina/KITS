"""vad 模块单元测试：只覆盖不碰 torch / 不触网的纯逻辑。

speech_to_gaps（语音区间取补集得非语音间隙）是分段切点的来源，纯逻辑、可独立单测。
真正的 VAD 推理需要 torch + silero 模型，不在此单测范围内。
"""

from __future__ import annotations

from kits.vad import speech_to_gaps


class TestSpeechToGaps:
    def test_gaps_between_speech(self):
        # 两段语音之间的空档是间隙；首尾也各有间隙
        speech = [(10.0, 20.0), (30.0, 40.0)]
        gaps = speech_to_gaps(speech, 50.0)
        assert gaps == [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]

    def test_empty_speech_whole_duration(self):
        # 全程无人声（纯 BGM）：整段是一个大间隙，交给 plan_segments 自由切
        assert speech_to_gaps([], 100.0) == [(0.0, 100.0)]

    def test_speech_from_start_no_leading_gap(self):
        # 语音从 0 开始：开头无间隙
        gaps = speech_to_gaps([(0.0, 30.0)], 50.0)
        assert gaps == [(30.0, 50.0)]

    def test_speech_to_end_no_trailing_gap(self):
        # 语音持续到结尾：结尾无间隙
        gaps = speech_to_gaps([(0.0, 20.0), (30.0, 50.0)], 50.0)
        assert gaps == [(20.0, 30.0)]

    def test_full_speech_no_gap(self):
        # 全程都是人声：无间隙
        assert speech_to_gaps([(0.0, 50.0)], 50.0) == []

    def test_adjacent_speech_no_gap_between(self):
        # 紧邻的两段语音（end==next start）之间不产生间隙
        gaps = speech_to_gaps([(0.0, 20.0), (20.0, 40.0)], 40.0)
        assert gaps == []

    def test_unsorted_input_is_sorted(self):
        # 上游顺序异常时按起点排序后再取补集，结果不错乱
        speech = [(30.0, 40.0), (10.0, 20.0)]
        gaps = speech_to_gaps(speech, 50.0)
        assert gaps == [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]

    def test_overlapping_speech_merged(self):
        # 重叠语音区间：cursor 取 max 推进，不产生负长度/错乱间隙
        speech = [(10.0, 30.0), (20.0, 40.0)]
        gaps = speech_to_gaps(speech, 50.0)
        assert gaps == [(0.0, 10.0), (40.0, 50.0)]

    def test_nested_speech_interval(self):
        # 后一区间被前一区间完全包含：cursor 不回退
        speech = [(10.0, 40.0), (20.0, 30.0)]
        gaps = speech_to_gaps(speech, 50.0)
        assert gaps == [(0.0, 10.0), (40.0, 50.0)]
