"""filters 模块单元测试：纯逻辑，不依赖 torch / 网络。"""

from __future__ import annotations

from kits.filters import (
    VALORANT_CALLOUTS,
    filter_sentences,
    is_game_callout,
    resolve_games,
    supported_games,
)
from kits.subtitle import Sentence


def _sent(text: str) -> Sentence:
    return {"start": 0.0, "end": 1.0, "text": text}


class TestIsGameCallout:
    def test_exact_match(self):
        assert is_game_callout("アラームボット配置")

    def test_match_ignores_trailing_punctuation(self):
        assert is_game_callout("残り1名！")

    def test_match_ignores_spaces(self):
        assert is_game_callout("セントリー 設置")

    def test_keyword_with_long_vowel_preserved(self):
        # 长音符不被当标点剥离，セントリー 仍能匹配
        assert is_game_callout("セントリー設置")

    def test_callout_with_kill_location(self):
        assert is_game_callout("Bでキャリアダウン")

    def test_non_callout_human_speech(self):
        assert not is_game_callout("おはようございます")

    def test_partial_match_not_removed(self):
        # 播报词只是整条的一部分时不算匹配（保守策略，避免误删人声）
        assert not is_game_callout("中央に敵だご視聴ありがとうございました")

    def test_empty_string(self):
        assert not is_game_callout("")


class TestFilterSentences:
    def test_removes_only_callouts(self):
        sentences = [
            _sent("おはようございます"),
            _sent("アラームボット配置"),
            _sent("元気してましたか"),
            _sent("残り1名"),
        ]
        kept, removed = filter_sentences(sentences)
        assert removed == 2
        assert [s["text"] for s in kept] == ["おはようございます", "元気してましたか"]

    def test_keeps_mixed_lines(self):
        # 播报混入人声的条目整条保留（仅删完全匹配）
        sentences = [_sent("中央敵ご視聴ありがとうございました中央に敵だ")]
        kept, removed = filter_sentences(sentences)
        assert removed == 0
        assert len(kept) == 1

    def test_empty_input(self):
        kept, removed = filter_sentences([])
        assert kept == []
        assert removed == 0

    def test_all_callouts_removed(self):
        sentences = [_sent("残り1名"), _sent("最後の一人だ")]
        kept, removed = filter_sentences(sentences)
        assert kept == []
        assert removed == 2

    def test_custom_callout_set(self):
        kept, removed = filter_sentences(
            [_sent("foo"), _sent("bar")], callouts=frozenset({"foo"})
        )
        assert removed == 1
        assert [s["text"] for s in kept] == ["bar"]


class TestCalloutSet:
    def test_set_is_normalized_and_nonempty(self):
        assert len(VALORANT_CALLOUTS) > 0
        # 词表已归一化：无空格、无首尾标点
        for c in VALORANT_CALLOUTS:
            assert c == c.strip("。！？、 ")
            assert " " not in c


class TestResolveGames:
    def test_resolve_full_name(self):
        assert resolve_games(["valorant"]) == VALORANT_CALLOUTS

    def test_resolve_is_case_insensitive(self):
        assert resolve_games(["VALORANT"]) == VALORANT_CALLOUTS
        assert resolve_games(["Valorant"]) == VALORANT_CALLOUTS

    def test_resolve_alias(self):
        assert resolve_games(["valo"]) == VALORANT_CALLOUTS

    def test_resolve_strips_whitespace(self):
        assert resolve_games(["  valo  "]) == VALORANT_CALLOUTS

    def test_resolve_multiple_dedupes(self):
        # 同义重复传入合并后等于单个词表
        assert resolve_games(["valorant", "valo"]) == VALORANT_CALLOUTS

    def test_resolve_empty_list(self):
        assert resolve_games([]) == frozenset()

    def test_unknown_game_raises(self):
        try:
            resolve_games(["lol"])
        except ValueError as e:
            assert "lol" in str(e)
        else:
            raise AssertionError("应对未知游戏名抛 ValueError")

    def test_supported_games_lists_aliases(self):
        games = supported_games()
        assert "valorant" in games
        assert "valo" in games
