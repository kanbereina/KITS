"""downloader 模块单元测试：仅覆盖不触网的纯逻辑（URL 解析）。"""

from __future__ import annotations

import pytest

from kits.downloader import parse_url_pattern


class TestParseUrlPattern:
    def test_parses_standard_url(self):
        base, num, ext = parse_url_pattern(
            "https://example.com/path/chunked/1710.ts"
        )
        assert base == "https://example.com/path/chunked/"
        assert num == 1710
        assert ext == ".ts"

    def test_parses_zero_index(self):
        base, num, ext = parse_url_pattern("https://x.com/chunked/0.ts")
        assert num == 0

    def test_raises_on_invalid_url(self):
        with pytest.raises(ValueError):
            parse_url_pattern("https://example.com/not-a-chunk.mp4")

    def test_raises_without_chunked_segment(self):
        with pytest.raises(ValueError):
            parse_url_pattern("https://example.com/video/1710.ts")
