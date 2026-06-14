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

    def test_parses_quality_dir_other_than_chunked(self):
        # 分片目录名随画质变化，160p30 / 720p60 等都要能解析
        base, num, ext = parse_url_pattern(
            "https://d3vd9lfkzbru3h.cloudfront.net/abc_kanotic_123/160p30/3.ts"
        )
        assert base == "https://d3vd9lfkzbru3h.cloudfront.net/abc_kanotic_123/160p30/"
        assert num == 3
        assert ext == ".ts"

    def test_raises_on_non_ts_extension(self):
        with pytest.raises(ValueError):
            parse_url_pattern("https://example.com/chunked/1710.mp4")

    def test_raises_on_non_numeric_filename(self):
        with pytest.raises(ValueError):
            parse_url_pattern("https://example.com/chunked/playlist.ts")
