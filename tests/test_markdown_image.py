"""markdown_image 模块单元测试：覆盖 Markdown -> HTML 纯逻辑，不启动浏览器。"""

from __future__ import annotations

import pytest

from kits.markdown_image import (
    MarkdownImageError,
    available_themes,
    build_markdown_html,
    default_image_output_path,
)


class TestBuildMarkdownHtml:
    def test_renders_markdown_to_html_document(self):
        html = build_markdown_html(
            "# まとめ\n\n- こんにちは\n- 歌います\n\n| 時間 | 内容 |\n| --- | --- |\n| 00:01 | 雑談 |",
            title="live.summary.md",
        )

        assert "<!doctype html>" in html
        assert '<main class="kits-card">' in html
        assert "<h1>まとめ</h1>" in html
        assert "<li>こんにちは</li>" in html
        assert "<table>" in html
        assert "live.summary.md" in html

    def test_escapes_raw_html(self):
        html = build_markdown_html("<script>alert(1)</script>")

        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_supports_dark_theme(self):
        html = build_markdown_html("# 标题", theme="dark")

        assert "color-scheme: dark" in html

    def test_rejects_unknown_theme(self):
        with pytest.raises(MarkdownImageError):
            build_markdown_html("# x", theme="sepia")

    def test_rejects_too_narrow_width(self):
        with pytest.raises(MarkdownImageError):
            build_markdown_html("# x", width=320)

    def test_rejects_empty_markdown(self):
        with pytest.raises(MarkdownImageError):
            build_markdown_html("  ")


class TestHelpers:
    def test_available_themes(self):
        assert available_themes() == ["dark", "light"]

    def test_default_image_output_path(self):
        assert str(default_image_output_path("live.summary.md")).endswith("live.summary.png")
