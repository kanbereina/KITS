# KITS - 鹿乃 Twitch 直播工具
# Copyright (C) 2026 KanbeReina
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Markdown 总结渲染为分享图片：Markdown -> HTML -> 浏览器截图。

Markdown 解析使用 markdown-it-py；截图使用 Playwright，且只在实际截图时延迟导入。
"""

from __future__ import annotations

from html import escape
from pathlib import Path

__all__ = [
    "MarkdownImageError",
    "available_themes",
    "build_markdown_html",
    "default_image_output_path",
    "render_markdown_to_image",
]


class MarkdownImageError(RuntimeError):
    """Markdown 渲染图片过程中的错误。"""


_THEME_CSS = {
    "light": {
        "page_bg": "#e9edf2",
        "card_bg": "#fbfaf7",
        "text": "#24292f",
        "muted": "#69717d",
        "accent": "#b64b5d",
        "border": "#d8dde4",
        "soft": "#f0ebe4",
        "code_bg": "#eef0f3",
        "shadow": "0 24px 80px rgba(30, 36, 45, 0.18)",
    },
    "dark": {
        "page_bg": "#111317",
        "card_bg": "#1f2329",
        "text": "#edf1f5",
        "muted": "#aeb6c3",
        "accent": "#ffad9a",
        "border": "#3b424d",
        "soft": "#2a3038",
        "code_bg": "#15191f",
        "shadow": "0 24px 80px rgba(0, 0, 0, 0.45)",
    },
}


def available_themes() -> list[str]:
    """返回可用主题名。"""
    return sorted(_THEME_CSS)


def default_image_output_path(markdown_path: str | Path) -> Path:
    """根据 Markdown 文件路径生成默认 PNG 输出路径。"""
    return Path(markdown_path).with_suffix(".png")


def _markdown_to_html(markdown: str) -> str:
    try:
        from markdown_it import MarkdownIt
    except ModuleNotFoundError as e:
        raise MarkdownImageError("未安装 markdown-it-py，无法渲染 Markdown。请运行：uv sync --extra image") from e

    parser = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": True}).enable("table")
    return parser.render(markdown)


def _css(width: int, theme: str) -> str:
    colors = _THEME_CSS[theme]
    card_width = max(640, width)
    return f"""
    :root {{
      color-scheme: {theme};
      --page-bg: {colors["page_bg"]};
      --card-bg: {colors["card_bg"]};
      --text: {colors["text"]};
      --muted: {colors["muted"]};
      --accent: {colors["accent"]};
      --border: {colors["border"]};
      --soft: {colors["soft"]};
      --code-bg: {colors["code_bg"]};
      --shadow: {colors["shadow"]};
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: var(--page-bg); }}
    body {{
      font-family: "Yu Gothic", "YuGothic", "Meiryo", "Microsoft YaHei UI", "Microsoft YaHei",
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans JP", "Noto Sans SC",
        "Hiragino Sans", "PingFang SC", system-ui, sans-serif;
      color: var(--text);
      line-height: 1.72;
      letter-spacing: 0;
      padding: 48px;
    }}
    .kits-card {{
      width: {card_width}px;
      max-width: {card_width}px;
      margin: 0 auto;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 64px 72px;
      overflow: hidden;
    }}
    .kits-meta {{
      color: var(--muted);
      font-size: 20px;
      margin-bottom: 32px;
    }}
    h1, h2, h3, h4, h5, h6 {{
      color: var(--accent);
      line-height: 1.28;
      margin: 1.15em 0 0.55em;
      font-weight: 750;
    }}
    h1:first-child, h2:first-child, h3:first-child {{ margin-top: 0; }}
    h1 {{ font-size: 48px; }}
    h2 {{ font-size: 38px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }}
    h3 {{ font-size: 30px; }}
    h4, h5, h6 {{ font-size: 26px; }}
    p, li, td, th, blockquote {{ font-size: 25px; }}
    p {{ margin: 0 0 1.05em; }}
    ul, ol {{ margin: 0 0 1.15em 1.35em; padding: 0; }}
    li {{ margin: 0.35em 0; padding-left: 0.15em; }}
    li::marker {{ color: var(--accent); font-weight: 700; }}
    strong {{ font-weight: 760; }}
    em {{ color: var(--muted); }}
    a {{ color: var(--accent); text-decoration-thickness: 2px; text-underline-offset: 4px; }}
    blockquote {{
      margin: 1.25em 0;
      padding: 22px 28px;
      border-left: 8px solid var(--accent);
      background: var(--soft);
      border-radius: 8px;
    }}
    blockquote p:last-child {{ margin-bottom: 0; }}
    code {{
      font-family: "Cascadia Mono", "Consolas", "Menlo", monospace;
      font-size: 0.9em;
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.12em 0.35em;
    }}
    pre {{
      margin: 1.2em 0;
      padding: 22px 24px;
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }}
    pre code {{ border: 0; padding: 0; background: transparent; }}
    hr {{ border: 0; border-top: 1px solid var(--border); margin: 36px 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1.2em 0;
      overflow-wrap: anywhere;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 14px 16px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--accent); background: var(--soft); font-weight: 760; }}
    img {{ max-width: 100%; }}
    """


def build_markdown_html(
    markdown: str,
    *,
    width: int = 1200,
    theme: str = "light",
    title: str | None = None,
) -> str:
    """把 Markdown 包装成可截图的完整 HTML。"""
    if width < 640:
        raise MarkdownImageError("图片宽度不能小于 640")
    if theme not in _THEME_CSS:
        raise MarkdownImageError(f"不支持的图片主题: {theme!r}。当前可用: {', '.join(available_themes())}")
    if not markdown.strip():
        raise MarkdownImageError("Markdown 内容为空，无法渲染图片")

    body = _markdown_to_html(markdown)
    meta = f'<div class="kits-meta">{escape(title)}</div>' if title else ""
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>{_css(width, theme)}</style>
</head>
<body>
  <main class="kits-card">
    {meta}
    <article class="kits-markdown">
      {body}
    </article>
  </main>
</body>
</html>
"""


def _playwright_install_hint(error: Exception) -> str:
    message = str(error)
    if "Executable doesn't exist" in message or "playwright install" in message.lower():
        return "Playwright 浏览器未安装。请运行：uv run --extra image playwright install chromium"
    return f"Playwright 截图失败: {message}"


def render_markdown_to_image(
    markdown: str,
    output_path: str | Path,
    *,
    width: int = 1200,
    theme: str = "light",
    title: str | None = None,
    scale: float = 2.0,
) -> Path:
    """把 Markdown 渲染为 HTML 后截图为 PNG，返回输出路径。"""
    if scale <= 0:
        raise MarkdownImageError("截图缩放比例必须大于 0")

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as e:
        raise MarkdownImageError("未安装 Playwright，无法截图。请运行：uv sync --extra image") from e

    html = build_markdown_html(markdown, width=width, theme=theme, title=title)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(
                    viewport={"width": width + 96, "height": 800},
                    device_scale_factor=scale,
                )
                page.set_content(html, wait_until="networkidle")
                card = page.locator(".kits-card")
                card.screenshot(path=str(output), type="png")
            finally:
                browser.close()
    except PlaywrightError as e:
        raise MarkdownImageError(_playwright_install_hint(e)) from e

    return output
