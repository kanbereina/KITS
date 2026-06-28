"""cli 模块单元测试：只覆盖不碰 GPU / 不触网的纯逻辑（进度条工厂）。"""

from __future__ import annotations

from kits.cli import _make_bar, build_parser


class TestMakeBar:
    def test_returns_usable_tqdm(self):
        # _make_bar 应返回一个可 update/close 的 tqdm 实例
        bar = _make_bar(total=10, desc="测试", unit="条")
        try:
            assert bar.total == 10
            bar.update(3)
            assert bar.n == 3
        finally:
            bar.close()

    def test_total_none_then_reset(self):
        # total=None 先建空条，后续 reset(total=...) 设定量程（转录流程用法）
        bar = _make_bar(total=None, desc="转录", unit="s")
        try:
            assert bar.total is None
            bar.reset(total=100.0)
            assert bar.total == 100.0
            bar.update(50.0)
            assert bar.n == 50.0
        finally:
            bar.close()

    def test_write_does_not_raise(self, capsys):
        # tqdm.write 是类方法，打印日志不抛错、不影响进度条状态
        from tqdm import tqdm

        bar = _make_bar(total=5, desc="x", unit="it")
        try:
            tqdm.write("一条日志")
            bar.update(1)
            assert bar.n == 1
        finally:
            bar.close()

    def test_desc_set(self):
        bar = _make_bar(total=1, desc="🎬 转录进度", unit="s")
        try:
            assert "转录进度" in bar.desc
        finally:
            bar.close()


class TestRenderImageParser:
    def test_render_image_defaults(self):
        args = build_parser().parse_args(["render-image", "-i", "summary.md"])

        assert args.input == "summary.md"
        assert args.output is None
        assert args.width == 1200
        assert args.theme == "light"
        assert args.scale == 2.0

    def test_render_image_accepts_options(self):
        args = build_parser().parse_args(
            [
                "img",
                "-i",
                "summary.md",
                "-o",
                "summary.png",
                "--width",
                "900",
                "--theme",
                "dark",
                "--scale",
                "1.5",
                "--title",
                "直播总结",
            ]
        )

        assert args.output == "summary.png"
        assert args.width == 900
        assert args.theme == "dark"
        assert args.scale == 1.5
        assert args.title == "直播总结"

    def test_summarize_accepts_render_image_options(self):
        args = build_parser().parse_args(
            [
                "summarize",
                "-i",
                "live.srt",
                "--render-image",
                "--image-output",
                "live.png",
                "--image-theme",
                "dark",
            ]
        )

        assert args.render_image is True
        assert args.image_output == "live.png"
        assert args.image_theme == "dark"
        assert args.image_width == 1200
