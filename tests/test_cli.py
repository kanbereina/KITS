"""cli 模块单元测试：只覆盖不碰 GPU / 不触网的纯逻辑（进度条工厂）。"""

from __future__ import annotations

import pytest

from kits.cli import _make_bar, _split_ytdlp_passthrough, build_parser


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


class TestDownloadParser:
    def test_download_defaults_to_ytdlp_audio(self):
        args = build_parser().parse_args(["download", "https://example.com/watch?v=abc"])

        assert args.url == "https://example.com/watch?v=abc"
        assert args.output == "output"
        assert args.dir == "downloads"
        assert args.yt_dlp_args is None
        assert args.srt is False

    def test_download_accepts_ytdlp_args_string(self):
        args = build_parser().parse_args(
            [
                "download",
                "https://www.youtube.com/watch?v=abc",
                "--yt-dlp-args",
                "-f bestaudio --extract-audio",
            ]
        )

        assert args.yt_dlp_args == "-f bestaudio --extract-audio"

    def test_split_ytdlp_passthrough_keeps_normal_kits_args(self):
        argv, passthrough = _split_ytdlp_passthrough(["download", "url", "--srt"])

        assert argv == ["download", "url", "--srt"]
        assert passthrough == []

    def test_split_ytdlp_passthrough_extracts_args_after_separator(self):
        argv, passthrough = _split_ytdlp_passthrough(["download", "url", "--srt", "--", "-f", "bestaudio"])

        assert argv == ["download", "url", "--srt"]
        assert passthrough == ["-f", "bestaudio"]

    def test_rejects_passthrough_for_non_download_command(self):
        with pytest.raises(SystemExit):
            from kits.cli import main

            main(["subtitle", "-i", "audio.mp3", "--", "-f", "bestaudio"])
