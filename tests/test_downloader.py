"""downloader 模块单元测试：仅覆盖不触网的纯逻辑与 yt-dlp 参数构造。"""

from __future__ import annotations

import pytest

from kits.downloader import (
    YtDlpDownloader,
    is_twitch_ts_url,
    parse_url_pattern,
    select_download_backend,
)


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


class TestSelectDownloadBackend:
    def test_auto_uses_twitch_for_numeric_ts_url(self):
        url = "https://example.com/path/chunked/1710.ts"
        assert is_twitch_ts_url(url) is True
        assert select_download_backend(url) == "twitch"

    def test_auto_uses_ytdlp_for_video_page(self):
        url = "https://www.youtube.com/watch?v=abc"
        assert is_twitch_ts_url(url) is False
        assert select_download_backend(url) == "yt-dlp"

    def test_explicit_backend_wins(self):
        url = "https://example.com/path/chunked/1710.ts"
        assert select_download_backend(url, "yt-dlp") == "yt-dlp"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            select_download_backend("https://example.com/video", "unknown")


class TestYtDlpDownloader:
    def test_build_options_keeps_video_and_extracts_mp3(self, tmp_path):
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        options = downloader._build_options(
            "live",
            extract_audio=True,
            keep_video=True,
        )

        assert options["outtmpl"] == str(tmp_path / "live.%(ext)s")
        assert options["concurrent_fragment_downloads"] == 5
        assert options["merge_output_format"] == "mp4"
        assert options["keepvideo"] is True
        assert options["postprocessors"] == [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ]

    def test_build_options_audio_only(self, tmp_path):
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        options = downloader._build_options(
            "live",
            extract_audio=True,
            keep_video=False,
        )

        assert options["format"] == "bestaudio/best"
        assert "merge_output_format" not in options
        assert options["keepvideo"] is False

    def test_build_options_accepts_custom_concurrency(self, tmp_path):
        downloader = YtDlpDownloader(download_dir=str(tmp_path), max_concurrent=8)

        options = downloader._build_options(
            "live",
            extract_audio=False,
            keep_video=True,
        )

        assert options["concurrent_fragment_downloads"] == 8

    def test_build_options_accepts_custom_format(self, tmp_path):
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        options = downloader._build_options(
            "live",
            extract_audio=False,
            keep_video=True,
            format_selector="bestvideo+bestaudio/best",
        )

        assert options["format"] == "bestvideo+bestaudio/best"

    def test_download_from_url_uses_ytdlp_api_without_network(self, tmp_path, monkeypatch):
        seen: dict = {}

        class FakeYoutubeDL:
            def __init__(self, options):
                seen["options"] = options

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download):
                seen["url"] = url
                seen["download"] = download
                (tmp_path / "live.mp4").write_bytes(b"video")
                (tmp_path / "live.mp3").write_bytes(b"audio")
                return {"id": "fake"}

        class FakeYtDlp:
            YoutubeDL = FakeYoutubeDL

        monkeypatch.setattr(YtDlpDownloader, "_import_ytdlp", staticmethod(lambda: FakeYtDlp))
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        outputs = downloader.download_from_url(
            "https://example.com/watch?v=1",
            output_name="live",
            extract_audio=True,
        )

        assert seen["url"] == "https://example.com/watch?v=1"
        assert seen["download"] is True
        assert seen["options"]["outtmpl"] == str(tmp_path / "live.%(ext)s")
        assert outputs == {
            "mp4": tmp_path / "live.mp4",
            "mp3": tmp_path / "live.mp3",
        }
