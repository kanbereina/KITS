"""downloader 模块单元测试：仅覆盖不触网的 yt-dlp 命令构造与输出解析。"""

from __future__ import annotations

import subprocess
import sys

import pytest

from kits.downloader import DEFAULT_CONCURRENT_FRAGMENTS, YtDlpDownloader


class TestYtDlpDownloader:
    def test_build_command_defaults_to_audio_extraction(self, tmp_path):
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        command = downloader._build_command("https://example.com/watch?v=1", "live")

        assert command[:3] == [sys.executable, "-m", "yt_dlp"]
        assert command[-1] == "https://example.com/watch?v=1"
        assert command[command.index("--paths") + 1] == str(tmp_path)
        assert command[command.index("--output") + 1] == "live.%(ext)s"
        assert command[command.index("--format") + 1] == "bestaudio/best"
        assert "--extract-audio" in command
        assert command[command.index("--audio-format") + 1] == "mp3"
        assert command[command.index("--concurrent-fragments") + 1] == str(DEFAULT_CONCURRENT_FRAGMENTS)

    def test_build_command_appends_passthrough_args_before_url(self, tmp_path):
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        command = downloader._build_command(
            "https://example.com/watch?v=1",
            "live",
            ["-f", "bestaudio", "--extract-audio"],
        )

        assert command[-4:] == ["-f", "bestaudio", "--extract-audio", "https://example.com/watch?v=1"]

    def test_rejects_invalid_concurrency(self, tmp_path):
        with pytest.raises(ValueError):
            YtDlpDownloader(download_dir=str(tmp_path), concurrent_fragments=0)

    def test_resolve_audio_path_prefers_printed_file(self, tmp_path):
        audio = tmp_path / "printed.mp3"
        audio.write_bytes(b"audio")
        fallback = tmp_path / "live.mp3"
        fallback.write_bytes(b"fallback")
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        path = downloader._resolve_audio_path("live", str(audio), newer_than=0)

        assert path == audio

    def test_download_from_url_uses_subprocess_without_network(self, tmp_path, monkeypatch):
        seen: dict = {}
        audio = tmp_path / "live.mp3"

        def fake_run(command, capture_output, text, encoding, errors, check):
            seen["command"] = command
            audio.write_bytes(b"audio")
            return subprocess.CompletedProcess(command, 0, stdout=str(audio), stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        outputs = downloader.download_from_url(
            "https://example.com/watch?v=1",
            output_name="live",
            extra_args=["--cookies", "cookies.txt"],
        )

        assert seen["command"][-3:] == ["--cookies", "cookies.txt", "https://example.com/watch?v=1"]
        assert outputs == {"audio": audio}

    def test_download_from_url_raises_on_ytdlp_failure(self, tmp_path, monkeypatch):
        def fake_run(command, capture_output, text, encoding, errors, check):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")

        monkeypatch.setattr(subprocess, "run", fake_run)
        downloader = YtDlpDownloader(download_dir=str(tmp_path))

        with pytest.raises(RuntimeError, match="yt-dlp 下载失败"):
            downloader.download_from_url("https://example.com/watch?v=1")
