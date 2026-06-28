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

"""yt-dlp 下载后端。

KITS 的下游流水线只需要音频文件，因此 download 子命令统一通过 yt-dlp 提取音频。
yt-dlp 本身负责 Twitch / YouTube / 直播回放等站点适配、分片下载与合并。
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["DEFAULT_CONCURRENT_FRAGMENTS", "YtDlpDownloader"]

DEFAULT_CONCURRENT_FRAGMENTS = 5


class YtDlpDownloader:
    """调用 yt-dlp CLI 的通用音频下载器。

    用子进程执行 `python -m yt_dlp`，避免在 KITS 内维护 yt-dlp 的参数映射。
    默认下载最佳音频并提取为 MP3；用户透传的 yt-dlp 原生参数会追加在默认参数后，
    因而可按 yt-dlp 自身规则覆盖默认 format / 音频格式等设置。
    """

    _AUDIO_SUFFIXES = {".mp3", ".m4a", ".opus", ".ogg", ".wav", ".flac", ".aac", ".webm"}
    _TEMP_SUFFIXES = {".part", ".temp", ".tmp", ".ytdl"}

    def __init__(self, download_dir: str = "downloads", concurrent_fragments: int = DEFAULT_CONCURRENT_FRAGMENTS):
        if concurrent_fragments < 1:
            raise ValueError("yt-dlp 分片并发数必须大于 0")
        self.download_dir = Path(download_dir)
        self.concurrent_fragments = concurrent_fragments
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _build_command(
        self,
        url: str,
        output_name: str = "output",
        extra_args: Sequence[str] | None = None,
    ) -> list[str]:
        """构造 yt-dlp 原生命令。独立出来便于单测。"""
        output_template = f"{output_name}.%(ext)s"
        return [
            sys.executable,
            "-m",
            "yt_dlp",
            "--paths",
            str(self.download_dir),
            "--output",
            output_template,
            "--format",
            "bestaudio/best",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "0",
            "--no-playlist",
            "--concurrent-fragments",
            str(self.concurrent_fragments),
            "--print",
            "after_move:filepath",
            *(extra_args or []),
            url,
        ]

    @staticmethod
    def _printed_paths(stdout: str) -> list[Path]:
        """从 yt-dlp --print 输出中提取已存在的文件路径。"""
        paths: list[Path] = []
        for line in stdout.splitlines():
            text = line.strip().strip('"')
            if not text:
                continue
            path = Path(text)
            if path.is_file():
                paths.append(path)
        return paths

    def _find_recent_audio(self, output_name: str, newer_than: float) -> Path | None:
        """兜底查找以 output_name 为 stem 的最新音频文件。"""
        candidates = [
            path
            for path in self.download_dir.glob(f"{output_name}.*")
            if path.is_file()
            and path.suffix.lower() in self._AUDIO_SUFFIXES
            and path.suffix.lower() not in self._TEMP_SUFFIXES
            and path.stat().st_mtime >= newer_than
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _resolve_audio_path(self, output_name: str, stdout: str, newer_than: float) -> Path:
        """解析 yt-dlp 最终音频产物路径。"""
        printed_audio = [
            path for path in self._printed_paths(stdout) if path.suffix.lower() in self._AUDIO_SUFFIXES
        ]
        if printed_audio:
            return printed_audio[-1]

        path = self._find_recent_audio(output_name, newer_than)
        if path is not None:
            return path

        raise RuntimeError("yt-dlp 已结束，但未找到音频产物。若透传参数改变了输出类型，请确认仍会生成音频文件")

    @staticmethod
    def _echo_output(lines: Iterable[str]) -> None:
        for line in lines:
            if line:
                print(line)

    def download_from_url(
        self,
        url: str,
        output_name: str = "output",
        *,
        extra_args: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        """用 yt-dlp 下载并提取音频，返回产物路径字典。"""
        print(f"\n🎬 yt-dlp 开始处理: {url}")
        print("=" * 60)

        cmd = self._build_command(url, output_name, extra_args)
        started_at = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self._echo_output(result.stdout.splitlines())
        self._echo_output(result.stderr.splitlines())
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp 下载失败，退出码 {result.returncode}")

        audio_path = self._resolve_audio_path(output_name, result.stdout, started_at)

        print("\n" + "=" * 60)
        print("🎉 yt-dlp 下载处理完成！")
        return {"audio": audio_path}
