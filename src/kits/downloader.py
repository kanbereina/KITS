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

"""Twitch 直播下载：异步下载 TS 分片 -> 合并 MP4 -> 可选提取 MP3。

不依赖 torch / transformers。产出的 MP3 可交给 kits.transcriber 转字幕。
合并与音频提取依赖系统已安装的 ffmpeg。
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from pathlib import Path

import httpx

__all__ = ["TwitchDownloader", "parse_url_pattern"]

# 从形如 https://.../chunked/1710.ts 或 https://.../160p30/3.ts 的 URL 中
# 拆出 (基础URL, 编号, 扩展名)。分片目录名随画质而变（chunked / 160p30 / 720p60 等），
# 故只认「末段目录 + 纯数字文件名 + .ts」，不写死目录名。
_URL_PATTERN = re.compile(r"(.+/)(\d+)(\.ts)$")

_HEADERS = {
    "referer": "https://www.twitch.tv/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36 Edg/147.0.0.0"
    ),
}


def parse_url_pattern(url: str) -> tuple[str, int, str]:
    """解析 TS URL，返回 (基础URL, 起始编号, 扩展名)。"""
    match = _URL_PATTERN.match(url)
    if not match:
        raise ValueError(f"无法解析 URL 格式: {url}")
    return match.group(1), int(match.group(2)), match.group(3)


def _check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用。"""
    # noinspection PyDeprecation
    if shutil.which("ffmpeg") is None:
        return False
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
    return result.returncode == 0


class TwitchDownloader:
    """Twitch TS 下载器。可编程 API，无交互式输入。"""

    def __init__(self, download_dir: str = "downloads", max_concurrent: int = 5):
        self.download_dir = Path(download_dir)
        self.ts_dir = self.download_dir / "ts_files"
        self.max_concurrent = max_concurrent
        self.ts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    async def _exists(client: httpx.AsyncClient, url: str) -> bool:
        try:
            resp = await client.head(url, timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def detect_end_number(
        self, base_url: str, extension: str, start: int = 0
    ) -> int:
        """指数探测 + 二分查找最后一个存在的分片编号。

        请求数约 2·log2(N)，远快于逐个线性探测。
        start 为已知存在的下界（默认 0）。探测点示例:
        0 -> 1 -> 2 -> 4 -> ... -> 1024 -> 2048(✗)，再在 (1024, 2048) 二分。
        """
        print("🔍 正在检测视频长度（指数探测 + 二分）...")
        async with httpx.AsyncClient(headers=_HEADERS) as client:
            if not await self._exists(client, f"{base_url}{start}{extension}"):
                raise RuntimeError(
                    f"起始分片 {start}.ts 不存在，请用 --start 指定真实起始编号"
                )

            # 1) 指数探测上界：步长翻倍，直到某编号不存在
            lo = start  # 已知存在
            step = 1
            hi = start + step
            while await self._exists(client, f"{base_url}{hi}{extension}"):
                print(f"  ✓ {hi}.ts 存在，继续向后探测...", end="\r")
                lo = hi
                step *= 2
                hi = start + step
            print(f"\n  📈 上界落在 ({lo}, {hi}) 区间，开始二分...")

            # 2) 二分查找：lo 存在、hi 不存在，收敛到最后一个存在的编号
            left, right = lo, hi
            while left + 1 < right:
                mid = (left + right) // 2
                if await self._exists(client, f"{base_url}{mid}{extension}"):
                    left = mid
                    print(f"  ✓ {mid}.ts 存在", end="\r")
                else:
                    right = mid
                    print(f"  ✗ {mid}.ts 不存在", end="\r")
            last = left

        total = last - start + 1
        print(f"\n✅ 检测完成: 从 {start} 到 {last}，共 {total} 个文件")
        return last

    async def _download_one(
        self, client: httpx.AsyncClient, url: str, index: int, retry: int = 3
    ) -> Path | None:
        for attempt in range(retry):
            try:
                resp = await client.get(url, timeout=30.0)
                if resp.status_code == 200 and resp.content:
                    path = self.ts_dir / f"segment_{index:05d}.ts"
                    path.write_bytes(resp.content)
                    print(f"✓ [{index}] 下载成功 ({len(resp.content) / 1024:.1f} KB)")
                    return path
                if resp.status_code == 404:
                    return None
                print(f"⚠ [{index}] HTTP {resp.status_code}, 重试 {attempt + 1}/{retry}")
            except httpx.HTTPError as e:
                print(f"⚠ [{index}] 下载失败: {str(e)[:50]}, 重试 {attempt + 1}/{retry}")
            if attempt < retry - 1:
                await asyncio.sleep(1)
        return None

    async def download_range(
        self, base_url: str, start: int, end: int, extension: str = ".ts"
    ) -> list[Path]:
        """并发下载 [start, end] 范围内的 TS 分片，返回成功的文件路径（已排序）。"""
        total = end - start + 1
        print(f"\n🚀 开始下载 {total} 个 TS 文件 (并发数: {self.max_concurrent})...")
        semaphore = asyncio.Semaphore(self.max_concurrent)
        downloaded: list[Path] = []

        async def _limited(_client: httpx.AsyncClient, num: int) -> Path | None:
            async with semaphore:
                url = f"{base_url}{num}{extension}"
                return await self._download_one(_client, url, num - start)

        async with httpx.AsyncClient(headers=_HEADERS, timeout=30.0) as client:
            tasks = [_limited(client, num) for num in range(start, end + 1)]
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                if (result := await coro) is not None:
                    downloaded.append(result)
                print(f"📊 总进度: {i}/{total} ({i / total * 100:.1f}%)", end="\r")

        print(f"\n\n✅ 下载完成: 成功 {len(downloaded)}/{total} 个文件")
        if len(downloaded) < total:
            print(f"⚠️ 警告: {total - len(downloaded)} 个文件下载失败")
        return sorted(downloaded)

    def cleanup_ts_files(self) -> None:
        """删除临时 TS 文件目录。"""
        if self.ts_dir.exists():
            shutil.rmtree(self.ts_dir, ignore_errors=True)
            print("🧹 已清理临时 TS 文件")

    def merge_to_mp4(self, ts_files: list[Path], output_name: str = "output.mp4") -> Path:
        """用 ffmpeg 把 TS 分片合并为 MP4，返回输出路径。"""
        if not ts_files:
            raise RuntimeError("没有 TS 文件可以合并")
        if not _check_ffmpeg():
            raise RuntimeError(
                "未找到 ffmpeg，请先安装：https://ffmpeg.org/download.html"
            )

        list_file = self.download_dir / "file_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for ts in ts_files:
                f.write(f"file '{ts.absolute().as_posix()}'\n")

        if not output_name.endswith(".mp4"):
            output_name += ".mp4"
        output_path = self.download_dir / output_name

        print(f"\n🔄 开始合并 {len(ts_files)} 个文件为 MP4...")
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy", "-bsf:a", "aac_adtstoasc",
            str(output_path), "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        list_file.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"合并失败: {result.stderr[:500]}")

        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✅ 成功合并: {output_path} ({size_mb:.2f} MB)")
        return output_path

    def extract_mp3(self, video_path: Path, output_name: str | None = None) -> Path:
        """用 ffmpeg 从视频提取 MP3 音轨，返回 MP3 路径。"""
        if not _check_ffmpeg():
            raise RuntimeError(
                "未找到 ffmpeg，请先安装：https://ffmpeg.org/download.html"
            )
        mp3_path = (
            video_path.with_suffix(".mp3")
            if output_name is None
            else self.download_dir / output_name
        )

        print(f"\n🎵 正在提取 MP3: {mp3_path.name}")
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            str(mp3_path), "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"提取 MP3 失败: {result.stderr[:500]}")
        print(f"✅ MP3 已保存: {mp3_path}")
        return mp3_path

    async def download_from_url(
        self,
        url: str,
        output_name: str = "output",
        start_num: int | None = None,
        end_num: int | None = None,
        keep_ts: bool = False,
        extract_audio: bool = False,
    ) -> dict[str, Path]:
        """从 TS URL 一键下载并合并，返回产物路径字典。

        keys: "mp4"，extract_audio 为真时含 "mp3"。
        """
        print(f"\n🎬 开始处理: {url}")
        print("=" * 60)

        # URL 里的编号仅用于定位 base_url，默认从 0 开始下载整场直播。
        # 只有显式传入 --start 才作为起点，配合 --end 可下载指定范围分段。
        base_url, _url_num, extension = parse_url_pattern(url)
        start = 0 if start_num is None else start_num
        if end_num is None:
            end_num = await self.detect_end_number(base_url, extension, start=start)
        if start > end_num:
            raise ValueError(f"起始编号 {start} 大于结束编号 {end_num}")

        ts_files = await self.download_range(base_url, start, end_num, extension)
        if not ts_files:
            raise RuntimeError("未能下载任何 TS 文件")

        # noinspection PyDictCreation
        outputs: dict[str, Path] = {}
        outputs["mp4"] = self.merge_to_mp4(ts_files, output_name)
        if extract_audio:
            outputs["mp3"] = self.extract_mp3(outputs["mp4"])
        if not keep_ts:
            self.cleanup_ts_files()

        print("\n" + "=" * 60)
        print("🎉 下载处理完成！")
        return outputs
