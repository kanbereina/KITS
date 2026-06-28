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

"""语音活动检测（VAD）：用 Silero VAD 探出人声区间，反推出非语音间隙作为分段切点来源。

长音频分段转录要在「无人说话处」下刀，避免把语句拦腰截断。纯音量阈值
（旧版 ffmpeg silencedetect）在鹿乃长时间唱歌 / BGM 的场景失效——音量持续高、
探不到静音，只能退到硬上限强切。Silero VAD 能区分「人声 vs 音乐/噪音」，在唱歌段
也能找到真正的人声间隙，切点质量明显优于音量阈值，故用它取代 silencedetect。

设计要点：
  - 走 silero 默认的 jit/torch 后端（不装 onnxruntime extra），避开与
    audio-separator[gpu] 的 onnxruntime-gpu 共目录冲突（见 separator 模块注释）。
  - 模型文件随 silero-vad 包分发，加载不联网、无下载噪音。
  - 音频解码复用 ffmpeg（同 transcriber.slice_audio 路子），不引入 torchaudio 的读取。
  - 重依赖（torch + silero-vad）延迟导入，对齐 punctuator / separator 范式。

`speech_to_gaps` 是纯逻辑（取补集），不依赖 torch、可独立单测；`VADetector` 封装
模型加载与推理。产出的间隙与 transcriber.plan_segments 期望的「静音区间」同构，可直接喂入。
"""

from __future__ import annotations

import subprocess

__all__ = ["VADetector", "decode_pcm", "speech_to_gaps"]

# Silero VAD 在 16kHz 上训练，固定用此采样率
SAMPLING_RATE = 16000


def speech_to_gaps(
    speech: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """把语音区间取补集，得到 [0, duration] 内的非语音间隙 [(start, end), ...]（秒）。

    speech 须按起点升序、彼此不重叠（silero get_speech_timestamps 的输出即如此）。
    返回的间隙与 transcriber.plan_segments 期望的「静音区间」同构，可直接喂入分段规划：
    相邻语音之间的空档、以及开头 / 结尾的非语音段都算作可下刀的间隙。

    speech 为空（全程无人声，如纯 BGM）时返回整段 [(0, duration)] 一个大间隙，
    交由 plan_segments 自由切。区间起点容错排序，避免上游顺序异常导致补集错乱。
    """
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(speech):
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def decode_pcm(audio_file: str, sampling_rate: int = SAMPLING_RATE):
    """用 ffmpeg 把音频解码成单声道 float32 PCM，返回 1D torch.Tensor（值域约 [-1, 1]）。

    Silero VAD 要求 16kHz 单声道、归一化的一维浮点张量。ffmpeg 直接输出 f32le 裸流到
    stdout，再 frombuffer 成张量，省去临时 wav 文件与 torchaudio 读取依赖。
    """
    import torch  # 重依赖延迟导入

    cmd = [
        "ffmpeg", "-i", audio_file,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sampling_rate), "-vn",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace")[:300]
        raise RuntimeError(f"VAD 音频解码失败: {err}")
    # bytearray 让底层 buffer 可写，规避 torch.frombuffer 对只读 bytes 的告警
    return torch.frombuffer(bytearray(result.stdout), dtype=torch.float32)


class VADetector:
    """Silero VAD 语音活动探测器。延迟加载模型，复用实例可处理多个文件。

    threshold: 语音概率阈值（0~1），高于此算人声；越大越严格（保留更少人声、间隙更多）。
    min_silence: 短于此（秒）的停顿不切断人声、并入语音区间，对应 silero 的
        min_silence_duration_ms。语义同旧版 --min-silence（多短的停顿才算间隙）。
    device: 推理设备。None / "cpu" 走 CPU（轻量、稳妥、不抢转录显存）；CUDA 时可传 "cuda"
        加速（长音频解码出的张量较大，但模型极小）。MPS 兼容性不稳，调用方应回落 CPU。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence: float = 0.5,
        device: str | None = None,
    ):
        self.threshold = threshold
        self.min_silence = min_silence
        self.device = device
        self._model = None

    def load(self) -> None:
        """加载 silero VAD 模型（jit 后端，不碰 onnxruntime）。延迟导入 silero_vad。"""
        try:
            from silero_vad import load_silero_vad
        except ImportError as e:
            raise RuntimeError(
                "未安装 silero-vad，请先安装：uv add silero-vad"
            ) from e
        print("\n🗣️  加载 Silero VAD 模型...")
        self._model = load_silero_vad()  # onnx=False（默认），走 jit/torch
        if self.device and self.device != "cpu":
            self._model.to(self.device)
        print("✅ VAD 模型加载完成")

    def detect_gaps(self, audio_file: str, duration: float) -> list[tuple[float, float]]:
        """探出音频人声区间，返回其补集——非语音间隙 [(start, end), ...]（秒）。

        间隙与 transcriber.plan_segments 期望的「静音区间」同构，可直接喂入分段规划。
        duration 为音频总时长（秒，调用方已 probe，避免重复探测），用于补出末尾间隙。
        """
        from silero_vad import get_speech_timestamps

        if self._model is None:
            self.load()
        audio = decode_pcm(audio_file, SAMPLING_RATE)
        if self.device and self.device != "cpu":
            audio = audio.to(self.device)
        # return_seconds=True：元素为 {"start": s, "end": e}（秒）
        speech = get_speech_timestamps(
            audio,
            self._model,
            threshold=self.threshold,
            sampling_rate=SAMPLING_RATE,
            min_silence_duration_ms=int(self.min_silence * 1000),
            return_seconds=True,
        )
        intervals = [(float(d["start"]), float(d["end"])) for d in speech]
        return speech_to_gaps(intervals, duration)
