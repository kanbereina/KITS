# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

KITS 是基于 `openai/whisper-large-v3-turbo` 的鹿乃 Twitch 直播工具：下载 Twitch 直播分片、合并 MP4 / 提取 MP3，并把音频转成带时间戳的 SRT 字幕。沟通与代码注释一律用中文；日语仅是转录处理的目标内容。

## 常用命令

依赖与运行统一走 `uv`（包管理 + 运行器）：

```bash
uv sync                                   # 同步依赖（含 dev 组）
uv run kits download "<ts_url>" -o name   # 下载直播
uv run kits subtitle -i audio.mp3         # 音频转 SRT
uv run python main.py subtitle -i ...     # 等价入口（main.py 为薄入口）

uv run ruff check .                       # lint
uv run ruff check --fix .                 # lint 并自动修复
uv run pytest                             # 跑测试（目前尚无测试文件）
uv run pytest tests/test_subtitle.py::test_name  # 跑单个测试
```

注意：

- PyTorch 从自定义索引 `pytorch-cu128`（CUDA 12.8）安装，见 `pyproject.toml` 的 `[tool.uv.sources]`。不要把 torch 换成 PyPI 默认源。
- 转录（`subtitle` / `download --srt`）强制要求可用的 CUDA GPU，CPU 环境会直接抛错。
- 合并 MP4、提取 MP3 依赖系统 `ffmpeg`，须在 PATH 中。

## 架构

四个模块按依赖方向严格分层，核心设计意图是把**纯逻辑**与**重依赖（torch/网络）**解耦：

- `subtitle.py` — 纯函数库，**不依赖 torch/网络**，可独立单测。负责单词级时间戳 → 完整句子的断句、SRT 渲染。`Word` / `Sentence` 是贯穿全项目的 TypedDict 数据契约。
- `transcriber.py` — 封装 Whisper pipeline 的加载与转录，产出 `list[Word]`（单词级时间戳）。依赖 torch/transformers。
- `downloader.py` — `TwitchDownloader`，异步下载 TS 分片 → ffmpeg 合并 MP4 → 可选提取 MP3。仅依赖 httpx + ffmpeg，**不依赖 torch**。
- `cli.py` — argparse 子命令 `download` / `subtitle`，把三者串成流水线。

数据流：`downloader`(MP3) → `transcriber.transcribe()`(list[Word]) → `subtitle.segment_sentences()`(list[Sentence]) → `subtitle.write_srt()`。

关键约定：

- `cli.py` 中对 `transcriber` / `downloader` 用**延迟导入**（函数内 import），避免仅下载时也加载 GPU 栈。新增重依赖模块时沿用此模式。
- `download --srt` 隐含 `--mp3`（生成字幕必须先有音频），逻辑在 `_run_download` 的 `need_mp3 = args.mp3 or args.srt`。
- 断句参数（`--max-gap` 等）由 `_add_subtitle_args` 在两个子命令间共用。
- TS URL 里的分片编号**仅用于定位 base_url**，下载范围默认从 0 开始、用指数探测+二分定位结尾（`detect_end_number`）。

## 扩展点

按 README 规划，接入 DeepSeek 总结时：新增 `src/kits/summarizer.py`，消费 `transcriber` 的文本或 `subtitle` 的句子，并在 `cli.py` 加 `summarize` 子命令（沿用延迟导入）。
