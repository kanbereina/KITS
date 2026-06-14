# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

KITS 是基于 `openai/whisper-large-v3-turbo` 的鹿乃 Twitch 直播工具：下载 Twitch 直播分片、合并 MP4 / 提取 MP3，把音频转成带时间戳的 SRT 字幕，并可调用 DeepSeek 把日语字幕翻译成中文。沟通与代码注释一律用中文；日语仅是转录处理的目标内容。

## 常用命令

依赖与运行统一走 `uv`（包管理 + 运行器）：

```bash
uv sync                                   # 同步依赖（含 dev 组）
uv run kits download "<ts_url>" -o name   # 下载直播
uv run kits subtitle -i audio.mp3         # 音频转 SRT
uv run kits translate -i live.srt         # 日语 SRT 译中文（需 DEEPSEEK_API_KEY）
uv run python main.py subtitle -i ...     # 等价入口（main.py 为薄入口）

uv run ruff check .                       # lint
uv run ruff check --fix .                 # lint 并自动修复
uv run pytest                             # 跑测试
uv run pytest tests/test_subtitle.py::TestParseSrt  # 跑单个测试类
```

注意：

- PyTorch 从自定义索引 `pytorch-cu128`（CUDA 12.8）安装，见 `pyproject.toml` 的 `[tool.uv.sources]`。不要把 torch 换成 PyPI 默认源。
- 转录（`subtitle` / `download --srt`）强制要求可用的 CUDA GPU，CPU 环境会直接抛错。
- 合并 MP4、提取 MP3 依赖系统 `ffmpeg`，须在 PATH 中。
- `translate` 需要 DeepSeek API Key，走 `--api-key` 或环境变量 `DEEPSEEK_API_KEY`。

## 架构

模块按依赖方向严格分层，核心设计意图是把**纯逻辑**与**重依赖（torch/网络）**解耦：

- `subtitle.py` — 纯函数库，**不依赖 torch/网络**，可独立单测。负责单词级时间戳 → 完整句子的断句、SRT 渲染、SRT 解析（`parse_srt` / `srt_time_to_seconds`），以及增量写入器 `SrtWriter`（分段转录时序号跨段连续、逐段 flush 落盘）。`Word` / `Sentence` 是贯穿全项目的 TypedDict 数据契约。
- `transcriber.py` — 封装 Whisper pipeline 的加载与转录。`transcribe()` 整段转；`transcribe_segmented()` 按静音切分长音频后分段流式产出（生成器）。依赖 torch/transformers，静音探测与切分依赖 ffmpeg/ffprobe。
- `downloader.py` — `TwitchDownloader`，异步下载 TS 分片 → ffmpeg 合并 MP4 → 可选提取 MP3。仅依赖 httpx + ffmpeg，**不依赖 torch**。
- `translator.py` — `DeepSeekTranslator`，调用 DeepSeek chat API 把日语句子逐批译成中文。仅依赖 httpx，**不依赖 torch**。
- `cli.py` — argparse 子命令 `download` / `subtitle` / `translate`，把各模块串成流水线。

数据流：

- 转字幕：`downloader`(MP3) → `transcriber.transcribe_segmented()`（按静音分段，逐段产出 list[Word]）→ 每段 `subtitle.segment_sentences()`(list[Sentence]) → `subtitle.SrtWriter.append()` 增量写盘。`_audio_to_srt` 是这条流水线，边转边写、显示分段进度。
- 译字幕：SRT 文件 → `subtitle.parse_srt()`(list[Sentence]) → `translator.translate()`(中文 list[Sentence]) → `subtitle.write_srt()`。

关键约定：

- `cli.py` 中对 `transcriber` / `downloader` / `translator` 用**延迟导入**（函数内 import），避免无谓加载 GPU 栈或网络栈。新增重依赖模块时沿用此模式。
- `download --srt` 隐含 `--mp3`（生成字幕必须先有音频），逻辑在 `_run_download` 的 `need_mp3 = args.mp3 or args.srt`。
- 断句与分段参数（`--max-gap` / `--target-chunk` 等）由 `_add_subtitle_args` 在 download / subtitle 间共用。
- TS URL 里的分片编号**仅用于定位 base_url**，下载范围默认从 0 开始、用指数探测+二分定位结尾（`detect_end_number`）。
- 长音频分段：`transcribe_segmented` 用 ffmpeg `silencedetect` 探测静音，`plan_segments` 贪心在静音中点切（段内无静音则在 `max_chunk` 硬上限强切兜底），切点落在无人说话处故不打断语句、精度不变。短音频（≤ `max_chunk`）退化为整段转录。分段时词级时间戳要 `_shift_words` 加回段起始偏移对齐全局时间轴。
- `translator` 按批发送字幕，用「序号|||文本」格式让模型逐条对应，按序号回填以防时间轴错位；译文缺失时回退保留原文。API Key 走 `--api-key` 或 `DEEPSEEK_API_KEY`。翻译只改文本、时间戳原样保留。

## 扩展点

按 README 规划，接入 DeepSeek 总结时：新增 `src/kits/summarizer.py`，消费 `transcriber` 的文本或 `subtitle` 的句子，并在 `cli.py` 加 `summarize` 子命令（沿用延迟导入，可参考已有的 `translator` + `translate` 子命令）。
