# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

KITS 是基于 Whisper 的鹿乃 Twitch 直播工具：下载 Twitch 直播分片、合并 MP4 / 提取 MP3，把音频转成带时间戳的 SRT 字幕，可调用 DeepSeek 把日语字幕翻译成中文、对字幕做 AI 总结，并支持用 audio-separator 分离人声。沟通与代码注释一律用中文；日语仅是转录处理的目标内容。

## 常用命令

依赖与运行统一走 `uv`（包管理 + 运行器）：

```bash
uv sync                                   # 同步依赖（含 dev 组）
uv run kits download "<ts_url>" -o name   # 下载直播
uv run kits subtitle -i audio.mp3         # 音频转 SRT
uv run kits translate -i live.srt         # 日语 SRT 译中文（需 DEEPSEEK_API_KEY）
uv run kits separate -i audio.mp3         # 分离人声（需 audio-separator + GPU）
uv run kits sum -i live.srt               # AI 总结 SRT（需 DEEPSEEK_API_KEY）
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
- `translate` / `sum` 需要 DeepSeek API Key，走 `--api-key` 或环境变量 `DEEPSEEK_API_KEY`。
- `separate` 依赖 `audio-separator[gpu]`（含 onnxruntime-gpu），首次运行会下载分离模型；同样要 CUDA GPU。

## 架构

模块按依赖方向严格分层，核心设计意图是把**纯逻辑**与**重依赖（torch/网络）**解耦：

- `subtitle.py` — 纯函数库，**不依赖 torch/网络**，可独立单测。负责单词级时间戳 → 完整句子的断句、SRT 渲染、SRT 解析（`parse_srt` / `srt_time_to_seconds`），以及增量写入器 `SrtWriter`（分段转录时序号跨段连续、逐段 flush 落盘）。`Word` / `Sentence` 是贯穿全项目的 TypedDict 数据契约。
- `filters.py` — 纯函数库，剔除游戏内系统播报 / 技能语音（整条完全匹配）。
- `deepseek.py` — **公共 DeepSeek 客户端**，仅依赖 httpx。封装 API Key 解析（参数优先于 `DEEPSEEK_API_KEY`）、`chat()` 单次请求、错误处理（`DeepSeekError`）。`translator` 与 `summarizer` 共用，批处理 / map-reduce 等领域策略留各自模块。
- `transcriber.py` — 封装 Whisper pipeline 的加载与转录。`transcribe()` 整段转；`transcribe_segmented()` 按静音切分长音频后分段流式产出（生成器）。依赖 torch/transformers，静音探测与切分依赖 ffmpeg/ffprobe。
- `downloader.py` — `TwitchDownloader`，异步下载 TS 分片 → ffmpeg 合并 MP4 → 可选提取 MP3。仅依赖 httpx + ffmpeg，**不依赖 torch**。
- `translator.py` — `DeepSeekTranslator`，经 `deepseek.DeepSeekClient` 把日语句子逐批译成中文。`TranslationError` 继承 `DeepSeekError`（保留历史契约）。
- `separator.py` — `VocalSeparator`，封装 audio-separator 分离人声（默认只出 Vocals 轨）。重依赖 audio-separator（含 torch/onnxruntime），**延迟导入**（`load()` 内才 import）。
- `summarizer.py` — `Summarizer`，经 `deepseek.DeepSeekClient` 对 SRT 做 AI 总结。提示词走 JSON 预设（包内 `prompts.json` + 用户 `--prompt-file` 覆盖），长字幕 map-reduce 分块。纯逻辑（`load_presets` / `resolve_preset` / `format_sentences` / `chunk_sentences`）不触网、可单测。
- `prompts.json` — 内置总结提示词预设（timeline / summary / highlights / setlist）+ `reduce_system` 合并提示词，随包发布。
- `cli.py` — argparse 子命令 `download` / `subtitle` / `translate` / `separate` / `sum`，把各模块串成流水线。

数据流：

- 转字幕：`downloader`(MP3) →（可选 `separator` 分离人声）→ `transcriber.transcribe_segmented()`（按静音分段，逐段产出 list[Word]）→ 每段 `subtitle.segment_sentences()`(list[Sentence]) → `subtitle.SrtWriter.append()` 增量写盘。`_audio_to_srt` 是这条流水线，`_maybe_separate` 在转录前按 `--separate` 决定是否预处理。
- 译字幕：SRT 文件 → `subtitle.parse_srt()`(list[Sentence]) → `translator.translate()`(中文 list[Sentence]) → `subtitle.write_srt()`。
- 分离人声：音频 → `separator.VocalSeparator.separate()` → 人声音频文件（可再交给 subtitle）。
- 总结：SRT 文件 → `subtitle.parse_srt()` → `summarizer.format_sentences/chunk_sentences` → `summarizer.Summarizer.summarize()`（按预设逐块总结，多块再 reduce 合并）→ Markdown 文件。

关键约定：

- `cli.py` 中对 `transcriber` / `downloader` / `translator` / `separator` / `summarizer` 用**延迟导入**（函数内 import），避免无谓加载 GPU 栈或网络栈。新增重依赖模块时沿用此模式。
- DeepSeek 调用统一走 `deepseek.DeepSeekClient`，不要在 `translator` / `summarizer` 里重复写 HTTP / 鉴权。新增 DeepSeek 能力时复用该客户端，领域逻辑（分批、提示词）留各自模块。
- `download --srt` 隐含 `--mp3`（生成字幕必须先有音频），逻辑在 `_run_download` 的 `need_mp3 = args.mp3 or args.srt`。
- 断句与分段参数（`--max-gap` / `--target-chunk` 等）由 `_add_subtitle_args` 在 download / subtitle 间共用。
- TS URL 里的分片编号**仅用于定位 base_url**，下载范围默认从 0 开始、用指数探测+二分定位结尾（`detect_end_number`）。
- 长音频分段：`transcribe_segmented` 用 ffmpeg `silencedetect` 探测静音，`plan_segments` 贪心在静音中点切（段内无静音则在 `max_chunk` 硬上限强切兜底），切点落在无人说话处故不打断语句、精度不变。短音频（≤ `max_chunk`）退化为整段转录。分段时词级时间戳要 `_shift_words` 加回段起始偏移对齐全局时间轴。
- `translator` / `summarizer` 按批 / 分块发送，DeepSeek 调用经公共客户端。翻译用「序号|||文本」格式逐条对应、按序号回填防错位；总结用 map-reduce（逐块总结后 reduce 合并），提示词来自 `prompts.json` 预设。翻译只改文本、时间戳原样保留。

## 扩展点

- **新增总结预设**：直接在 `src/kits/prompts.json` 的 `presets` 里加一项（含 `description` + `system`），或让用户用 `--prompt-file` 传外部 JSON 覆盖，无需改代码。
- **新增游戏播报过滤**：在 `filters.py` 的 `GAME_CALLOUTS` / `_GAME_ALIASES` 登记词表与别名。
- **新增 DeepSeek 能力**：复用 `deepseek.DeepSeekClient`，在 `cli.py` 加子命令（沿用延迟导入，参考 `translate` / `sum`）。
- **换人声分离模型**：`separate --model <文件名>` 或 `subtitle --separate --separate-model <文件名>`，模型名走 audio-separator 的模型库。
