# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

KITS 是基于 Whisper 的鹿乃 Twitch 直播工具：下载 Twitch 直播分片、合并 MP4 / 提取 MP3，把音频转成带时间戳的 SRT 字幕，可调用 DeepSeek 把日语字幕翻译成中文、对字幕做 AI 总结，并支持用 audio-separator 分离人声。沟通与代码注释一律用中文；日语仅是转录处理的目标内容。

## 常用命令

依赖与运行统一走 `uv`（包管理 + 运行器）：

```bash
uv sync                                   # 同步依赖（含 dev 组）
uv run kits download "<ts_url>" -o name   # 下载直播（别名 dl）
uv run kits subtitle -i audio.mp3         # 音频转 SRT（别名 srt）
uv run kits translate -i live.srt         # 日语 SRT 译中文（别名 tr；需 DEEPSEEK_API_KEY）
uv run kits separate -i audio.mp3         # 分离人声（别名 sep；需 audio-separator + GPU）
uv run kits summarize -i live.srt         # AI 总结 SRT（别名 sum；需 DEEPSEEK_API_KEY）
uv run python main.py subtitle -i ...     # 等价入口（main.py 为薄入口）

uv run ruff check .                       # lint
uv run ruff check --fix .                 # lint 并自动修复
uv run pytest                             # 跑测试
uv run pytest tests/test_subtitle.py::TestParseSrt  # 跑单个测试类
```

注意：

- PyTorch 从自定义索引 `pytorch-cu128`（CUDA 12.8）安装，见 `pyproject.toml` 的 `[tool.uv.sources]`。不要把 torch 换成 PyPI 默认源。
- 转录（`subtitle` / `download --srt`）强制要求 GPU 加速：CUDA(Nvidia) 或 MPS(Apple Silicon)，纯 CPU 环境会直接抛错（设备选择见 `transcriber.select_device()`，优先 CUDA、其次 MPS）。
- **依赖按平台分流**（见 `pyproject.toml`）：Linux/Win 走 CUDA 栈（torch `pytorch-cu128` + `onnxruntime-gpu` + `audio-separator[gpu]`）；macOS(Apple Silicon) 无 CUDA 轮子，自动落到 PyPI 默认源 torch（自带 MPS）+ `onnxruntime`(CPU/CoreML) + `audio-separator[cpu]`。关键三处：torch 源用 `marker = "sys_platform != 'darwin'"` 仅对非 darwin 生效、`pytorch-cu128` 索引置 `explicit = true`（否则 torch 被绑死该索引、mac 不回落 PyPI）、`required-environments` 声明 darwin-arm64（强制 uv 为 mac 单独求解出有 arm64 轮子的 torch 版本，否则通用解析会把全平台锁到 `+cu128` 本地版导致 mac sync 失败）。`override-dependencies` 的 onnxruntime 禁装 marker 改为 `sys_platform == 'darwin'`：mac 上*需要* CPU 版、非 darwin 才禁（防 CPU 版顶掉 GPU 版 dll）。改依赖后 `uv lock` 会按平台分出两份 torch stanza。
- 默认模型 `kotoba-tech/kotoba-whisper-v2.2`（日语识别更准的蒸馏模型）。它只产 chunk 级时间戳、不带句末标点，故转录后默认走 `punctuator` 标点恢复再断句（`--no-punctuate` 关闭）。换用本身带 word 级时间戳 + 标点的模型时可关。
- 合并 MP4、提取 MP3 依赖系统 `ffmpeg`，须在 PATH 中。
- `translate` / `sum` 需要 LLM API Key，走 `--api-key` 或环境变量 `KITS_LLM_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`。默认走 DeepSeek，可用 `--base-url`（或 `KITS_LLM_BASE_URL`）接入其他 OpenAI 兼容端点（OpenAI / Ollama / vLLM 等）；自定义端点允许空 Key。
- `summarize --render-image` 需要可选 extra：先 `uv sync --extra image`，再 `uv run --extra image playwright install chromium`。实现是 Markdown → HTML → Chromium 截图；Markdown 解析与 Playwright 都在图片渲染路径内按需使用。
- `separate` 依赖 `audio-separator[gpu]`（含 onnxruntime-gpu），首次运行会下载分离模型；同样要 CUDA GPU。
- **onnxruntime GPU 加速的两个坑**（`.onnx`/MDX 模型走 onnxruntime，`.ckpt`/MDXC roformer 走 torch）：
  1. `punctuators` 间接依赖 **CPU 版 `onnxruntime`**，会和 `audio-separator[gpu]` 的 `onnxruntime-gpu` 装进同一个 `onnxruntime/` 目录、CPU 版 dll 顶掉 GPU 版，导致 `CUDAExecutionProvider` 丢失、`.onnx` 分离静默退回 CPU（慢数倍）。靠 `pyproject.toml` 的 `[tool.uv] override-dependencies = ["onnxruntime ; sys_platform == 'never'"]` 禁止 CPU 版被装入。改完需 `uv sync --reinstall-package onnxruntime-gpu` 恢复被覆盖的 GPU dll。
  2. onnxruntime-gpu 需要 `cublasLt64_12.dll` / `cudnn64_9.dll` 等 CUDA12/cuDNN9 运行时；本机不装独立 CUDA Toolkit，靠 `separator._expose_torch_cuda_dlls()` 在 import onnxruntime 前把 `torch/lib`（torch cu128 自带这些 dll）加进 DLL 搜索路径复用。
- **Windows GBK 终端**下输出 emoji（🎚️ 等）会 `UnicodeEncodeError` 崩溃，`cli.main()` 开头把 stdout/stderr `reconfigure` 成 UTF-8 兜底。新增带 emoji 的 print 不必担心，但别在 main 之外的早期路径打 emoji。
- 人声分离加速参数（`separate` / `subtitle --separate` 共用，后者加 `--separate-` 前缀）：`--segment-size`（默认 512，越大越快越吃显存）、`--overlap`（MDX 用 0~1 小数，默认 0.1；MDXC roformer 是整数步数，语义不同，代码里分开传）、`--segment-minutes`、`--output-bitrate`。注意 MDX 路径里 `batch_size` 基本无效（每次循环只 1 个 chunk），真正提速靠 segment_size/overlap。

## 架构

模块按依赖方向严格分层，核心设计意图是把**纯逻辑**与**重依赖（torch/网络）**解耦：

- `subtitle.py` — 纯函数库，**不依赖 torch/网络**，可独立单测。负责单词级时间戳 → 完整句子的断句、SRT 渲染、SRT 解析（`parse_srt` / `srt_time_to_seconds`），以及增量写入器 `SrtWriter`（分段转录时序号跨段连续、逐段 flush 落盘）。`Word` / `Sentence` 是贯穿全项目的 TypedDict 数据契约。**VAD 修正漂移**：`segment_sentences` 收可选 `speech`（VAD 人声区间，全局时间轴），`_flush` 对明显虚高的句（时长/字符 > `vad_ratio_threshold`，默认 1.0）**优先**用 `_vad_voice_end`（bisect 找 start 后第一个人声区间 end）把 kotoba 虚高的 chunk end 夹回真实人声边界；VAD 对不上（start 在所有人声后、或夹出的 end 不落在 (start,end) 内）才**回退** `max_seconds_per_char` 字符速率收缩。二者单点决策、互斥。`speech=None`（短音频/未传）时行为与引入 VAD 前完全一致。实测 VAD 边界比字符速率准（偏差 0.1~0.3s vs 2.5~3.5s）、能修约 82% 虚高条目。
- `filters.py` — 纯函数库，剔除游戏内系统播报 / 技能语音（整条完全匹配）。
- `llm.py` — **公共 OpenAI 兼容 LLM 客户端** `LLMClient`，仅依赖 httpx。封装 base_url 解析（参数 > `KITS_LLM_BASE_URL` > 默认 DeepSeek，自动补 `/chat/completions`）、API Key 解析（参数 > `KITS_LLM_API_KEY` > `OPENAI_API_KEY` > `DEEPSEEK_API_KEY`）、`chat()` 单次请求、错误处理（`LLMError`）。空 Key 策略：默认 DeepSeek 端点强制要 Key，自定义端点允许空 Key（本地 Ollama 等）。`translator` 与 `summarizer` 共用，批处理 / map-reduce 等领域策略留各自模块。
- `transcriber.py` — 封装 Whisper pipeline 的加载与转录。`transcribe()` 整段转；长音频有**两种分段模式**（都在转录进度条创建前先跑规划阶段，VAD 加载/扫描日志才不冲乱进度条）：
  - **默认：逐 VAD 人声窗口**。`plan_windows()` 用调好的 silero 原生参数（`VAD_WINDOW_*`：threshold=0.4、min_silence=1.5s、max_speech=18s、min_speech=0.25s）直接探出「话语级人声窗口」，返回 `(duration, windows)`；`transcribe_windows()` 逐窗口转、以窗口起点做偏移锚点、**丢弃窗口间静音**。锚定真实人声边界、根治前导静音漂移（kotoba 把前导静音里的首句标成 0:00）。**代价：silero 对唱歌延音系统性漏判**（持续长元音不判为语音）→ 那段不建窗口、字幕整段丢失（调低 threshold 救不回，已诊断坐实）。窗口间被静音隔开、互不相邻，故不需要 overlap 接缝去重。
  - **`--full-transcribe`：整段连续转**。`plan_audio()` 探时长 + VAD 探人声间隙 + `plan_segments` 在 `(target_chunk, max_chunk)` 窗口内挑「时长最长的非语音间隙中点」切含静音大段（返回 `(duration, segments, speech)`），`transcribe_segments()` 拿预规划的段逐段流式产出。不靠 VAD 决定转不转，**唱歌延音也能识别、不丢字幕**；代价是前导静音可能让首句时间戳偏早。每段取数窗口两侧 pad `--segment-overlap` 垫料、转录后 `_keep_core_words` 按词中心裁回去重。
  - 共性：依赖 torch/transformers，VAD 走 `kits.vad`、切片走 ffmpeg/ffprobe。**时间戳粒度**用 `return_timestamps=True`（chunk/短语级），不用 `"word"`——kotoba 蒸馏模型解码器仅 2 层，但 alignment_heads 继承自 large-v3（引用第 25 层），抽词级时间戳会 `IndexError`。chunk 结构 `{"text", "timestamp": (start, end)}` 兼容 `Word` 契约。
- `vad.py` — `VADetector`，用 **Silero VAD** 探出人声区间（`detect_speech`）。人声区间两用：① `speech_to_gaps` 取补集得非语音间隙，喂 `transcriber.plan_segments` 作分段切点（取代旧版 ffmpeg silencedetect 纯音量阈值——VAD 能区分「人声 vs 音乐/噪音」，唱歌 / BGM 段也找得到真正人声间隙）；② 直接作真实人声边界，透传给 `subtitle._flush` 把 kotoba 虚高的 chunk end 夹回真实结束处（见下「VAD 修正漂移」）。`speech_to_gaps` 纯逻辑可单测。**走 silero 默认 jit 后端（不装 onnxruntime extra）**，避开与 `audio-separator[gpu]` 的 onnxruntime-gpu 共目录冲突。音频解码复用 ffmpeg（`decode_pcm` 出 16k 单声道 f32le 裸流 → `torch.frombuffer`，不引入 torchaudio 读取）。重依赖（torch + silero-vad）**延迟导入**。**VAD 固定走 CPU**：silero VAD 是 LSTM、隐状态时间步串行依赖（窗口 N 必须等 N-1 的 hidden state），架构上无法并行/批处理——这是 GPU 跑不起来的主因（逐窗口 CPU↔GPU 同步是次因），GPU 比 CPU 慢几个数量级（CPU 实测 ~119x，silero 的 model.py 亦 set_num_threads(1)、官方建议跑 CPU）。`detect_speech` 接 silero 的 `progress_tracking_callback` 回报百分比（长音频扫全程耗时可观，借此刷进度避免看似卡死）。
- `punctuator.py` — `Punctuator`，给无标点的转录 chunk 批量补日语句读（。！？），**时间戳原样保留**。复用 kotoba 官方同款标点模型（punctuators 库 `PunctCapSegModelONNX`）。蒸馏模型 chunk 不带句末标点会使 `segment_sentences` 的标点断句失效，补标点后才能在 chunk 边界正常断句。重依赖 punctuators（ONNX），**延迟导入**。
- `downloader.py` — `TwitchDownloader`，异步下载 TS 分片 → ffmpeg 合并 MP4 → 可选提取 MP3。仅依赖 httpx + ffmpeg，**不依赖 torch**。
- `translator.py` — `LLMTranslator`，经 `llm.LLMClient` 把日语句子逐批译成中文。`TranslationError` 继承 `LLMError`。
- `separator.py` — `VocalSeparator`，封装 audio-separator 分离人声（默认只出 Vocals 轨）。重依赖 audio-separator（含 torch/onnxruntime），**延迟导入**（`load()` 内才 import）。底层统一产出**无损 WAV** 作中间产物，最终格式/比特率由 `_encode_final` 用 ffmpeg 一次性套用。长音频按 `segment_minutes`（默认 15）切段→逐段分离→ffmpeg concat 合并，避免一次性出整轨爆内存。比特率默认对齐原音频（探音频流比特率 → 向上取整到 2 的幂 kbps、夹 [32,320]、留 5% 容差吸收 MP3 标称偏差），`--output-bitrate` 可覆盖；无损格式忽略比特率。
- `summarizer.py` — `Summarizer`，经 `llm.LLMClient` 对 SRT 做 AI 总结。提示词走 JSON 预设（包内 `prompts.json` + 用户 `--prompt-file` 覆盖），长字幕 map-reduce 分块。纯逻辑（`load_presets` / `resolve_preset` / `format_sentences` / `chunk_sentences`）不触网、可单测。
- `markdown_image.py` — 总结 Markdown 渲染图片。可选依赖 `markdown-it-py` 负责 Markdown → HTML；可选重依赖 Playwright 负责 Chromium 元素截图，**仅在图片渲染路径内导入/使用**。内置 light/dark HTML/CSS 主题和中日文字体栈。
- `data/prompts.json` — 内置总结提示词预设（timeline / summary / highlights / setlist）+ `reduce_system` 合并提示词，随包发布（置于 `data/` 子目录与 .py 源码分离）。
- `cli.py` — argparse 子命令 `download` / `subtitle` / `translate` / `separate` / `summarize`（分别带简写别名 `dl` / `srt` / `tr` / `sep` / `sum`），把各模块串成流水线。子命令用 `set_defaults(func=...)` 绑定处理函数，别名与规范名统一分发。

数据流：

- 转字幕：`downloader`(MP3) →（可选 `separator` 分离人声）→ **规划阶段**（进度条前跑完）：默认 `transcriber.plan_windows()`（VAD 探话语级人声窗口，返回 `(duration, windows)`）或 `--full-transcribe` 时 `plan_audio()`（VAD 探间隙切含静音大段，返回 `(duration, segments, speech)`）→ **转录阶段**：对应 `transcribe_windows()`（逐窗口、丢窗口间静音）或 `transcribe_segments()`（逐段含静音连续转），均逐段产出 chunk 级 list[Word] →（可选 `punctuator.restore()` 补标点）→ 每段 `subtitle.segment_sentences()`(list[Sentence]，传入 `speech`/`windows` 把虚高 end 夹回人声边界) → `subtitle.SrtWriter.append()` 增量写盘。`_audio_to_srt` 是这条流水线，按 `--full-transcribe` 二选一分支，`_maybe_separate` 按 `--separate` 决定是否预处理人声，标点恢复默认开启（`--no-punctuate` 关闭）。
- 译字幕：SRT 文件 → `subtitle.parse_srt()`(list[Sentence]) → `translator.translate()`(中文 list[Sentence]) → `subtitle.write_srt()`。
- 分离人声：音频 →（长音频先按 `segment_minutes` 切段）→ `separator.VocalSeparator.separate()`（逐段分离出无损 WAV → ffmpeg concat 合并 → 按目标格式/比特率编码）→ 人声音频文件（可再交给 subtitle）。
- 总结：SRT 文件 → `subtitle.parse_srt()` → `summarizer.format_sentences/chunk_sentences` → `summarizer.Summarizer.summarize()`（按预设逐块总结，多块再 reduce 合并）→ Markdown 文件。
- 渲染图片：`summarize` 的 Markdown 结果 → `markdown_image.build_markdown_html()` 生成完整 HTML → Playwright Chromium 打开 HTML → 截 `.kits-card` 元素为 PNG。图片渲染是总结的可选附加产物，失败只打印 warning，不中断已写盘的 Markdown 总结。

关键约定：

- `cli.py` 中对 `transcriber` / `downloader` / `translator` / `separator` / `summarizer` / `punctuator` / `markdown_image` 用**延迟导入**（函数内 import），避免无谓加载 GPU 栈、网络栈或浏览器栈。新增重依赖模块时沿用此模式。
- LLM 调用统一走 `llm.LLMClient`（OpenAI 兼容，默认 DeepSeek），不要在 `translator` / `summarizer` 里重复写 HTTP / 鉴权。新增 LLM 能力时复用该客户端，领域逻辑（分批、提示词）留各自模块。
- `download --srt` 隐含 `--mp3`（生成字幕必须先有音频），逻辑在 `_run_download` 的 `need_mp3 = args.mp3 or args.srt`。
- 断句与分段参数（`--max-gap` / `--target-chunk` 等）由 `_add_subtitle_args` 在 download / subtitle 间共用。
- TS URL 里的分片编号**仅用于定位 base_url**，下载范围默认从 0 开始、用指数探测+二分定位结尾（`detect_end_number`）。
- 长音频分段（两模式，**VAD 都固定走 CPU**：silero 是 LSTM、隐状态时间步串行依赖，架构上无法并行，GPU 反被拖垮，CPU ~119x 实时）：
  - **默认逐窗口**（`plan_windows` + `transcribe_windows`）：silero 原生参数（`VAD_WINDOW_*`）直接吐话语级人声窗口，逐窗转、窗口起点做偏移锚点、丢弃窗口间静音。根治前导静音漂移。**唱歌延音被 silero 漏判 → 整段丢字幕**（1343~1357s 实测无窗口；threshold 压到 0.1 也只救回末尾 1~2s 碎片）。窗口间静音隔开、不相邻，无需 overlap 去重。窗口两侧加 `pad`（0.2s）防裁首尾音素，窗口间隔 ≥ min_silence(1.5s) > 2*pad 故不重叠。
  - **`--full-transcribe`**（`plan_audio` + `transcribe_segments`）：`speech_to_gaps` 反推非语音间隙，`plan_segments` 在 `(target_chunk, max_chunk)` 窗口内选时长最长间隙中点切含静音大段（无间隙才 `max_chunk` 硬切兜底），整段连续转。不靠 VAD 决定转不转 → 唱歌不丢。每段取数窗口两侧外扩 `--segment-overlap`（2s）垫料、`_shift_words` 加回偏移、`_keep_core_words` 按词中心 [start,end) 裁回去重（首段左/末段右不设限），接缝无缝。`--vad-threshold`/`--min-silence` 调间隙探测灵敏度（仅此模式用）。
  - 取舍：默认对齐最准（纯说话场次），`--full-transcribe` 不丢唱歌（歌枠）。直播总结场景默认即可，延音丢失影响小。
- `translator` / `summarizer` 按批 / 分块发送，DeepSeek 调用经公共客户端。翻译用「序号|||文本」格式逐条对应、按序号回填防错位；总结用 map-reduce（逐块总结后 reduce 合并），提示词来自 `prompts.json` 预设。翻译只改文本、时间戳原样保留。

## 扩展点

- **新增总结预设**：直接在 `src/kits/data/prompts.json` 的 `presets` 里加一项（含 `description` + `system`），或让用户用 `--prompt-file` 传外部 JSON 覆盖，无需改代码。
- **新增游戏播报过滤**：在 `filters.py` 的 `GAME_CALLOUTS` / `_GAME_ALIASES` 登记词表与别名。
- **新增 LLM 能力**：复用 `llm.LLMClient`（默认 DeepSeek，可配 `base_url` 接其他 OpenAI 兼容端点），在 `cli.py` 加子命令（沿用延迟导入，参考 `translate` / `sum`）。
- **换人声分离模型**：`separate --model <文件名>` 或 `subtitle --separate --separate-model <文件名>`，模型名走 audio-separator 的模型库（`uv run audio-separator --list_models` 查可用名）。`.onnx`/MDX 走 onnxruntime GPU、`.ckpt`/MDXC roformer 走 torch GPU，两条链路的加速参数语义不同（见「注意」段）。
- **调人声分离速度/体积**：慢先确认 onnxruntime 走的是 GPU（`uv run python -c "import onnxruntime;print(onnxruntime.get_available_providers())"` 应含 `CUDAExecutionProvider`）；再调 `--segment-size` / `--overlap`。长音频爆内存调小 `--segment-minutes`。输出体积靠 `--output-bitrate`（默认已对齐原音频）。
