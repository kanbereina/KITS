<div align="center">

# KITS

_鹿乃 Twitch 直播智能总结_

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Release](https://img.shields.io/github/v/release/kanbereina/KITS?display_name=tag&sort=semver)](https://github.com/kanbereina/KITS/releases)
[![Python](https://img.shields.io/badge/Python-3.12~3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-cu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

一站式处理鹿乃 Twitch 直播：**下载直播 → 分离人声 → 转写 SRT 日语字幕文件 → 可选转写 SRT 中文字幕文件 → AI 总结**。

</div>

---

## 目录

- [为什么用 KITS](#为什么用-kits-)
- [功能特性](#功能特性)
- [安装与部署](#安装与部署)
  - [环境要求一览](#环境要求一览)
- [使用方法](#使用方法)
  - [download：下载 Twitch 直播](#download下载-twitch-直播)
  - [subtitle：音频转字幕](#subtitle音频转字幕)
    - [标点恢复（默认开启）](#标点恢复默认开启)
    - [长音频分段转录](#长音频分段转录)
    - [剔除游戏播报（--filter-game）](#剔除游戏播报---filter-game)
  - [translate：日语字幕译中文](#translate日语字幕译中文)
  - [separate：分离人声](#separate分离人声)
  - [summarize：AI 总结字幕](#summarizeai-总结字幕)
  - [示例](#示例)
- [项目结构](#项目结构)
- [断句逻辑](#断句逻辑)
- [支持的音频格式](#支持的音频格式)
- [输出示例](#输出示例)
- [常见问题](#常见问题)

## 为什么用 KITS ？

- **一条龙流水线** — 从直播 URL 到中文字幕、AI 总结，全程命令行串联，无需手动倒腾中间文件。
- **日语识别更准** — 默认 [kotoba-whisper-v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) 蒸馏模型，自动补日语句读后再断句，字幕是完整句子而非碎片。
- **句子完整不截断** — 长音频按静音切分、分段流式转写，切点落在无人说话处，边转边落盘，精度与整段转写一致。
- **唱歌场次友好** — 内置 audio-separator 人声分离，长音频自动切段防爆内存、输出比特率对齐原音频，去掉 BGM / 伴奏再识别。
- **省心的中文化** — DeepSeek 逐条翻译保留时间轴，并能按时间线 / 概述 / 高光 / 歌单等预设一键总结整场直播。
- **专为 50 系显卡调优** — PyTorch 走 `pytorch-cu128`（CUDA 12.8），onnxruntime 复用 torch 自带 CUDA 运行时，开箱即用 GPU 加速。

## 功能特性

- 🐙 异步并发下载 Twitch 直播 TS 分片，自动探测视频长度，合并为 MP4
- 🎵 可选提取 MP3 音频，可选保留 / 清理临时 TS 文件
- 🎤 使用 Whisper 进行语音识别，默认 `kotoba-whisper-v2.2`（日语识别更准的蒸馏模型）
- ✂️ 句子级断句，按句末标点 / 停顿 / 长度上限智能切分，保证句子完整不被拦腰截断
- ✒️ 蒸馏模型只产短语级时间戳且不带标点，自动用标点模型补日语句读（。！？）后再断句，时间戳原样保留
- 🪓 长音频**按静音切分、分段流式转录**：实时进度、边转边写盘，切点落在无人说话处，精度不受影响
- 🔗 下载与字幕一条龙：一条命令从直播 URL 直达 SRT 字幕
- 🌐 调用 DeepSeek 把日语 SRT 翻译成中文 SRT，逐条对应、保留原时间轴
- 🤖 调用 DeepSeek 对 SRT 字幕做 AI 总结，提示词走 JSON 预设（时间线 / 概述 / 高光 / 歌单），长字幕自动分块
- 🎚️ 用 audio-separator 分离人声，可单独导出，也可在转录前 `--separate` 预处理去掉 BGM / 唱歌干扰
- 🧹 自动清理重复字符与乱码，并抑制模型的幻觉式重复
- 🎮 可选剔除 VALORANT 游戏内系统播报 / 技能语音（如「残り1名」「グレネード配置」），让字幕聚焦主播人声
- 📄 输出标准 SRT，可直接拖入播放器或视频剪辑软件

## 安装与部署

按以下步骤从零部署，全程约几分钟（不含模型下载）：

**1. 准备系统依赖**

- [uv](https://docs.astral.sh/uv/)（包管理 + 运行器）— 装好后 `uv --version` 应有输出
- [ffmpeg](https://ffmpeg.org/download.html) — 须在 PATH 中，`ffmpeg -version` 应有输出（合并 MP4 / 提取 MP3 / 音频切分都依赖它）
- Nvidia 显卡驱动 — 转字幕、分离人声需要，`nvidia-smi` 应能看到显卡

**2. 克隆并同步依赖**

```bash
git clone https://github.com/kanbereina/KITS.git
cd KITS
uv sync              # 创建虚拟环境并装齐所有依赖（含 dev 组）
```

`uv sync` 会自动从 `pytorch-cu128` 源装好 CUDA 12.8 版 PyTorch，以及 audio-separator（含 onnxruntime-gpu）、punctuators 等。无需手动装 CUDA Toolkit——onnxruntime 复用 torch 自带的 CUDA 运行时。

**3. 验证安装**

```bash
uv run kits --help          # 看到子命令说明即安装成功
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"  # 应输出 CUDA: True
```

**4.（可选）配置 DeepSeek**

`translate` / `summarize` 需要 DeepSeek API Key，使用命令时传入或设置环境变量：

```bash
export DEEPSEEK_API_KEY=sk-xxxx      # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-xxxx"
```

> 首次运行转字幕会自动从 Hugging Face 下载模型（kotoba-whisper 约几个 GB + 标点模型约 1GB），需要联网；之后走本地缓存。

### 环境要求一览

| 项 | 要求 | 说明 |
| --- | --- | --- |
| Python | 3.12 ~ 3.14 | 由 uv 管理虚拟环境 |
| GPU | 支持 CUDA 的 Nvidia 显卡 | 转字幕 / 分离人声强制要求；仅下载不需要 |
| CUDA | 12.8 | PyTorch 从 `pytorch-cu128` 源安装，勿换 PyPI 默认源 |
| ffmpeg | 在 PATH 中 | 合并 MP4、提取 MP3、音频切分 |
| API Key | DeepSeek（可选） | 仅 `translate` / `summarize` 需要 |

## 使用方法

工具提供五个子命令:`download`（下载直播）、`subtitle`（音频转字幕）、`translate`（日语字幕译中文）、`separate`（人声分离）和 `summarize`（AI 总结字幕）。每个命令都带一个简写别名，可互换使用。

| 命令 | 别名 | 用途 | 最简示例 | 需 GPU | 需 API Key |
| --- | --- | --- | --- | :---: | :---: |
| [`download`](#download下载-twitch-直播) | `dl` | 下载 Twitch 直播，合并 MP4 / 提取 MP3 | `uv run kits dl "<ts_url>" -o name` | 否¹ | 否 |
| [`subtitle`](#subtitle音频转字幕) | `srt` | 音频转带时间戳的 SRT 字幕 | `uv run kits srt -i audio.mp3` | 是 | 否 |
| [`translate`](#translate日语字幕译中文) | `tr` | 日语 SRT 翻译成中文 SRT | `uv run kits tr -i live.srt` | 否 | 是 |
| [`separate`](#separate分离人声) | `sep` | 从音频分离出人声，去 BGM / 伴奏 | `uv run kits sep -i audio.mp3` | 是 | 否 |
| [`summarize`](#summarizeai-总结字幕) | `sum` | 对 SRT 做 AI 总结（时间线 / 歌单等） | `uv run kits sum -i live.srt` | 否 | 是 |

> ¹ `download` 本身不需要 GPU；但加 `--srt`（下载后自动转字幕）会调用 Whisper，需要 GPU。

### download：下载 Twitch 直播

传入任意一个 TS 分片的 URL（形如 `https://.../chunked/1710.ts`），自动探测整场直播范围并下载合并:

```bash
# 下载并合并为 MP4
uv run kits download "https://.../chunked/1710.ts" -o my_stream

# 下载 + 提取 MP3
uv run kits download "https://.../chunked/1710.ts" -o my_stream --mp3

# 一条龙：下载 -> 提取音频 -> 生成 SRT 字幕
uv run kits download "https://.../chunked/1710.ts" -o my_stream --srt
```

产物默认放在 `downloads/` 目录下。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `url` | （必填） | TS 分片示例 URL（仅用于定位地址，编号被忽略） |
| `-o, --output` | `output` | 输出文件名（不含扩展名） |
| `--dir` | `downloads` | 下载 / 输出目录 |
| `--start` | `0` | 起始分片编号，默认从 0 下载整场直播 |
| `--end` | 二分探测 | 结束分片编号，默认指数探测 + 二分自动定位结尾 |
| `--concurrent` | `5` | 最大并发下载数 |
| `--keep-ts` | 关闭 | 保留临时 TS 文件 |
| `--mp3` | 关闭 | 额外导出 MP3 |
| `--srt` | 关闭 | 额外生成 SRT 字幕（自动转录，隐含导出 MP3） |

> `--srt` 会调用 Whisper 转录，需要 GPU。还支持下方 `subtitle` 的全部断句参数（`--max-gap` 等）。

### subtitle：音频转字幕

已有音频文件时，直接转 SRT。输入为**必填项**，用 `-i` 指定:

```bash
uv run kits subtitle -i your_audio.mp3
```

默认输出到 `subtitle.srt`，可用 `-o` 自定义:

```bash
uv run kits subtitle -i your_audio.mp3 -o output.srt
```

> 也可以用 `uv run python main.py subtitle -i ...`，效果等价（`main.py` 是薄入口）。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-i, --input` | （必填） | 输入音频文件路径 |
| `-o, --output` | `subtitle.srt` | 输出 SRT 文件路径 |
| `--language` | `japanese` | 识别语言 |
| `--beams` | `1` | beam search 数量，`1` 为贪心解码，更快 |
| `--max-gap` | `0.7` | 判定断句的最大停顿（秒），越小切得越碎 |
| `--max-chars` | `60` | 单条字幕最大字符数 |
| `--max-duration` | `15.0` | 单条字幕最大时长（秒） |
| `--target-chunk` | `300.0` | 分段目标时长（秒），长音频每段约这么长 |
| `--max-chunk` | `600.0` | 单段硬上限（秒），段内无静音时在此强切 |
| `--silence-db` | `-45.0` | 静音判定响度阈值（dB），音量低于此值算静音；越负越严格（检出静音更少），越接近 0 越宽松 |
| `--min-silence` | `0.5` | 最短静音时长（秒），短于此不算切点 |
| `--filter-game` | 关闭 | 剔除指定游戏的播报 / 技能语音，按游戏名启用、可多次指定（整条完全匹配才删） |
| `--separate` | 关闭 | 转录前先用 audio-separator 分离人声（去 BGM / 唱歌干扰，需安装 audio-separator + GPU） |
| `--separate-model` | `UVR-MDX-NET_Main_427.onnx` | 人声分离模型文件名，仅在 `--separate` 时生效 |
| `--separate-segment-size` | `512` | 人声分离分块大小，越大越快越吃显存，仅在 `--separate` 时生效 |
| `--separate-overlap` | `0.1` | 人声分离分块重叠（0~1），越小越快，仅在 `--separate` 时生效 |
| `--separate-segment-minutes` | `15` | 人声分离长音频切段时长（分钟），防爆内存（`<=0` 关闭），仅在 `--separate` 时生效 |
| `--separate-output-bitrate` | 自动对齐原音频 | 人声输出比特率（如 `128k`），无损格式忽略，仅在 `--separate` 时生效 |
| `--no-punctuate` | （默认补标点） | 关闭标点恢复；模型本身已输出标点时可关 |
| `--punct-model` | xlm-roberta 日语句读 | 标点恢复模型 |

#### 标点恢复（默认开启）

默认模型 `kotoba-whisper-v2.2` 是蒸馏模型，日语识别更准，但只产出**短语级时间戳且不带句末标点**。无标点会让断句只能靠长度上限硬切，字幕被压成一条条 15 秒的长块。

为此转录后会自动用标点模型（kotoba 官方同款 [xlm-roberta 日语句读模型](https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase)）给每个短语补上 `。！？、`，**时间戳原样保留**，让断句在句末标点处自然切开。实测一段 120 秒音频，补标点后字幕从 7 条细化到 19 条、被硬切的从 5 条降到 2 条。

```bash
# 默认补标点，无需额外参数
uv run kits subtitle -i live.mp3

# 若换用本身已带标点的模型，可关闭标点恢复
uv run kits subtitle -i live.mp3 --no-punctuate
```

首次运行会下载标点模型（约 1GB）。标点模型在 CPU 上即可快速推理。

#### 长音频分段转录

音频时长超过 `--max-chunk`（默认 10 分钟）时，会自动启用分段转录：

1. 先用 ffmpeg `silencedetect` 探测全部静音区间
2. 从每段起点出发累积到 `--target-chunk`（默认 5 分钟），在其后**第一个静音中点**处切开；若到 `--max-chunk` 仍无静音可切（如长时间唱歌 / BGM），则在硬上限处强切兜底
3. 逐段切出临时音频 → 转录 → 该段字幕**立即追加写入 SRT**

好处：实时显示 `转录第 i/N 段` 进度、边转边落盘（中途中断已转部分仍是合法 SRT）、峰值显存更低。切点都落在无人说话处，**不会把句子拦腰截断，精度与整段转录一致**。短音频则自动整段转录，无额外开销。

> 鹿乃直播常有唱歌 / BGM，这些不是静音，若某段一直有声音会触发硬上限强切。可调大 `--silence-db`（往接近 0 调，如 `-35`）放宽静音判定，或调大 `--max-chunk` 容忍更长的段。

#### 剔除游戏播报（--filter-game）

直播玩 VALORANT 时，麦克风会混入大量游戏音——系统播报（「残り1名」「ディフェンダーの勝利」）和特工技能语音（「アラームボット配置」「グレネード配置」）。这些不是主播说的话，对「直播字幕」是噪声，还会在后续翻译时白白消耗 token。用 `--filter-game` 指定游戏名即可剔除：

```bash
# 指定游戏名（大小写不敏感，支持简写 valo）
uv run kits subtitle -i live.mp3 --filter-game valorant
uv run kits subtitle -i live.mp3 --filter-game valo

# 多款游戏可重复指定（词表合并）
uv run kits subtitle -i live.mp3 --filter-game valorant --filter-game valo
```

目前内置词表：`valorant`（简写 `valo`）。传入未收录的游戏名会报错并列出当前支持的名字。

采用**整条完全匹配**策略：仅当一条字幕（去空格、去首尾标点后）恰好等于内置词表中的某条播报词时才删除，最大限度避免误删主播人声。因此：

- ✅ 整条纯播报（如 `セントリー設置`）会被删
- ⚠️ 播报与人声混在同一条（如 `中央に敵だ…ご視聴ありがとうございました`）会**整条保留**
- ⚠️ 词表未收录的转录变体可能漏删

默认关闭，仅在玩对应游戏的场次按需开启。新增游戏只需在 `filters.py` 的 `GAME_CALLOUTS` / `_GAME_ALIASES` 登记词表与别名。

### translate：日语字幕译中文

把已有的日语 SRT 字幕调用 DeepSeek 翻译成中文 SRT，逐条对应、保留原时间轴。需要 DeepSeek API Key：

```bash
# 用环境变量提供 Key（推荐）
export DEEPSEEK_API_KEY=sk-xxxx
uv run kits translate -i live.srt

# 或用命令行参数传入 Key，并自定义输出
uv run kits translate -i live.srt -o live_cn.srt --api-key sk-xxxx
```

不指定 `-o` 时，输出默认在原文件名后插入 `.zh`，如 `live.srt` → `live.zh.srt`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-i, --input` | （必填） | 输入 SRT 字幕文件路径 |
| `-o, --output` | `原名.zh.srt` | 输出 SRT 文件路径 |
| `--api-key` | 读环境变量 | DeepSeek API Key，缺省读 `DEEPSEEK_API_KEY` |
| `--model` | `deepseek-chat` | DeepSeek 模型名 |
| `--batch-size` | `20` | 每批翻译的字幕条数 |

> 翻译只替换文本、保留时间戳。字幕按批发送给模型逐条翻译，某条译文缺失时会回退保留原日语，避免时间轴错位。

### separate：分离人声

用 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)（UVR/MDX 模型）从音频中分离出人声，去掉 BGM / 伴奏。适合鹿乃唱歌场次：先分离人声再转录，可显著降低背景音乐对识别的干扰。默认模型 `UVR-MDX-NET_Main_427.onnx`（MDX 架构，走 onnxruntime GPU 加速）。

```bash
# 分离人声，默认输出 MP3 到 output/（原名_(Vocals).mp3）
uv run kits separate -i live.mp3

# 直接指定输出文件路径（与其他命令一致，-o 给文件名）
uv run kits separate -i live.mp3 -o vocals.mp3

# 指定格式与模型
uv run kits separate -i live.mp3 -o vocals.wav --model Kim_Vocal_2.onnx
```

首次运行会自动下载分离模型。需要 CUDA GPU 与 `audio-separator[gpu]`（已在依赖中，`uv sync` 即装）。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-i, --input` | （必填） | 输入音频文件路径 |
| `-o, --output` | `--dir/原名_(Vocals).格式` | 输出人声文件路径，指定时输出格式按其扩展名 |
| `--dir` | `output` | 人声输出目录（未指定 `-o` 时生效） |
| `--model` | `UVR-MDX-NET_Main_427.onnx` | 分离模型文件名（audio-separator 模型库中的文件名） |
| `--format` | `MP3` | 输出音频格式（WAV / MP3 / FLAC 等） |
| `--segment-size` | `512` | 分块大小，越大越快越吃显存（显存紧张可降到 256） |
| `--overlap` | `0.1` | MDX(.onnx) 分块重叠（0~1），越小越快、接缝质量略降 |
| `--segment-minutes` | `15` | 长音频按此时长（分钟）切段逐段分离再合并，防爆内存（`<=0` 关闭） |
| `--output-bitrate` | 自动对齐原音频 | 输出比特率（如 `128k`），默认探测原音频并向上取整到 2 的幂；无损格式忽略 |

> 长音频（如 4 小时录播）一次性分离会撑爆内存，故默认按 `--segment-minutes` 切段、逐段分离再用 ffmpeg 无缝合并；中间产物为无损 WAV，最终只编码一次。输出比特率默认对齐原音频（人声分离后内容简单，不超过原始比特率即可保真），避免固定高码率导致文件虚大。
>
> 也可以不单独跑 `separate`，直接在 `subtitle` / `download` 上加 `--separate`，转录前自动分离人声。

### summarize：AI 总结字幕

把已有 SRT 字幕交给 DeepSeek 做总结，方便快速回顾整场直播。提示词走 JSON 预设，长字幕会自动分块总结再合并。需要 DeepSeek API Key（命令可简写为 `sum`）：

```bash
export DEEPSEEK_API_KEY=sk-xxxx

# 默认预设（timeline，时间线分段总结），输出到 live.summary.md
uv run kits summarize -i live.srt

# 指定预设：summary 概述 / highlights 高光 / setlist 歌单（用别名 sum）
uv run kits sum -i live.srt --preset setlist -o setlist.md

# 用自定义提示词 JSON 覆盖内置预设
uv run kits summarize -i live.srt --prompt-file my_prompts.json --preset mine
```

内置预设：

| 预设 | 说明 |
| --- | --- |
| `timeline` | 按话题分段、每段带时间戳的时间线总结（默认） |
| `summary` | 几段连贯文字的整体概述 |
| `highlights` | 高光时刻 / 要点列表，带时间戳 |
| `setlist` | 歌单提取，识别演唱的歌曲并按时间列出 |

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-i, --input` | （必填） | 输入 SRT 字幕文件路径 |
| `-o, --output` | `原名.summary.md` | 输出总结文件路径 |
| `--api-key` | 读环境变量 | DeepSeek API Key，缺省读 `DEEPSEEK_API_KEY` |
| `--model` | `deepseek-chat` | DeepSeek 模型名 |
| `--preset` | 配置 default | 总结预设名（`timeline` / `summary` / `highlights` / `setlist`） |
| `--prompt-file` | 无 | 自定义提示词 JSON，覆盖内置预设 |
| `--max-chars` | `8000` | 单块送审最大字符数，超长字幕按此分块 |

> 自定义提示词 JSON 格式：顶层 `presets` 是「预设名 → {description, system}」字典，可选 `default`（默认预设名）和 `reduce_system`（多块合并时的提示词）。用户预设与内置预设浅合并，同名覆盖。

### 示例

```bash
# 转录日语直播音频，输出到指定文件
uv run kits subtitle -i live_2026.mp3 -o live_2026.srt

# 句子偏碎时，调大停顿阈值让句子更连贯
uv run kits subtitle -i live_2026.mp3 --max-gap 1.0

# 玩 VALORANT 的场次，剔除游戏系统播报 / 技能语音
uv run kits subtitle -i live_2026.mp3 --filter-game valorant

# 转录英语内容
uv run kits subtitle -i talk.mp3 --language english

# 唱歌场次：先分离人声再转录，降低 BGM 干扰
uv run kits subtitle -i live_2026.mp3 --separate

# 转录完后做一份时间线总结，方便回顾
uv run kits sum -i live_2026.srt

# 提取整场直播的歌单
uv run kits sum -i live_2026.srt --preset setlist
```

## 项目结构

```
src/kits/
  __init__.py      # 包入口，导出字幕相关纯逻辑 API
  subtitle.py      # 纯逻辑：单词时间戳 -> 完整句子 -> SRT，含 SRT 解析与增量写入（无 torch 依赖，可单测）
  filters.py       # 纯逻辑：剔除游戏内系统播报 / 技能语音（无 torch 依赖，可单测）
  deepseek.py      # 公共 DeepSeek 客户端：鉴权 + HTTP + 错误处理（仅 httpx，translate/sum 共用）
  transcriber.py   # Whisper 模型加载 + GPU 转录，长音频按静音切分、分段流式产出 chunk 级时间戳
  punctuator.py    # 标点恢复：给无标点的转录 chunk 补日语句读（延迟导入 punctuators），时间戳不变
  downloader.py    # Twitch 直播下载：异步下载 TS -> 合并 MP4 -> 提取 MP3
  translator.py    # 调用 DeepSeek 把日语 SRT 翻译成中文 SRT（经 deepseek 客户端）
  separator.py     # 人声分离：封装 audio-separator（延迟导入，默认只出 Vocals 轨）
  summarizer.py    # 调用 DeepSeek 总结 SRT，提示词走 JSON 预设、长字幕 map-reduce 分块
  data/
    prompts.json   # 内置总结提示词预设（timeline / summary / highlights / setlist）
  cli.py           # 命令行入口（download / subtitle / translate / separate / summarize 子命令，各带简写别名）
main.py            # 薄入口，委托给 kits.cli
```

各模块职责清晰、相互解耦:

- `downloader.TwitchDownloader` 下载并合并直播，产出 MP4 / MP3，不依赖 torch
- `transcriber.Transcriber.transcribe()` 把音频转成（chunk/短语级）时间戳列表；`transcribe_segmented()` 按静音切分长音频、分段流式产出
- `punctuator.Punctuator.restore()` 给无标点的 chunk 补日语句读，时间戳不变（延迟导入重依赖）
- `subtitle.segment_sentences()` 负责断句、`write_srt()` / `SrtWriter` 负责落盘（后者支持分段增量写）、`parse_srt()` 负责把 SRT 读回句子列表
- `deepseek.DeepSeekClient` 集中 DeepSeek 鉴权与请求；`translator.DeepSeekTranslator` 与 `summarizer.Summarizer` 复用它
- `separator.VocalSeparator.separate()` 用 audio-separator 分离人声（延迟导入重依赖）
- `summarizer.Summarizer.summarize()` 按预设提示词总结字幕，长字幕分块再合并
- `cli` 把它们串成流水线：`download --srt` 即「下载 -> 提取音频 -> 转字幕」，`subtitle --separate` 即「分离人声 -> 转字幕」

## 断句逻辑

字幕按以下优先级切分句子，确保完整性:

1. **句末标点**：遇到 `。！？」` 等结尾标点即认为一句结束（蒸馏模型无标点时，先经标点恢复补上）
2. **停顿**：与上一段的间隔超过 `--max-gap` 时断句
3. **长度上限**：超过 `--max-chars` 或 `--max-duration` 时，优先在逗号/读点处切开，避免单条字幕过长

## 支持的音频格式

依赖底层的 `ffmpeg`/`transformers` 解码，常见的 `.mp3`、`.wav`、`.m4a`、`.flac` 等均可。

## 输出示例

```srt
1
00:00:00,000 --> 00:00:02,600
こんにちは、今日は配信です。

2
00:00:05,000 --> 00:00:06,800
ありがとうございました
```

## 常见问题

**报错 `请安装 GPU 版本的 CUDA！`**
程序检测不到可用的 CUDA 设备。请确认显卡驱动、CUDA 已正确安装，且 PyTorch 是 GPU 版本。

**模型下载失败**
检查网络连接（需要访问 Hugging Face）。如已有本地缓存，下载失败时会自动回退到缓存。

**字幕太碎 / 太长**
用 `--max-gap` 调节断句松紧（调大更连贯），用 `--max-chars`、`--max-duration` 控制单条字幕的上限。
