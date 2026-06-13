# KITS

基于 [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) 的鹿乃 Twitch 直播总结工具。可以**下载 Twitch 直播**、合并为 MP4 / 提取 MP3，并把音频转换成**完整句子、带时间戳的 SRT 字幕文件**。支持 50 系 Nvidia 显卡（CUDA 12.8）。

## 特性

- 🐙 异步并发下载 Twitch 直播 TS 分片，自动探测视频长度，合并为 MP4
- 🎵 可选提取 MP3 音频，可选保留 / 清理临时 TS 文件
- 🎤 使用 Whisper large-v3-turbo 进行语音识别，速度快、精度高
- ✂️ **单词级时间戳**断句，按句末标点 / 停顿 / 长度上限智能切分，保证句子完整不被拦腰截断
- 🔗 下载与字幕一条龙：一条命令从直播 URL 直达 SRT 字幕
- 🧹 自动清理重复字符与乱码，并抑制模型的幻觉式重复
- 📄 输出标准 SRT，可直接拖入播放器或视频剪辑软件

## 环境要求

- Python 3.12 ~ 3.14
- 支持 CUDA 的 Nvidia 显卡（转录字幕时强制要求 GPU；仅下载视频则不需要）
- CUDA 12.8（PyTorch 从 `pytorch-cu128` 源安装）
- [ffmpeg](https://ffmpeg.org/download.html)（合并 MP4、提取 MP3 需要，须在 PATH 中）
- 已安装 [uv](https://docs.astral.sh/uv/)

## 安装

```bash
# 克隆仓库后，在项目根目录同步依赖
uv sync
```

首次运行会自动从 Hugging Face 下载模型（约几个 GB），需要联网。下载后会走本地缓存。

## 使用方法

工具提供两个子命令:`download`（下载直播）和 `subtitle`（音频转字幕）。

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
| `url` | （必填） | TS 分片示例 URL |
| `-o, --output` | `output` | 输出文件名（不含扩展名） |
| `--dir` | `downloads` | 下载 / 输出目录 |
| `--start` | 从 URL 提取 | 起始分片编号 |
| `--end` | 自动探测 | 结束分片编号 |
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

### 示例

```bash
# 转录日语直播音频，输出到指定文件
uv run kits subtitle -i live_2026.mp3 -o live_2026.srt

# 句子偏碎时，调大停顿阈值让句子更连贯
uv run kits subtitle -i live_2026.mp3 --max-gap 1.0

# 转录英语内容
uv run kits subtitle -i talk.mp3 --language english
```

## 项目结构

```
src/kits/
  __init__.py      # 包入口，导出字幕相关 API
  subtitle.py      # 纯逻辑：单词时间戳 -> 完整句子 -> SRT（无 torch 依赖，可单测）
  transcriber.py   # Whisper 模型加载 + GPU 转录，产出单词级时间戳
  downloader.py    # Twitch 直播下载：异步下载 TS -> 合并 MP4 -> 提取 MP3
  cli.py           # 命令行入口（download / subtitle 子命令）
main.py            # 薄入口，委托给 kits.cli
```

各模块职责清晰、相互解耦:

- `downloader.TwitchDownloader` 下载并合并直播，产出 MP4 / MP3，不依赖 torch
- `transcriber.Transcriber.transcribe()` 把音频转成单词级时间戳列表
- `subtitle.segment_sentences()` 负责断句、`write_srt()` 负责落盘
- `cli` 把三者串成流水线：`download --srt` 即「下载 -> 提取音频 -> 转字幕」

后续接入 **DeepSeek 总结分析** 时，只需在 `src/kits/` 下新增 `summarizer.py`，吃 `transcriber` 的文本或 `subtitle` 的句子即可，并在 `cli` 中加一个 `summarize` 子命令。

## 断句逻辑

字幕按以下优先级切分句子，确保完整性:

1. **句末标点**：遇到 `。！？」` 等结尾标点即认为一句结束
2. **停顿**：与上一个词的间隔超过 `--max-gap` 时断句
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
