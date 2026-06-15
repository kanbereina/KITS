# KITS

基于 [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) 的鹿乃 Twitch 直播总结工具。可以**下载 Twitch 直播**、合并为 MP4 / 提取 MP3，把音频转换成**完整句子、带时间戳的 SRT 字幕文件**，并可调用 **DeepSeek 把日语字幕翻译成中文**。支持 50 系 Nvidia 显卡（CUDA 12.8）。

## 特性

- 🐙 异步并发下载 Twitch 直播 TS 分片，自动探测视频长度，合并为 MP4
- 🎵 可选提取 MP3 音频，可选保留 / 清理临时 TS 文件
- 🎤 使用 Whisper large-v3-turbo 进行语音识别，速度快、精度高
- ✂️ **单词级时间戳**断句，按句末标点 / 停顿 / 长度上限智能切分，保证句子完整不被拦腰截断
- 🪓 长音频**按静音切分、分段流式转录**：实时进度、边转边写盘，切点落在无人说话处，精度不受影响
- 🔗 下载与字幕一条龙：一条命令从直播 URL 直达 SRT 字幕
- 🌐 调用 DeepSeek 把日语 SRT 翻译成中文 SRT，逐条对应、保留原时间轴
- 🧹 自动清理重复字符与乱码，并抑制模型的幻觉式重复
- 🎮 可选剔除 VALORANT 游戏内系统播报 / 技能语音（如「残り1名」「グレネード配置」），让字幕聚焦主播人声
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

工具提供三个子命令:`download`（下载直播）、`subtitle`（音频转字幕）和 `translate`（日语字幕译中文）。

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
| `--silence-db` | `-30.0` | 静音判定响度阈值（dB），越负越宽松 |
| `--min-silence` | `0.5` | 最短静音时长（秒），短于此不算切点 |
| `--filter-game` | 关闭 | 剔除指定游戏的播报 / 技能语音，按游戏名启用、可多次指定（整条完全匹配才删） |

#### 长音频分段转录

音频时长超过 `--max-chunk`（默认 10 分钟）时，会自动启用分段转录：

1. 先用 ffmpeg `silencedetect` 探测全部静音区间
2. 从每段起点出发累积到 `--target-chunk`（默认 5 分钟），在其后**第一个静音中点**处切开；若到 `--max-chunk` 仍无静音可切（如长时间唱歌 / BGM），则在硬上限处强切兜底
3. 逐段切出临时音频 → 转录 → 该段字幕**立即追加写入 SRT**

好处：实时显示 `转录第 i/N 段` 进度、边转边落盘（中途中断已转部分仍是合法 SRT）、峰值显存更低。切点都落在无人说话处，**不会把句子拦腰截断，精度与整段转录一致**。短音频则自动整段转录，无额外开销。

> 鹿乃直播常有唱歌 / BGM，这些不是静音，若某段一直有声音会触发硬上限强切。可调大 `--silence-db`（如 `-35`）放宽静音判定，或调大 `--max-chunk` 容忍更长的段。

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
```

## 项目结构

```
src/kits/
  __init__.py      # 包入口，导出字幕相关 API
  subtitle.py      # 纯逻辑：单词时间戳 -> 完整句子 -> SRT，含 SRT 解析与增量写入（无 torch 依赖，可单测）
  filters.py       # 纯逻辑：剔除游戏内系统播报 / 技能语音（无 torch 依赖，可单测）
  transcriber.py   # Whisper 模型加载 + GPU 转录，长音频按静音切分、分段流式产出词级时间戳
  downloader.py    # Twitch 直播下载：异步下载 TS -> 合并 MP4 -> 提取 MP3
  translator.py    # 调用 DeepSeek 把日语 SRT 翻译成中文 SRT（仅依赖 httpx）
  cli.py           # 命令行入口（download / subtitle / translate 子命令）
main.py            # 薄入口，委托给 kits.cli
```

各模块职责清晰、相互解耦:

- `downloader.TwitchDownloader` 下载并合并直播，产出 MP4 / MP3，不依赖 torch
- `transcriber.Transcriber.transcribe()` 把音频转成单词级时间戳列表；`transcribe_segmented()` 按静音切分长音频、分段流式产出
- `subtitle.segment_sentences()` 负责断句、`write_srt()` / `SrtWriter` 负责落盘（后者支持分段增量写）、`parse_srt()` 负责把 SRT 读回句子列表
- `translator.DeepSeekTranslator.translate()` 把句子列表逐批译成中文，仅依赖 httpx
- `cli` 把它们串成流水线：`download --srt` 即「下载 -> 提取音频 -> 转字幕」

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
