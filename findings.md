# 发现与决策

## 需求
- CLI 从 3 子命令扩到 5：download / subtitle / translate / **separate** / **sum**
- separate：分离音频中的人声（audio-separator，UVR/MDX 模型）
- sum：对已有 SRT 用 DeepSeek 总结，提示词走 JSON 预设、后续调用直接输出
- 同时重构：把 translate 与 sum 共用的 DeepSeek 调用抽成公共客户端

## 现有架构基线（已读代码）
- `subtitle.py` 纯逻辑：Word/Sentence TypedDict、segment_sentences、parse_srt、write_srt、SrtWriter（增量写）
- `transcriber.py` torch：Transcriber.transcribe / transcribe_segmented（按静音分段流式产出 list[Word]）
- `downloader.py` httpx：TwitchDownloader 下载 TS → 合并 MP4 → 提取 MP3
- `translator.py` httpx：DeepSeekTranslator，按批「序号|||文本」翻译，按序号回填
- `filters.py` 纯逻辑：游戏播报过滤
- `cli.py` argparse：3 子命令，重依赖延迟导入
- DeepSeek 调用细节：URL=https://api.deepseek.com/chat/completions，model=deepseek-chat，
  Bearer 鉴权，messages=[system,user]，从 choices[0].message.content 取结果
- 基线测试：91 passed

## 研究发现（audio-separator，来源 PyPI JSON + README，外部内容仅记此处）
- 版本 0.44.2，`requires-python>=3.10`（项目 3.12~3.15 兼容）
- 依赖 `torch>=2.3`（项目已有 2.11）、`numpy>=2`、librosa、onnxruntime
- GPU 安装：`pip install "audio-separator[gpu]"` → `onnxruntime-gpu>=1.17`
- **CUDA 兼容**：onnxruntime-gpu 要求 cuDNN 9.* + CUDA 12.*，与项目 cu128 栈一致 ✓
- 默认模型：`model_bs_roformer_ep_317_sdr_12.9755`（BS-Roformer，高质量）
- Python API：
  ```python
  from audio_separator.separator import Separator
  sep = Separator(output_dir='out', output_format='MP3', output_single_stem='Vocals')
  sep.load_model()                      # 不传则用默认模型
  sep.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')  # 指定模型
  files = sep.separate('audio.wav')     # 返回输出文件路径 list[str]
  files = sep.separate('audio.wav', output_names)  # 可自定义输出名
  ```
- 关键构造参数：output_dir、model_file_dir（默认 /tmp/audio-separator-models/）、
  output_format（默认 WAV）、output_single_stem（'Vocals' 只出人声）
- 风险：引入 onnxruntime-gpu + numpy>=2，需 uv sync 验证与现有 torch/transformers 不冲突

## 技术决策
| 决策 | 理由 |
|------|------|
| DeepSeek 公共客户端抽 HTTP+Key+错误 | translator/summarizer 共用，批处理逻辑留各自模块 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
|      |         |

## 资源
- DeepSeek API: https://api.deepseek.com/chat/completions（现有 translator 已用）

## kotoba-whisper-v2.2 适配 + 标点恢复（阶段 6，外部内容仅记此处）
- 崩溃根因：模型 config decoder_layers=2 / encoder_layers=32，但 generation_config.json
  的 alignment_heads = [[7,0],[10,17],...,[25,6]]（继承自 large-v3，max 层=25）。
  return_timestamps="word" → _extract_token_timestamps 执行 cross_attentions[l][:,h]，
  l 取 7..25 而 cross_attentions 只有 2 层 → IndexError。
- 修复：_transcribe_file 改 return_timestamps=True（chunk/短语级，不走 alignment_heads）。
  chunk 结构 {"text", "timestamp": (start,end)}，兼容 Word 契约，segment_sentences 不改。
- 实测代价（120s VALORANT 音频）：chunk 无句末标点、时间戳连续 → 标点/停顿断句失效，
  7 条里 5 条被 max_duration=15 硬钳，时间轴压缩、有重复文本。
- 官方 kotoba_whisper.py 的标点方案（README/源码）：
  - `from punctuators.models import PunctCapSegModelONNX`
  - 模型 `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`
  - `Punctuator.punctuate(text)`：若文本已含 !?、。则原样返回；否则
    `"".join(model.infer([text])[0])`；结果含 'unk' 则回退原文
  - 官方对「按说话人聚合的整段文本」标点化，丢时间戳 → 不适合 SRT
- 本项目方案：逐 chunk 批量标点化（infer 支持 batch），时间戳不动，
  让 segment_sentences 在 chunk 边界靠标点断句。待实测确认 infer 只插标点不乱改字。
- 用户动机：选 kotoba 是因日语识别更准 → 值得加适配，不回退原版。

---
*每执行2次查看/浏览器/搜索操作后更新此文件*
