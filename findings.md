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

## 视觉/浏览器发现
<!-- 每执行2次查看/搜索操作后更新此部分 -->
-

---
*每执行2次查看/浏览器/搜索操作后更新此文件*
