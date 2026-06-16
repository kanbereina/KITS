# 进度日志

## 会话：2026-06-15

### 阶段 1：需求与发现
- **状态：** in_progress
- 执行的操作：
  - 通读 cli/subtitle/transcriber/downloader/translator/filters 6 个模块 + 测试
  - 跑基线测试：91 passed in ~7s
  - 跑 git diff：transcriber.py 仅改了 MODEL_ID（whisper-large-v3-turbo → kotoba-whisper-v2.2，未提交）
  - 用 AskUserQuestion 确认 4 个关键决策
  - 创建 task_plan.md / findings.md / progress.md
- 创建/修改的文件：
  - task_plan.md / findings.md / progress.md（新建）

### 阶段 2：规划与结构
- **状态：** pending

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 基线 pytest | 现有代码 | 全过 | 91 passed | ✅ |
| translator 重构后 | test_translator.py | 全过 | 10 passed | ✅ |
| 全量 pytest | 含新增 deepseek/summarizer/separator 测试 | 全过 | 112 passed | ✅ |
| ruff check . | 全仓 | 无告警 | All checks passed | ✅ |
| CLI 冒烟 | build_parser / separate -h / sum -h | 正常 | OK | ✅ |
| summarizer 纯逻辑 | 预设/格式化/分块/错误路径 | 符合预期 | OK | ✅ |
| uv lock | 加 audio-separator[gpu] | 解析无冲突 | 106 包，含 onnxruntime-gpu 1.26.0 | ✅ |
| uv build --wheel | 验证打包 | prompts.json 入包 | wheel 含 kits/prompts.json | ✅ |

## 未能在本会话验证（需用户环境）
- `uv sync` 实际安装 audio-separator[gpu]（重下载，含 onnxruntime-gpu）
- `separate` 子命令端到端跑分离（需安装 + 真实 GPU）
- `--separate` 集成转录、`sum` 端到端（需 GPU / DeepSeek Key）
- separator.py 用延迟导入 + 已按官方 API 写，单测验证了构造不触发重依赖导入

## 阶段 6：kotoba 适配 + 标点恢复（2026-06-15 续）
- 根因：kotoba-whisper-v2.2 decoder_layers=2，但 alignment_heads 继承 large-v3（引用第 25 层）→ word 级时间戳 IndexError
- 改 transcriber `_transcribe_file` → `return_timestamps=True`（chunk 级），崩溃消除
- 装 punctuators 0.0.7，新建 punctuator.py（Punctuator.restore 批量补标点、时间戳不变）
- cli._audio_to_srt 在断句前接入标点恢复（默认开，--no-punctuate 关）

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| chunk 级转录探针 | 60s 真实音频 | 不崩溃、结构兼容 Word | 7 chunk，结构 OK | ✅ |
| punctuator API 探针 | 5 条真实 chunk 文本 | 合理补 。！？ | 质量好，副带 ・ | ✅ |
| 端到端探针（标点前） | 120s 音频 | — | 7 条字幕，5 条被 15s 硬切 | 基线 |
| 端到端探针（标点后） | 120s 音频 | 断句改善 | 19 条字幕，2 条被硬切 | ✅ |
| test_punctuator.py | 假模型替身 | 纯逻辑全过 | 8 passed | ✅ |
| 全量 pytest | 含标点测试 | 全过 | 120 passed | ✅ |
| ruff check . | 全仓 | 无告警 | All checks passed | ✅ |
| CLI --no-punctuate | 参数解析 | 默认 True，关后 False | 符合 | ✅ |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
|        |      | 1       |         |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 1 末（研究 audio-separator）→ 阶段 2 |
| 我要去哪里？ | 规划 → 实现 → 测试 → 交付 |
| 目标是什么？ | CLI 扩到 5 子命令 + 抽 DeepSeek 公共客户端，无回归 |
| 我学到了什么？ | 见 findings.md |
| 我做了什么？ | 见上方记录 |

---
*每个阶段完成后或遇到错误时更新此文件*
