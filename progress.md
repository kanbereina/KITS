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

## 阶段 7：外层静音/重叠分段修复（2026-06-17）
根因：用户反馈「v2.2 适配不够，尤其静音/重叠」。核查后定性——不在模型版本（v2.2 自带 pipeline 强制
pyannote diarization + 整段处理 + 标点/时间戳分叉，反而不适合本项目，现状方向正确），而在外层 plan_segments：
硬切无重叠 → 唱歌长无静音段被 600s 硬切拦腰截断，接缝吞字/重复；内层 stride 只管单段内部、救不了接缝。

改动：
- transcriber.py 新增纯逻辑 `_word_center` / `_keep_core_words`（中心时间归属判定 + overlap 去重）
- transcribe_segmented 加 `overlap` 形参（默认 2s）：每段取数窗口两侧 pad → 转录 → _shift_words(按窗口起点)
  → _keep_core_words 按词中心裁回逻辑区间（首段左/末段右不设限）。plan_segments **不动**，5 个契约测试全保住
- slice_audio 改 `-ss {start} -i ... -t {时长}`（弃 `-to` 绝对结束，跨 ffmpeg 版本语义稳定）
- cli 加 `--segment-overlap`（默认 2.0，0 关闭），透传到 transcribe_segmented
- 迁移残留清理：模块 docstring + pyproject 描述 → kotoba-whisper-v2.2
- 真实运行（250min 直播，RTX 5060）暴露并修正一处判断：日志显示模型在做 language detection。查 generation_config.json
  确认 `is_multilingual=true` 且 forced_decoder_ids 语言槽=null（不固定语言）→ 不传 language 则每段自动检测语种，
  对唱歌/BGM/日英混杂段易误判、那段质量骤降。**修正上一轮的错误注释**（曾误判为「单语模型、language 无操作」），
  按官方 kotoba_whisper.py 放开 `language="ja"`+`task="transcribe"`，消除误判与两条相关弃用告警
- pipeline 构造 `torch_dtype` → `dtype`（transformers 5.x 弃用旧名），消除告警
- 新增 9 个纯逻辑单测（TestWordCenter ×4 + TestKeepCoreWords ×5）
- silence_db 二次宽松探测（用户选定方案）：plan_segments 加可选 `fallback_silences`（默认 None=原行为）；
  transcribe_segmented 加 `fallback_db`（默认 -35，> noise_db 才二次探测），硬切前先用宽松候选找次优切点；
  CLI `--fallback-db`。再 +4 单测（None保原行为/宽松优先于硬切/区间外回退硬切/严格优先于宽松）

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| test_transcriber.py | 含新 13 测 | 全过 | 22 passed（原 9 + 新 13） | ✅ |
| 全量 pytest | 含本阶段 | 全过 | 133 passed（原 120 + 13） | ✅ |
| ruff check . | 全仓 | 无告警 | All checks passed | ✅ |
| CLI 两新参数 | subtitle -h | 均出现 | --segment-overlap / --fallback-db 已注册 | ✅ |
| plan_segments 5 契约 | 原测试 | 不破坏 | 全过（fallback 默认 None 行为不变） | ✅ |

## 真实运行已确认（250min 直播，RTX 5060，2026-06-17）
- 模型名已在开头打印（🤖 使用模型: ...）；CUDA/加载正常
- overlap 去重生效：第 2 段日志「取数 299.6~608.2s」左边界 = 301.6 - 2.0，垫料按预期外扩
- 二次宽松探测生效：严格 -45dB 找到 2967 段、宽松 -35dB 找到 2959 段，规划为 50 段
- generation_config.json 实锤 is_multilingual=true / 语言槽 null → 已放开 language=ja 修正
- **静音切分实测健康**：50 段 ÷ 15050s ≈ 301s/段 → 几乎全在静音切、基本无硬切；末条 15047.9s≈全长 → 覆盖完整
- **SRT 实测发现 NUL bug**：文件中段 [19398..32432) 整块 13034 个 NUL，后接旧序号 358/45min 内容
  → 上次运行被强杀留残留、本次 "w" 覆盖未截断尾部。SrtWriter.__init__ 加 truncate(0) 兜底（45 测过）
- **19% 字幕被 15s 钳满**：根因非静音切分，是长段内部 chunk 没补句末标点 → segment_sentences 无切点。待后续优化
- 用户选定「切点选窗口内最长静音」：plan_segments 改用 _longest_silence_midpoint（最长停顿=最可能语句间隙），
  并列取靠前者；严格→宽松→硬切三级兜底不变。+6 单测（最长非最早/并列靠前/窗口内最长/窗口外排除/空窗None/边界开区间）

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 全量 pytest（最终） | 含本阶段全部 | 全过 | 139 passed（原 120 + 19 新） | ✅ |
| ruff check .（最终） | 全仓 | 无告警 | All checks passed | ✅ |
| plan_segments 5 原契约 | 原测试 | 改最长静音后不破坏 | 全过（各窗口仅 1 个合格静音） | ✅ |

## 真实运行 #2 已确认（250min 直播，RTX 5060，2026-06-17，重跑）
- **NUL bug 已修**：重跑后 SRT 含 0 个 NUL 字节（truncate 生效）
- **overlap 去重无瑕**：1572 条字幕时间倒退 0 条、末条 15049.8s≈全长 → 接缝无重叠无丢失
- **最长静音切点生效**：段数 50→33，段长 243~599s 浮动（原固定 ~301s）= 等到窗口内最长停顿才切
- **18.2% 字幕仍被 15s 钳满（286 条）**：确认根因 = chunk 内部句末标点（。？！）断不开
- 用户选定「按句末标点切开」：新增 _split_internal_punctuation 预处理——把含内部句末标点的 chunk
  拆成多 Word、时间戳按字符比例分配，使 segment_sentences 能在句中标点断句。时间戳缺失则原样透传。
  +7 单测（拆分/多标点/标点在末尾不拆/无标点/缺时间戳/保留普通词/端到端句中断句）
- **模拟修复效果（拿真实 SRT 跑 _split_internal_punctuation）**：1572→3706 片段；
  拆分后仍 ≥14.9s 的片段仅 5 个、且其内部已无句末标点 → 18.2% 被钳实质降到 ~0.3% 且剩余合理

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 全量 pytest（最终） | 含本阶段全部 | 全过 | 146 passed（原 120 + 26 新） | ✅ |
| ruff check .（最终） | 全仓 | 无告警 | All checks passed | ✅ |
| 原 45 subtitle 契约 | 原测试 | 加句中断句后不破坏 | 全过（原测试句末标点均为独立 Word，不触发内部拆分） | ✅ |

## 未在本会话验证（需 GPU + 真实长音频）
- 放开 language=ja 后唱歌/BGM 段的语种误判是否消除、质量是否提升（改动后未重跑）
- overlap 去重在真实唱歌段的去吞字/去重复效果（仅纯逻辑单测覆盖归属判定）
- 切点选最长静音后真实分段质量（仅单测验证选择逻辑）
- NUL bug 修复：需用户删旧 output.srt 重跑确认干净输出
- 19% 被钳问题未动（句中标点断句优化，待用户决定是否做）
- slice_audio 新 ffmpeg 参数在本机 ffmpeg 的实际切段精度

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

## 阶段 9：CLI 别名 + pydantic 配置校验 + 发布 v1.3.1（2026-06-18）
- CLI：5 子命令加简写别名 dl/srt/tr/sep/sum，sum→规范名 summarize；分发从 if/elif 改 set_defaults(func=...)
  （实测确认：用别名调用时 args.command 拿到的是别名本身，故必须改 func 绑定才能正确路由）
- pydantic：prompts 配置改用 PromptPreset/PromptsConfig 校验，坏配置（缺/空 system、presets 空）加载时即报错；
  resolve/preset_names 收进模型，消除 load_presets 重复调用。决策：不引入 orjson/ujson（JSON 全冷路径，收益≈0）
- deepseek 抽 _build_payload 静态方法；全 9 模块补 __all__、补全类型标注；Word/Sentence 热路径 TypedDict 不动
- 死分支清理：_split_internal_punctuation 的 bounds 三元 else 恒不执行，简化为 [*meaningful, total]
- 交付：版本号 1.3.0→1.3.1、全量 148 passed、ruff 干净、README 展示别名
- git：dev 历史 6 个提交曾重写为 conventional 风格（备份分支验证零内容差异后推送）；
  PR #8/#9/#10 已合入 main（main 顶端 14f87c2）；发布 GitHub release **v1.3.1**（tag 指向 main 顶端，标记 Latest）

| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 全量 pytest（v1.3.1） | 含 pydantic 校验测 | 全过 | 148 passed | ✅ |
| ruff check . | 全仓 | 无告警 | All checks passed | ✅ |
| __all__ 名称解析 | 9 模块脚本校验 | 无缺失 | 全部解析 OK | ✅ |
| CLI 别名路由 | 规范名+别名 | 同一 handler | 5 组全 OK | ✅ |

## 当前状态（2026-06-18）
- main = dev 内容一致，已发布 **v1.3.1**（https://github.com/kanbereina/KITS/releases/tag/v1.3.1）
- 仍待 GPU 环境验证：language=ja 对唱歌段改善、overlap 去重真实接缝效果、句中断句端到端观感

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
