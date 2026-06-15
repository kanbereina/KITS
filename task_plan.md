# 任务计划：KITS 重构 — 新增 separate / sum 子命令并抽公共 DeepSeek 客户端

## 目标
把 KITS CLI 从 3 个子命令（download / subtitle / translate）扩展为 5 个：新增
`separate`（人声分离）和 `sum`（DeepSeek 总结），同时把 translate 与 sum 共用的 DeepSeek
调用逻辑抽成公共客户端，保持现有分层（纯逻辑 / 重依赖解耦、延迟导入）与全部现有测试通过。

## 当前阶段
全部完成 ✅（含阶段 6 kotoba 适配 + 标点恢复）

## 各阶段

### 阶段 1：需求与发现
- [x] 通读现有 5 个模块 + 测试，建立基线（91 测试通过）
- [x] 与用户确认 4 个关键决策（见「已做决策」）
- [x] 研究 audio-separator：0.44.2，torch>=2.3/numpy>=2，[gpu]→onnxruntime-gpu，cuDNN9+CUDA12 与 cu128 一致，默认 BS-Roformer，output_single_stem='Vocals'
- **状态：** complete

### 阶段 2：规划与结构
- [ ] 设计公共 `deepseek.py` 客户端接口（translator + summarizer 共用）
- [ ] 设计 `separator.py`（audio-separator 封装，延迟导入）
- [ ] 设计 `summarizer.py` + 提示词预设 JSON schema
- [ ] 设计 CLI 两个新子命令 + subtitle/download 的 `--separate` 集成
- **状态：** pending

### 阶段 3：实现
- [ ] 抽 `deepseek.py`，重构 `translator.py` 复用它（保持现有测试通过）
- [ ] 写 `separator.py` + `separate` 子命令
- [ ] 写 `summarizer.py` + 提示词预设 JSON + `sum` 子命令
- [ ] subtitle/download 加 `--separate` 可选集成
- [ ] 更新 pyproject.toml 依赖、__init__.py 导出
- **状态：** pending

### 阶段 4：测试与验证
- [ ] 为 deepseek / summarizer / separator 纯逻辑补单测
- [ ] 跑全量 pytest + ruff，确保无回归
- [ ] CLI 参数解析冒烟测试
- **状态：** pending

### 阶段 5：交付
- [ ] 更新 README.md 与 CLAUDE.md
- [ ] 检查产物完整、总结交付
- **状态：** pending

## 关键问题
1. audio-separator 是否支持 CUDA 12.8（onnxruntime-gpu）？默认模型是哪个？→ 阶段 1 研究
2. 公共 DeepSeek 客户端的边界：只抽 HTTP+Key，还是连批处理也抽？→ 倾向只抽 HTTP+Key+错误，批处理留各自模块
3. 提示词预设 JSON 放包内默认 + 用户 `--prompt-file` 覆盖？→ 倾向是

## 已做决策
| 决策 | 理由 |
|------|------|
| 同时重构现有代码 | 用户选择；把 translate/sum 的 DeepSeek 调用抽公共客户端，避免重复 |
| separate 用 audio-separator (UVR/MDX) | 用户选择；模型可选多，ONNX 后端 |
| separate = 独立命令 + 可选集成 | 用户选择；separate 子命令独立产出人声，subtitle/download 加 --separate 转录前预处理 |
| sum 用 JSON 提示词预设 | 用户选择；预置多种提示词预设，后续调用直接选预设输出 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
|      | 1       |         |

## 备注
- 沿用现有约定：纯逻辑无 torch 依赖可单测；重依赖（torch/网络）走函数内延迟导入
- 外部内容（网页/API 结果）只写入 findings.md，不写入本文件
- 决策前重读本计划

## 阶段 6：模型适配 kotoba-whisper-v2.2 + 标点恢复（2026-06-15 续）
背景：用户为「日语识别更准」换用 kotoba-whisper-v2.2（蒸馏，2 层解码器）。
- [x] 定位崩溃根因：alignment_heads 继承 large-v3（引用第 25 层）但模型仅 2 层 → word 级时间戳 IndexError
- [x] 改 transcriber `_transcribe_file` 用 `return_timestamps=True`（chunk 级），崩溃消除、112 测试仍过
- [x] 实测暴露代价：chunk 无句末标点 + 时间戳连续 → segment_sentences 标点/停顿规则失效，多条被 max_duration 硬切
- [x] 用户决策：接入官方标点模型（punctuators / PunctCapSegModelONNX）恢复断句
- [x] 装 punctuators 0.0.7，实测逐 chunk 批量标点化：infer 返回 list[list[str]]，质量好（。！？ 位置合理），副带 ・ 插入（无害）
- [x] 新建 punctuator.py（重依赖，延迟导入）：restore(list[Word]) 批量补标点、时间戳不变，含 unk 回退 + 已标点跳过
- [x] cli._audio_to_srt 在 segment_sentences 前接入标点恢复（延迟导入，整场复用一个模型）
- [x] 加 --no-punctuate（默认开）/ --punct-model + 补 8 个纯逻辑单测（假模型替身）+ 跑全量 120 过 + 更新文档
- [x] 端到端实测（120s）：补标点后字幕 7→19 条、被 15s 硬切 5→2 条，断句在句末标点处自然切开
- **状态：** complete
