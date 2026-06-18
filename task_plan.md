# 任务计划：KITS 重构 — 新增 separate / sum 子命令并抽公共 DeepSeek 客户端

## 目标
把 KITS CLI 从 3 个子命令（download / subtitle / translate）扩展为 5 个：新增
`separate`（人声分离）和 `sum`（DeepSeek 总结），同时把 translate 与 sum 共用的 DeepSeek
调用逻辑抽成公共客户端，保持现有分层（纯逻辑 / 重依赖解耦、延迟导入）与全部现有测试通过。

## 当前阶段
全部完成 ✅ — 已发布 v1.3.1 到 main（含阶段 7~9）。dev 与 main 内容一致。

## 各阶段

### 阶段 1：需求与发现
- [x] 通读现有 5 个模块 + 测试，建立基线（91 测试通过）
- [x] 与用户确认 4 个关键决策（见「已做决策」）
- [x] 研究 audio-separator：0.44.2，torch>=2.3/numpy>=2，[gpu]→onnxruntime-gpu，cuDNN9+CUDA12 与 cu128 一致，默认 BS-Roformer，output_single_stem='Vocals'
- **状态：** complete

### 阶段 2：规划与结构
- [x] 设计公共 `deepseek.py` 客户端接口（translator + summarizer 共用）
- [x] 设计 `separator.py`（audio-separator 封装，延迟导入）
- [x] 设计 `summarizer.py` + 提示词预设 JSON schema
- [x] 设计 CLI 两个新子命令 + subtitle/download 的 `--separate` 集成
- **状态：** complete

### 阶段 3：实现
- [x] 抽 `deepseek.py`，重构 `translator.py` 复用它（保持现有测试通过）
- [x] 写 `separator.py` + `separate` 子命令
- [x] 写 `summarizer.py` + 提示词预设 JSON + `sum` 子命令（后升级规范名 summarize）
- [x] subtitle/download 加 `--separate` 可选集成
- [x] 更新 pyproject.toml 依赖、__init__.py 导出
- **状态：** complete

### 阶段 4：测试与验证
- [x] 为 deepseek / summarizer / separator 纯逻辑补单测
- [x] 跑全量 pytest + ruff，确保无回归
- [x] CLI 参数解析冒烟测试
- **状态：** complete

### 阶段 5：交付
- [x] 更新 README.md 与 CLAUDE.md
- [x] 检查产物完整、总结交付
- **状态：** complete

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

## 阶段 7：外层静音/重叠分段修复（2026-06-17）
背景：用户反馈「kotoba-whisper-v2.2 适配不够，尤其静音区间/重叠音频」。经核查根因不在模型版本，
而在外层 plan_segments 静音分段：硬切无重叠 → 鹿乃唱歌长期无静音段被 600s 硬切拦腰截断，
接缝处吞字/重复；内层 HF pipeline 的 stride 滑窗只解决「单段内部」重叠，救不了「段与段接缝」。
（已确认：v2.2 自带 pipeline 强制 pyannote diarization + 整段处理 + 标点-时间戳分叉，不适合本项目，
现状「标准 pipeline + 手工 punctuator + 静音分段」方向正确，故只修外层分段，不换 trust_remote_code。）

- [x] **分段重叠去重（已完成）**：transcribe_segmented 给每段取数窗口两侧 pad overlap（默认 2s、CLI
      `--segment-overlap`，0 关闭），转录→_shift_words 对齐全局→_keep_core_words 按词中心裁回逻辑区间；
      首段左/末段右不设限。plan_segments 不动 → 5 个原测试契约全保住。新增 _word_center/_keep_core_words
      纯逻辑 + 9 个单测。
- [x] silence_db 偏严 → **区间内二次宽松探测**（用户选定）：plan_segments 加可选 `fallback_silences`
      （默认 None=原行为不变）；transcribe_segmented 加 `fallback_db`（默认 -35，> noise_db 才二次探测），
      严格阈值在某段探不到静音、本会硬切时改用宽松候选找次优切点，硬切降为最后兜底。CLI `--fallback-db`。
      +4 单测（None保持原行为/宽松候选优先于硬切/区间外回退硬切/严格候选优先）
- [x] slice_audio 改 `-ss {start} -i {file} -t {时长}`（弃 -to 绝对结束，跨 ffmpeg 版本语义稳定）
- [x] 迁移残留：transcriber.py 模块 docstring + pyproject 描述改为 kotoba-whisper-v2.2；
      language/task 保持注释但补「为何对单语蒸馏模型是无操作」的说明（不盲目放开，无 GPU 验不了）
- [x] 跑全量 pytest（133 passed，原 120 + 13 新）+ ruff（All checks passed）+ CLI 两参数冒烟 OK
- **状态：** complete（代码改动全部完成；真实长音频效果待用户在 GPU 环境验证）

### 关键约束（必须保住的现有测试契约 test_transcriber.py）
- plan_segments 返回的 (start, end) 当前语义是「输出覆盖区间」，5 个测试断言：①短音频整段 ②无缝覆盖[0,dur]
  ③相邻段 prev[1]==nxt[0] ④硬切落在 max_chunk ⑤静音中点切。**重叠去重不能破坏「逻辑输出区间无缝相接」**，
  故方案：plan_segments 仍返回无缝的「逻辑区间」，另出「带 pad 的取数窗口」给 slice_audio，
  转录后按逻辑区间过滤词 → 测试契约不变、接缝无缝。

## 阶段 8：句中标点断句 + 最长静音切点 + NUL 修复（2026-06-17，真实重跑驱动）
真实 250min 直播重跑实测发现 18.2% 字幕被 15s 钳满，根因 = chunk 内部句末标点断不开（非静音切分问题）。
- [x] 切点改「窗口内最长静音」：plan_segments 用 _longest_silence_midpoint（最长停顿=最可能语句间隙，并列取靠前）
- [x] 句中断句：subtitle 新增 _split_internal_punctuation，把含内部句末标点的 chunk 拆成多 Word、时间戳按字符比例分配
- [x] SrtWriter NUL 修复：__init__ 加 truncate(0)，兜底崩溃残留 + "w" 覆盖不截断尾部留下的 NUL 空洞
- [x] 移除 _split_internal_punctuation 的死分支（meaningful 已滤末尾切点，三元 else 恒不执行）
- [x] 实测：18.2%→~0.7% 被钳、时间倒退 0、NUL 0、覆盖完整；+13 单测，全量 148 passed
- **状态：** complete（已并入 main，PR #8/#9）

## 阶段 9：CLI 别名 + pydantic 配置校验 + 工程结构（2026-06-18）
- [x] 5 子命令加简写别名（dl/srt/tr/sep/sum），sum 升级规范名 summarize；分发改 set_defaults(func=...)
- [x] prompts 配置改用 pydantic 校验（PromptPreset/PromptsConfig），坏配置加载时即报错；resolve/preset_names 收进模型
- [x] deepseek 抽 _build_payload 静态方法；全模块补 __all__、补全类型标注
- [x] 决策：不引入 orjson/ujson（JSON 全是冷路径，收益≈0）；Word/Sentence 热路径 TypedDict 不动
- [x] 版本号 1.3.0→1.3.1，全量 148 passed、ruff 干净
- [x] 交付：README 展示别名、PR #10 合入 main、发布 GitHub release v1.3.1
- **状态：** complete

