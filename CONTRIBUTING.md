# 贡献指南

感谢参与 KITS 开发。本指南覆盖环境搭建、开发约定与提交流程。沟通与代码注释一律用中文。

## 目录

- [环境搭建](#环境搭建)
- [验证安装](#验证安装)
- [onnxruntime GPU 加速的两个坑](#onnxruntime-gpu-加速的两个坑)
- [开发约定](#开发约定)
- [提交前检查](#提交前检查)
- [Commit 规范](#commit-规范)
- [分支与 PR 流程](#分支与-pr-流程)
- [GPU 模块与协作分工](#gpu-模块与协作分工)

## 环境搭建

依赖与运行统一走 [uv](https://docs.astral.sh/uv/)（包管理 + 运行器）。

系统依赖：

- **uv** — `uv --version` 应有输出
- **ffmpeg** — 须在 PATH 中，`ffmpeg -version` 应有输出（合并 MP4 / 提取 MP3 / 音频切分都依赖它）
- **Nvidia 显卡驱动** — 转字幕、人声分离需要，`nvidia-smi` 应能看到显卡

克隆并同步依赖：

```bash
git clone git@github.com:kanbereina/KITS.git
cd KITS
uv sync          # 创建虚拟环境并装齐所有依赖（含 dev 组）
```

`uv sync` 会从自定义索引 `pytorch-cu128`（CUDA 12.8）装 PyTorch，以及 audio-separator（含 onnxruntime-gpu）、punctuators 等。**不要把 torch 换成 PyPI 默认源。**

## 验证安装

```bash
uv run kits --help                                                  # 看到子命令说明即安装成功
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"  # 应输出 CUDA: True
uv run pytest                                                       # 测试应全绿
```

## onnxruntime GPU 加速的两个坑

这是新人最容易踩坏、又最难自行排查的地方，务必了解。`.onnx`/MDX 人声分离模型走 onnxruntime，`.ckpt`/MDXC roformer 走 torch。

1. **别手动 `pip install` 动 onnxruntime。** `punctuators` 间接依赖 **CPU 版 `onnxruntime`**，会和 `audio-separator[gpu]` 的 `onnxruntime-gpu` 装进同一个 `onnxruntime/` 目录、CPU 版 dll 顶掉 GPU 版，导致 `CUDAExecutionProvider` 丢失、`.onnx` 分离静默退回 CPU（慢数倍）。`pyproject.toml` 的 `[tool.uv] override-dependencies` 已禁止 CPU 版被装入。**改动依赖后**需运行：

   ```bash
   uv sync --reinstall-package onnxruntime-gpu   # 恢复可能被覆盖的 GPU dll
   ```

2. **验证 onnxruntime 走的是 GPU：**

   ```bash
   uv run python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
   ```

   输出应包含 `CUDAExecutionProvider`。若只有 `CPUExecutionProvider`，参考上一条修复。

   > onnxruntime-gpu 需要 `cublasLt64_12.dll` / `cudnn64_9.dll` 等 CUDA12/cuDNN9 运行时；本项目不装独立 CUDA Toolkit，靠 `separator._expose_torch_cuda_dlls()` 在 import onnxruntime 前把 `torch/lib`（torch cu128 自带这些 dll）加进 DLL 搜索路径复用。

## 开发约定

项目按依赖方向严格分层，核心是把**纯逻辑**与**重依赖（torch / 网络）**解耦。新增代码请沿用：

- **纯逻辑模块**（`subtitle` / `filters` / `llm` / `summarizer` 纯函数部分）不依赖 torch / 网络，可独立单测。
- **重依赖延迟导入**：`transcriber` / `separator` / `punctuator` 等模块的 torch / transformers / audio-separator / punctuators / onnxruntime，一律在**函数内** import，不放模块顶层。这样无 GPU 环境也能 import 模块、跑纯逻辑测试，CI 也无需安装数 GB 重包。
- **LLM 调用**统一走 `llm.LLMClient`（OpenAI 兼容，默认 DeepSeek；`deepseek.DeepSeekClient` 为向后兼容别名），不要在 `translator` / `summarizer` 里重复写 HTTP / 鉴权。
- 更详细的架构说明见 [CLAUDE.md](CLAUDE.md)。

## 提交前检查

提交前必须保证以下两项全绿：

```bash
uv run ruff check .       # lint（自动修复：uv run ruff check --fix .）
uv run pytest             # 测试
```

Ruff 配置在 `pyproject.toml` 的 `[tool.ruff]`：行宽 120、规则集 `E` + `F` + `I`（pycodestyle + pyflakes + import 排序）。

> 单元测试均为纯逻辑、运行时只需 httpx + pydantic，几秒内跑完，无需 GPU。

## Commit 规范

使用中文 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<类型>(<可选范围>): <简短描述>
```

常用类型：`feat`（新功能）、`fix`（修 bug）、`refactor`（重构）、`chore`（杂务）、`docs`（文档）、`ci`（CI）、`test`（测试）。

示例：

```
feat(subtitle): 新增句中标点断句
fix(separator): 修复长音频分段合并的接缝吞字
docs: 补充 CONTRIBUTING 的 onnxruntime 排查步骤
```

## 分支与 PR 流程

- 从 `dev` 切功能分支：`feature/<简述>`（或 `fix/` `chore/` `docs/` 等前缀）。
- 完成后提 PR 合入 `dev`；积累稳定后由维护者合 `main` 并打 tag 发布。
- **不要直接推 `main`。** PR 需通过 CI（ruff + pytest）。
- PR 标题用 Conventional Commits 风格；描述写清改动动机、做法、验证方式。

## GPU 模块与协作分工

GitHub CI runner **没有 GPU**，无法验证需要真实 CUDA 的代码路径。提交时请按模块区分：

| 类别 | 模块 / 范围 | 谁能开发与验证 |
| --- | --- | --- |
| **纯逻辑（无需 GPU）** | `subtitle` / `filters` / `deepseek` / `translator` / `summarizer` 逻辑、`cli`、文档、测试、`prompts.json` | 任何人，本地 `pytest` 即可完整验证 |
| **需 CUDA GPU** | `transcriber` / `separator` / `punctuator`，以及端到端流水线改动 | 需有 CUDA 12.8 显卡的开发者 |

**改动了 GPU 模块的 PR**，因 CI 验不了，请在描述里附上**本地实跑结果**：显卡型号、音频时长，以及改动前后的对比（如字幕条数、被钳比例、接缝质量）。

带 `gpu-required` 标签的 Issue 需要真实显卡才能完成；`gpu-not-needed` 与 `good first issue` 适合任何协作者上手。
