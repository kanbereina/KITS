# KITS - 鹿乃 Twitch 直播工具
# Copyright (C) 2026 KanbeReina
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""命令行入口：子命令 download / subtitle / translate / separate / summarize（各带简写别名）。

download:  下载 Twitch 直播 -> 合并 MP4 -> 可选 MP3 / SRT
subtitle:  已有音频 -> SRT 字幕（可选 --separate 先分离人声）
translate: 日语 SRT -> 中文 SRT（AI 翻译）
separate:  从音频分离出人声（audio-separator）
sum:       已有 SRT -> AI 总结（提示词走 JSON 预设）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from kits.subtitle import (
    Sentence,
    SrtWriter,
    parse_srt,
    segment_sentences,
)


def _make_bar(total: float | None, desc: str, unit: str = "it"):
    """创建统一风格的 tqdm 进度条。

    用 tqdm：进度条自动钉在终端底部，配合 tqdm.write() 打印日志时日志逐行上滚、
    进度条不被冲乱。total=None 时先建空条，待已知量程再 reset(total=...)。
    """
    from tqdm import tqdm

    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)


def _add_subtitle_args(parser: argparse.ArgumentParser) -> None:
    """断句相关参数，download 和 subtitle 子命令共用。"""
    # 转录模型白名单（延迟导入：transcriber 顶层已无重依赖，仅取常量不会拉起 torch）
    from kits.transcriber import MODEL_ID, SUPPORTED_MODELS

    parser.add_argument(
        "--model",
        default=MODEL_ID,
        choices=SUPPORTED_MODELS,
        help=f"转录模型(默认 {MODEL_ID})；可选 kotoba-whisper v2.2 / v2.0",
    )
    parser.add_argument("--language", default="japanese", help="识别语言")
    parser.add_argument("--beams", type=int, default=3, help="beam search 数量(默认3更稳;1=贪心,更快)")
    parser.add_argument("--max-gap", type=float, default=0.7, help="判定断句的最大停顿(秒)")
    parser.add_argument("--max-chars", type=int, default=60, help="单条字幕最大字符数")
    parser.add_argument("--max-duration", type=float, default=15.0, help="单条字幕最大时长(秒)")
    parser.add_argument(
        "--max-seconds-per-char",
        type=float,
        default=0.5,
        help="单条字幕每字符最大时长(秒)，治蒸馏模型 chunk 时间戳虚高(短句标成满 max-duration)；"
        "按字符数估算朗读时长上限、超出则收缩 end(只缩时长不改文本)。默认 0.5(日语≥2字/秒)；<=0 关闭",
    )
    # 长音频分段转录（按静音切分）。短音频会自动整段转录。
    parser.add_argument("--target-chunk", type=float, default=300.0, help="分段目标时长(秒)")
    parser.add_argument("--max-chunk", type=float, default=600.0, help="单段硬上限时长(秒)")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD 语音概率阈值(0~1)，高于此算人声；越大越严格(检出人声更少、间隙更多)。默认 0.5",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=0.5,
        help="短于此(秒)的停顿并入人声、不算切点间隙；越大间隙越少、段越接近整段。默认 0.5",
    )
    parser.add_argument(
        "--segment-overlap",
        type=float,
        default=2.0,
        help="分段取数窗口两侧的重叠垫料(秒)，给模型在接缝处留上下文、避免硬切吞字；"
        "转录后按词中心裁回，接缝无缝去重。0 关闭重叠",
    )
    parser.add_argument(
        "--filter-game",
        action="append",
        metavar="GAME",
        default=None,
        help="剔除指定游戏的系统播报/技能语音(整条完全匹配才删)，"
        "大小写不敏感、可多次指定，如 --filter-game valorant；目前支持: valorant(valo)",
    )
    # 转录前可选先分离人声（去掉 BGM / 唱歌干扰，提升识别精度）
    parser.add_argument(
        "--separate",
        action="store_true",
        help="转录前先用 audio-separator 分离人声（需安装 audio-separator）",
    )
    parser.add_argument(
        "--separate-model",
        default=None,
        help="人声分离模型文件名(默认 UVR-MDX-NET_Main_427.onnx)，仅在 --separate 时生效",
    )
    parser.add_argument(
        "--separate-segment-size",
        type=int,
        default=512,
        help="人声分离分块大小，越大越快越吃显存(默认 512)，仅在 --separate 时生效",
    )
    parser.add_argument(
        "--separate-overlap",
        type=float,
        default=0.1,
        help="人声分离分块重叠(0~1)，越小越快(默认 0.1)，仅在 --separate 时生效",
    )
    parser.add_argument(
        "--separate-segment-minutes",
        type=float,
        default=15.0,
        help="人声分离长音频切段时长(分钟)，防爆内存(默认 15；<=0 关闭)，仅在 --separate 时生效",
    )
    parser.add_argument(
        "--separate-output-bitrate",
        default=None,
        help="人声输出比特率(如 128k)，默认自动对齐原音频；无损格式忽略，仅在 --separate 时生效",
    )
    # 标点恢复（蒸馏模型 chunk 无标点时靠它断句）。默认开启。
    parser.add_argument(
        "--no-punctuate",
        dest="punctuate",
        action="store_false",
        help="关闭标点恢复（默认开启；模型本身已输出标点时可关）",
    )
    parser.add_argument(
        "--punct-model",
        default=None,
        help="标点恢复模型(默认 xlm-roberta 日语句读模型)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kits",
        description="鹿乃 Twitch 直播工具：下载直播、生成 SRT 字幕",
    )
    # 全局 --verbose：默认安静（压掉 transformers / onnxruntime / hf_hub 的刷屏日志与
    # 进度条），加 --verbose 放行全部底层日志便于开发调试。用共享 parent 让每个子命令
    # 都带上（可写在子命令末尾，如 kits subtitle -i x --verbose）。
    verbose_parent = argparse.ArgumentParser(add_help=False)
    verbose_parent.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="显示底层库(transformers/onnxruntime 等)的调试日志，默认隐藏",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download 子命令 ---
    dl = sub.add_parser(
        "download", aliases=["dl"], parents=[verbose_parent],
        help="下载 Twitch 直播并合并为 MP4",
    )
    dl.add_argument("url", help="TS 文件示例 URL，形如 https://.../chunked/1710.ts")
    dl.add_argument("-o", "--output", default="output", help="输出文件名(不含扩展名)")
    dl.add_argument("--dir", default="downloads", help="下载目录")
    dl.add_argument("--start", type=int, default=None, help="起始编号(默认 0，下载整场)")
    dl.add_argument("--end", type=int, default=None, help="结束编号(默认二分探测)")
    dl.add_argument("--concurrent", type=int, default=5, help="最大并发下载数")
    dl.add_argument("--keep-ts", action="store_true", help="保留临时 TS 文件")
    dl.add_argument("--mp3", action="store_true", help="额外导出 MP3 音频")
    dl.add_argument("--srt", action="store_true", help="额外生成 SRT 字幕(自动转录音频)")
    _add_subtitle_args(dl)
    dl.set_defaults(func=_run_download)

    # --- subtitle 子命令 ---
    st = sub.add_parser(
        "subtitle", aliases=["srt"], parents=[verbose_parent],
        help="把已有音频转成 SRT 字幕",
    )
    st.add_argument("-i", "--input", required=True, help="输入音频文件(必填)")
    st.add_argument("-o", "--output", default="subtitle.srt", help="输出 SRT 文件")
    _add_subtitle_args(st)
    st.set_defaults(func=_run_subtitle)

    # --- translate 子命令 ---
    tr = sub.add_parser(
        "translate", aliases=["tr"], parents=[verbose_parent],
        help="把日语 SRT 翻译成中文 SRT(AI)",
    )
    tr.add_argument("-i", "--input", required=True, help="输入 SRT 字幕文件(必填)")
    tr.add_argument("-o", "--output", default=None, help="输出 SRT(默认在原名后加 .zh)")
    tr.add_argument(
        "--api-key", default=None,
        help="LLM API Key(默认读环境变量 KITS_LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY)",
    )
    tr.add_argument(
        "--base-url", default=None,
        help="OpenAI 兼容端点 base_url(默认 DeepSeek，可指向 OpenAI/Ollama/vLLM 等；"
             "也可读环境变量 KITS_LLM_BASE_URL)",
    )
    tr.add_argument("--model", default="deepseek-chat", help="LLM 模型名(默认 deepseek-chat)")
    tr.add_argument("--batch-size", type=int, default=20, help="每批翻译的字幕条数")
    tr.set_defaults(func=_run_translate)

    # --- separate 子命令 ---
    sp = sub.add_parser(
        "separate", aliases=["sep"], parents=[verbose_parent],
        help="从音频分离出人声(audio-separator)",
    )
    sp.add_argument("-i", "--input", required=True, help="输入音频文件(必填)")
    sp.add_argument(
        "-o", "--output", default=None,
        help="输出人声文件路径(默认在 --dir 下生成 原名_(Vocals).格式)",
    )
    sp.add_argument("--dir", default="output", help="人声输出目录(未指定 -o 时生效)")
    sp.add_argument("--model", default=None, help="分离模型文件名(默认 UVR-MDX-NET_Main_427.onnx)")
    sp.add_argument(
        "--format", default="MP3", help="输出音频格式(WAV/MP3/FLAC 等，默认 MP3)"
    )
    sp.add_argument(
        "--segment-size",
        type=int,
        default=512,
        help="分块大小，越大分块越少、越快越吃显存(默认 512；显存紧张可降到 256)",
    )
    sp.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="MDX(.onnx) 模型分块重叠(0~1)，越小越快、接缝质量略降(默认 0.1)",
    )
    sp.add_argument(
        "--segment-minutes",
        type=float,
        default=15.0,
        help="长音频按此时长(分钟)切段逐段分离再合并，防爆内存(默认 15；<=0 关闭)",
    )
    sp.add_argument(
        "--output-bitrate",
        default=None,
        help="输出比特率(如 128k)，默认自动对齐原音频(向上取整到 2 的幂)；无损格式忽略",
    )
    sp.set_defaults(func=_run_separate)

    # --- summarize 子命令 ---
    sm = sub.add_parser(
        "summarize", aliases=["sum"], parents=[verbose_parent],
        help="对已有 SRT 字幕用 AI 总结",
    )
    sm.add_argument("-i", "--input", required=True, help="输入 SRT 字幕文件(必填)")
    sm.add_argument("-o", "--output", default=None, help="输出总结文件(默认在原名后加 .summary.md)")
    sm.add_argument(
        "--api-key", default=None,
        help="LLM API Key(默认读环境变量 KITS_LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY)",
    )
    sm.add_argument(
        "--base-url", default=None,
        help="OpenAI 兼容端点 base_url(默认 DeepSeek，可指向 OpenAI/Ollama/vLLM 等；"
             "也可读环境变量 KITS_LLM_BASE_URL)",
    )
    sm.add_argument("--model", default="deepseek-chat", help="LLM 模型名(默认 deepseek-chat)")
    sm.add_argument(
        "--preset",
        default=None,
        help="总结预设名(默认用配置里的 default)，如 timeline/summary/highlights/setlist",
    )
    sm.add_argument(
        "--prompt-file",
        default=None,
        help="自定义提示词 JSON 文件，覆盖内置预设",
    )
    sm.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="单块送审的最大字符数，超长字幕按此分块总结",
    )
    sm.set_defaults(func=_run_sum)

    return parser


def _maybe_separate(audio_file: str, args: argparse.Namespace) -> str:
    """若开启 --separate，则先分离人声并返回人声音频路径；否则原样返回。

    延迟导入 separator 以避免无谓加载重依赖栈。
    """
    if not getattr(args, "separate", False):
        return audio_file

    from kits.separator import VocalSeparator

    print("\n🎚️  转录前预处理：分离人声")
    out_dir = str(Path(audio_file).parent)
    kwargs: dict = {"output_dir": out_dir}
    if getattr(args, "separate_model", None):
        kwargs["model_filename"] = args.separate_model
    if getattr(args, "separate_segment_size", None) is not None:
        kwargs["segment_size"] = args.separate_segment_size
    if getattr(args, "separate_overlap", None) is not None:
        kwargs["overlap"] = args.separate_overlap
    if getattr(args, "separate_segment_minutes", None) is not None:
        kwargs["segment_minutes"] = args.separate_segment_minutes
    if getattr(args, "separate_output_bitrate", None) is not None:
        kwargs["output_bitrate"] = args.separate_output_bitrate
    separator = VocalSeparator(**kwargs)
    return separator.separate(audio_file)


def _audio_to_srt(audio_file: str, output_srt: str, args: argparse.Namespace) -> list[Sentence]:
    """转录音频并写出 SRT。延迟导入 transcriber 以避免无谓加载 GPU 栈。

    先用 VAD 规划分段（plan_audio），再逐段转录、每段转完即断句并追加写盘（边转边出、
    可显示进度，中途中断时已写部分仍是合法 SRT）。切点落在 VAD 探出的人声间隙处，不打断
    语句。若开启 --separate，先分离人声再转录。
    """
    from kits.transcriber import Transcriber

    audio_file = _maybe_separate(audio_file, args)

    transcriber = Transcriber(model_id=args.model)
    all_sentences: list[Sentence] = []
    filtered_total = 0

    # 解析待过滤的游戏词表（提前解析，无效游戏名在转录前就报错）
    callouts = None
    if args.filter_game:
        from kits.filters import resolve_games

        callouts = resolve_games(args.filter_game)
        print(f"🎮 已启用游戏播报过滤: {', '.join(args.filter_game)}")

    # 标点恢复器：蒸馏模型（如 kotoba）产出的 chunk 无句末标点，补标点后才能断句。
    # 默认开启，可用 --no-punctuate 关闭。提前实例化，整场复用一个模型。
    punctuator = None
    if getattr(args, "punctuate", True):
        from kits.punctuator import Punctuator

        kwargs: dict = {}
        if getattr(args, "punct_model", None):
            kwargs["model"] = args.punct_model
        punctuator = Punctuator(**kwargs)

    print("\n" + "=" * 60)
    print("🎬 ASR 字幕生成")
    print("=" * 60)

    # 模型提前加载，让其（含 transformers）加载日志在进度条启动前打完，避免冲乱进度条。
    transcriber.load()
    if punctuator is not None:
        punctuator.load()

    # 规划阶段：探时长 + （长音频）VAD 扫描 + 分段，在进度条创建之前完成。VAD 模型加载与
    # 全程扫描耗时可观，放进度条之前跑完，其日志才不会冲乱进度条（与模型提前 load 同理）。
    duration, segments = transcriber.plan_audio(
        audio_file,
        target_chunk=args.target_chunk,
        max_chunk=args.max_chunk,
        vad_threshold=args.vad_threshold,
        min_silence=args.min_silence,
    )

    # 转录进度条按音频秒数推进，量程已知（plan_audio 已探得 duration），直接满量程建条。
    # 段号/累计条数收进进度条 desc/postfix，不再刷屏。
    bar = _make_bar(total=duration, desc="🎬 转录进度", unit="s")
    with SrtWriter(output_srt) as writer, bar:
        segments_words = transcriber.transcribe_segments(
            audio_file,
            segments,
            duration,
            language=args.language,
            beams=args.beams,
            overlap=args.segment_overlap,
            bar=bar,
        )
        for words in segments_words:
            # 转录后、断句前补标点（时间戳不变），让 segment_sentences 能在句末标点处断句
            if punctuator is not None:
                words = punctuator.restore(words)
            sentences = segment_sentences(
                words,
                max_gap=args.max_gap,
                max_chars=args.max_chars,
                max_duration=args.max_duration,
                max_seconds_per_char=args.max_seconds_per_char,
            )
            if callouts is not None:
                from kits.filters import filter_sentences

                sentences, removed = filter_sentences(sentences, callouts)
                filtered_total += removed
            total = writer.append(sentences)
            all_sentences.extend(sentences)
            # 累计条数收进进度条右侧（postfix），与 transcriber 设的段号描述（desc）互不覆盖，
            # 不再每段刷一行。需要逐段明细时加 --verbose（底层日志一并放行）。
            bar.set_postfix_str(f"累计 {total} 条")

    if callouts is not None:
        print(f"  🧹 已过滤游戏播报 {filtered_total} 条")
    if not all_sentences:
        raise RuntimeError("未获取到任何字幕内容")
    _print_preview(all_sentences, output_srt)
    return all_sentences


def _print_preview(sentences: list[Sentence], output_srt: str) -> None:
    print(f"\n✅ SRT 字幕已保存到: {output_srt}")
    print(f"📊 共 {len(sentences)} 条字幕")
    print("\n📝 预览前10条字幕:")
    for i, sent in enumerate(sentences[:10], 1):
        preview = sent["text"][:50] + ("..." if len(sent["text"]) > 50 else "")
        print(f"{i:3d}. [{sent['start']:6.1f}s -> {sent['end']:6.1f}s] {preview}")


def _run_download(args: argparse.Namespace) -> None:
    from kits.downloader import TwitchDownloader

    print("=" * 60)
    print("🐙 Twitch 直播下载")
    print("=" * 60)

    downloader = TwitchDownloader(download_dir=args.dir, max_concurrent=args.concurrent)
    # 生成 SRT 必然需要音频，故强制提取 MP3
    need_mp3 = args.mp3 or args.srt
    outputs = asyncio.run(
        downloader.download_from_url(
            args.url,
            output_name=args.output,
            start_num=args.start,
            end_num=args.end,
            keep_ts=args.keep_ts,
            extract_audio=need_mp3,
        )
    )

    if args.srt:
        mp3_path = outputs["mp3"]
        srt_path = str(Path(mp3_path).with_suffix(".srt"))
        _audio_to_srt(str(mp3_path), srt_path, args)

    print("\n✨ 产物:")
    for kind, path in outputs.items():
        print(f"   - {kind}: {path}")
    print("\n💡 提示: 可以直接将 SRT 文件拖入播放器或视频编辑软件使用")


def _run_subtitle(args: argparse.Namespace) -> None:
    _audio_to_srt(args.input, args.output, args)
    print("\n💡 提示: 可以直接将 SRT 文件拖入播放器或视频编辑软件使用")


def _run_translate(args: argparse.Namespace) -> None:
    from kits.translator import LLMTranslator

    print("=" * 60)
    print("🌐 AI 字幕翻译（日语 -> 中文）")
    print("=" * 60)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入字幕文件: {input_path}")

    sentences = parse_srt(input_path.read_text(encoding="utf-8"))
    if not sentences:
        raise RuntimeError(f"未能从 {input_path} 解析出任何字幕")
    print(f"📄 已读取 {len(sentences)} 条字幕")

    # 默认输出在原名后插入 .zh，如 live.srt -> live.zh.srt
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_suffix(f".zh{input_path.suffix}")
    )

    translator = LLMTranslator(
        api_key=args.api_key, model=args.model, batch_size=args.batch_size,
        base_url=args.base_url,
    )

    # 边翻译边写盘：批次严格串行（无并发），故产出顺序即字幕顺序，可安全增量写。
    # 中途中断时已写入部分仍是合法 SRT。进度用 tqdm 进度条展示（按已译条数推进）。
    total = len(sentences)
    preview: list[Sentence] = []
    bar = _make_bar(total=total, desc="🌐 翻译进度", unit="条")
    with SrtWriter(str(output_path)) as writer, bar:
        for batch_result, done, _total in translator.translate_iter(sentences):
            writer.append(batch_result)
            if len(preview) < 10:
                preview.extend(batch_result[: 10 - len(preview)])
            bar.update(done - bar.n)  # done 是累计已译条数，tqdm 收增量

    print(f"\n✅ 中文字幕已保存到: {output_path}（共 {total} 条）")
    print("\n📝 预览前10条:")
    for i, sent in enumerate(preview, 1):
        text = sent["text"][:50] + ("..." if len(sent["text"]) > 50 else "")
        print(f"{i:3d}. [{sent['start']:6.1f}s -> {sent['end']:6.1f}s] {text}")
    print("\n💡 提示: 可以直接将 SRT 文件拖入播放器或视频编辑软件使用")


def _run_separate(args: argparse.Namespace) -> None:
    from kits.separator import VocalSeparator

    print("=" * 60)
    print("🎚️  人声分离（audio-separator）")
    print("=" * 60)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入音频文件: {input_path}")

    # output_dir 仅用于未指定 -o 时派生默认输出名；中间产物统一进 .cache（见
    # separator），不再借用输出目录，故这里直接用 --dir。
    kwargs: dict = {
        "output_dir": args.dir,
        "output_format": args.format,
        "segment_size": args.segment_size,
        "overlap": args.overlap,
        "segment_minutes": args.segment_minutes,
        "output_bitrate": args.output_bitrate,
    }
    if args.model:
        kwargs["model_filename"] = args.model
    separator = VocalSeparator(**kwargs)
    vocals = separator.separate(str(input_path), output_path=args.output)

    print(f"\n✅ 人声音频已保存到: {vocals}")
    print("\n💡 提示: 可以把人声音频再交给 subtitle 子命令转字幕，降低 BGM 干扰")


def _run_sum(args: argparse.Namespace) -> None:
    from kits.summarizer import SummarizeError, Summarizer

    print("=" * 60)
    print("🤖 AI 字幕总结")
    print("=" * 60)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入字幕文件: {input_path}")

    sentences = parse_srt(input_path.read_text(encoding="utf-8"))
    if not sentences:
        raise RuntimeError(f"未能从 {input_path} 解析出任何字幕")
    print(f"📄 已读取 {len(sentences)} 条字幕")

    try:
        summarizer = Summarizer(
            api_key=args.api_key,
            model=args.model,
            preset=args.preset,
            prompt_file=args.prompt_file,
            max_chars=args.max_chars,
            base_url=args.base_url,
        )
        summary = summarizer.summarize(sentences)
    except SummarizeError as e:
        raise SystemExit(f"❌ 总结失败: {e}") from e

    # 默认输出在原名后插入 .summary，扩展名换成 .md
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_suffix(".summary.md")
    )
    output_path.write_text(summary, encoding="utf-8")

    print(f"\n✅ 总结已保存到: {output_path}")
    print("\n📝 总结预览:")
    preview = summary[:500] + ("..." if len(summary) > 500 else "")
    print(preview)


def main(argv: list[str] | None = None) -> None:
    # Windows 控制台默认 GBK，输出里的 emoji（🎚️ 等）会触发 UnicodeEncodeError
    # 让整条流水线崩溃。统一把标准流切到 UTF-8（errors=replace 兜底）。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    # 按 --verbose 统一设置底层库日志噪音（默认安静）。transcriber 顶层无重依赖，
    # 仅导入此函数不会拉起 torch。
    from kits.transcriber import configure_logging

    configure_logging(verbose=getattr(args, "verbose", False))
    # 各子命令用 set_defaults(func=...) 绑定处理函数，别名与规范名都能正确分发
    args.func(args)


if __name__ == "__main__":
    main()
