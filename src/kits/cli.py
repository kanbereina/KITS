"""命令行入口：子命令 download / subtitle / translate / separate / sum。

download:  下载 Twitch 直播 -> 合并 MP4 -> 可选 MP3 / SRT
subtitle:  已有音频 -> SRT 字幕（可选 --separate 先分离人声）
translate: 日语 SRT -> 中文 SRT（DeepSeek 翻译）
separate:  从音频分离出人声（audio-separator）
sum:       已有 SRT -> AI 总结（DeepSeek，提示词走 JSON 预设）
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
    write_srt,
)


def _add_subtitle_args(parser: argparse.ArgumentParser) -> None:
    """断句相关参数，download 和 subtitle 子命令共用。"""
    parser.add_argument("--language", default="japanese", help="识别语言")
    parser.add_argument("--beams", type=int, default=1, help="beam search 数量(1=贪心,更快)")
    parser.add_argument("--max-gap", type=float, default=0.7, help="判定断句的最大停顿(秒)")
    parser.add_argument("--max-chars", type=int, default=60, help="单条字幕最大字符数")
    parser.add_argument("--max-duration", type=float, default=15.0, help="单条字幕最大时长(秒)")
    # 长音频分段转录（按静音切分）。短音频会自动整段转录。
    parser.add_argument("--target-chunk", type=float, default=300.0, help="分段目标时长(秒)")
    parser.add_argument("--max-chunk", type=float, default=600.0, help="单段硬上限时长(秒)")
    parser.add_argument("--silence-db", type=float, default=-30.0, help="静音判定响度阈值(dB)")
    parser.add_argument("--min-silence", type=float, default=0.5, help="最短静音时长(秒)")
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
        help="人声分离模型文件名(默认 BS-Roformer)，仅在 --separate 时生效",
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
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download 子命令 ---
    dl = sub.add_parser("download", help="下载 Twitch 直播并合并为 MP4")
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

    # --- subtitle 子命令 ---
    st = sub.add_parser("subtitle", help="把已有音频转成 SRT 字幕")
    st.add_argument("-i", "--input", required=True, help="输入音频文件(必填)")
    st.add_argument("-o", "--output", default="subtitle.srt", help="输出 SRT 文件")
    _add_subtitle_args(st)

    # --- translate 子命令 ---
    tr = sub.add_parser("translate", help="把日语 SRT 翻译成中文 SRT(DeepSeek)")
    tr.add_argument("-i", "--input", required=True, help="输入 SRT 字幕文件(必填)")
    tr.add_argument("-o", "--output", default=None, help="输出 SRT(默认在原名后加 .zh)")
    tr.add_argument("--api-key", default=None, help="DeepSeek API Key(默认读环境变量 DEEPSEEK_API_KEY)")
    tr.add_argument("--model", default="deepseek-chat", help="DeepSeek 模型名")
    tr.add_argument("--batch-size", type=int, default=20, help="每批翻译的字幕条数")

    # --- separate 子命令 ---
    sp = sub.add_parser("separate", help="从音频分离出人声(audio-separator)")
    sp.add_argument("-i", "--input", required=True, help="输入音频文件(必填)")
    sp.add_argument("--dir", default="downloads", help="人声输出目录")
    sp.add_argument("--model", default=None, help="分离模型文件名(默认 BS-Roformer)")
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

    # --- sum 子命令 ---
    sm = sub.add_parser("sum", help="对已有 SRT 字幕用 AI 总结(DeepSeek)")
    sm.add_argument("-i", "--input", required=True, help="输入 SRT 字幕文件(必填)")
    sm.add_argument("-o", "--output", default=None, help="输出总结文件(默认在原名后加 .summary.md)")
    sm.add_argument("--api-key", default=None, help="DeepSeek API Key(默认读环境变量 DEEPSEEK_API_KEY)")
    sm.add_argument("--model", default="deepseek-chat", help="DeepSeek 模型名")
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
    separator = VocalSeparator(**kwargs)
    return separator.separate(audio_file)


def _audio_to_srt(audio_file: str, output_srt: str, args: argparse.Namespace) -> list[Sentence]:
    """转录音频并写出 SRT。延迟导入 transcriber 以避免无谓加载 GPU 栈。

    长音频按静音切分、分段转录，每段转完即断句并追加写盘（边转边出、可显示进度，
    中途中断时已写部分仍是合法 SRT）。切点落在静音处，不打断语句。
    若开启 --separate，先分离人声再转录。
    """
    from kits.transcriber import Transcriber

    audio_file = _maybe_separate(audio_file, args)

    transcriber = Transcriber()
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
    print("🎬 分段转录 + 生成 SRT 字幕")
    print("=" * 60)

    with SrtWriter(output_srt) as writer:
        segments_words = transcriber.transcribe_segmented(
            audio_file,
            language=args.language,
            beams=args.beams,
            target_chunk=args.target_chunk,
            max_chunk=args.max_chunk,
            noise_db=args.silence_db,
            min_silence=args.min_silence,
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
            )
            if callouts is not None:
                from kits.filters import filter_sentences

                sentences, removed = filter_sentences(sentences, callouts)
                filtered_total += removed
            total = writer.append(sentences)
            all_sentences.extend(sentences)
            print(f"  ✍️  已写入 {len(sentences)} 条，累计 {total} 条 -> {output_srt}")

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
    print("=" * 60)
    print("🎤 Whisper 语音识别 + SRT 字幕生成")
    print("=" * 60)
    _audio_to_srt(args.input, args.output, args)
    print("\n💡 提示: 可以直接将 SRT 文件拖入播放器或视频编辑软件使用")


def _run_translate(args: argparse.Namespace) -> None:
    from kits.translator import DeepSeekTranslator

    print("=" * 60)
    print("🌐 DeepSeek 字幕翻译（日语 -> 中文）")
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

    translator = DeepSeekTranslator(
        api_key=args.api_key, model=args.model, batch_size=args.batch_size
    )
    translated = translator.translate(sentences)
    write_srt(translated, str(output_path))

    print(f"\n✅ 中文字幕已保存到: {output_path}")
    print("\n📝 预览前10条:")
    for i, sent in enumerate(translated[:10], 1):
        preview = sent["text"][:50] + ("..." if len(sent["text"]) > 50 else "")
        print(f"{i:3d}. [{sent['start']:6.1f}s -> {sent['end']:6.1f}s] {preview}")
    print("\n💡 提示: 可以直接将 SRT 文件拖入播放器或视频编辑软件使用")


def _run_separate(args: argparse.Namespace) -> None:
    from kits.separator import VocalSeparator

    print("=" * 60)
    print("🎚️  人声分离（audio-separator）")
    print("=" * 60)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入音频文件: {input_path}")

    kwargs: dict = {
        "output_dir": args.dir,
        "output_format": args.format,
        "segment_size": args.segment_size,
        "overlap": args.overlap,
    }
    if args.model:
        kwargs["model_filename"] = args.model
    separator = VocalSeparator(**kwargs)
    vocals = separator.separate(str(input_path))

    print(f"\n✅ 人声音频已保存到: {vocals}")
    print("\n💡 提示: 可以把人声音频再交给 subtitle 子命令转字幕，降低 BGM 干扰")


def _run_sum(args: argparse.Namespace) -> None:
    from kits.summarizer import Summarizer, SummarizeError

    print("=" * 60)
    print("🤖 DeepSeek 字幕总结")
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
    if args.command == "download":
        _run_download(args)
    elif args.command == "subtitle":
        _run_subtitle(args)
    elif args.command == "translate":
        _run_translate(args)
    elif args.command == "separate":
        _run_separate(args)
    elif args.command == "sum":
        _run_sum(args)


if __name__ == "__main__":
    main()
