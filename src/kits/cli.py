"""命令行入口：子命令 download / subtitle / translate。

download:  下载 Twitch 直播 -> 合并 MP4 -> 可选 MP3 / SRT
subtitle:  已有音频 -> SRT 字幕
translate: 日语 SRT -> 中文 SRT（DeepSeek 翻译）
后续可在此扩展 summarize 子命令（DeepSeek 总结）。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from kits.subtitle import Sentence, parse_srt, segment_sentences, write_srt


def _add_subtitle_args(parser: argparse.ArgumentParser) -> None:
    """断句相关参数，download 和 subtitle 子命令共用。"""
    parser.add_argument("--language", default="japanese", help="识别语言")
    parser.add_argument("--beams", type=int, default=1, help="beam search 数量(1=贪心,更快)")
    parser.add_argument("--max-gap", type=float, default=0.7, help="判定断句的最大停顿(秒)")
    parser.add_argument("--max-chars", type=int, default=60, help="单条字幕最大字符数")
    parser.add_argument("--max-duration", type=float, default=15.0, help="单条字幕最大时长(秒)")


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

    return parser


def _audio_to_srt(audio_file: str, output_srt: str, args: argparse.Namespace) -> list[Sentence]:
    """转录音频并写出 SRT。延迟导入 transcriber 以避免无谓加载 GPU 栈。"""
    from kits.transcriber import Transcriber

    words = Transcriber().transcribe(audio_file, language=args.language, beams=args.beams)

    print("\n" + "=" * 60)
    print("🎬 生成 SRT 字幕")
    print("=" * 60)
    sentences = segment_sentences(
        words,
        max_gap=args.max_gap,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
    )
    write_srt(sentences, output_srt)
    _print_preview(sentences, output_srt)
    return sentences


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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        _run_download(args)
    elif args.command == "subtitle":
        _run_subtitle(args)
    elif args.command == "translate":
        _run_translate(args)


if __name__ == "__main__":
    main()
