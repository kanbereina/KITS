"""命令行入口：音频 -> SRT 字幕。

后续可在此扩展子命令（如 twitch 下载、deepseek 总结）。
"""

from __future__ import annotations

import argparse

from kits.subtitle import segment_sentences, write_srt
from kits.transcriber import Transcriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将直播音频转换为完整句子的 SRT 字幕（Whisper large-v3-turbo）"
    )
    parser.add_argument("-i", "--input", required=True, help="输入音频文件（必填）")
    parser.add_argument("-o", "--output", default="subtitle.srt", help="输出 SRT 文件")
    parser.add_argument("--language", default="japanese", help="识别语言")
    parser.add_argument("--beams", type=int, default=1, help="beam search 数量(1=贪心,更快)")
    parser.add_argument("--max-gap", type=float, default=0.7, help="判定断句的最大停顿(秒)")
    parser.add_argument("--max-chars", type=int, default=60, help="单条字幕最大字符数")
    parser.add_argument("--max-duration", type=float, default=15.0, help="单条字幕最大时长(秒)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("=" * 60)
    print("🎤 Whisper 语音识别 + SRT 字幕生成")
    print("=" * 60)

    transcriber = Transcriber()
    words = transcriber.transcribe(args.input, language=args.language, beams=args.beams)

    print("\n" + "=" * 60)
    print("🎬 生成 SRT 字幕")
    print("=" * 60)
    sentences = segment_sentences(
        words,
        max_gap=args.max_gap,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
    )
    write_srt(sentences, args.output)

    print(f"\n✅ SRT 字幕已保存到: {args.output}")
    print(f"📊 共 {len(sentences)} 条字幕")
    print("\n📝 预览前10条字幕:")
    for i, sent in enumerate(sentences[:10], 1):
        preview = sent["text"][:50] + ("..." if len(sent["text"]) > 50 else "")
        print(f"{i:3d}. [{sent['start']:6.1f}s -> {sent['end']:6.1f}s] {preview}")
    print("\n💡 提示: 可以直接将 SRT 文件拖入播放器或视频编辑软件使用")


if __name__ == "__main__":
    main()
