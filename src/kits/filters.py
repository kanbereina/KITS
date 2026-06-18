"""字幕后处理过滤：剔除游戏内系统播报 / 技能语音等非主播人声。

纯逻辑模块，不依赖 torch / 网络，可独立单测。

转录直播游戏时，麦克风会混入大量游戏音——VALORANT 的系统播报
（「残り1名」「ディフェンダーの勝利」）和特工技能语音（「アラームボット配置」
「グレネード配置」）。这些不是主播说的话，对「直播字幕」是噪声，还会在翻译时白烧 token。

本模块内置各游戏的播报词表，采用「整条完全匹配」策略：仅当一条字幕
（归一化后）恰好等于词表中某项时才剔除，最大限度避免误删主播人声。
代价是混入人声的条目、以及词表未收录的转录变体会被漏掉。

默认不启用，由 CLI 的 --filter-game <游戏名> 指定（大小写不敏感、支持简写、
可多次传入合并多款游戏）。新增游戏只需在 GAME_CALLOUTS / _GAME_ALIASES 登记。
"""

from __future__ import annotations

from kits.subtitle import Sentence

__all__ = [
    "filter_sentences",
    "is_game_callout",
    "resolve_games",
    "supported_games",
]

# 归一化时从首尾剥离的标点（不含长音符 ー，否则 セントリー -> セントリ 会破坏匹配）
_EDGE_PUNCT = "。．.、，,!！?？…‥・「」『』【】（）()〜~　 \t\n"


def _normalize(text: str) -> str:
    """归一化用于完全匹配：删空白、首尾去标点、ASCII 转小写。

    长音符 ー 及假名/汉字/数字全部保留，确保只做形态归一、不做模糊匹配。
    """
    text = text.replace(" ", "").replace("　", "")
    text = text.strip(_EDGE_PUNCT)
    return text.lower()


# VALORANT 系统播报 / 特工技能语音。含转录常见变体（Whisper 对同一句的不同误写）。
# 仅收录「整条出现时几乎不可能是主播人声」的高特异性短语。
_RAW_CALLOUTS: tuple[str, ...] = (
    # KAY/O：闪光 / 知识碎片
    "アラームボット配置",
    "アラームボット展開",
    # Sage/KAY/O 等：手雷 / 道具（含转录变体）
    "グレネード配置",
    "グルネード配置",
    "グレネード配地",
    "グルーネードハイチ",
    # Killjoy：警报机器人 / 炮台
    "セントリー設置",
    "タレット展開",
    # Cypher：传感器 / 监视设备
    "サウンドセンサー設置",
    # 通用：装置 / 无人机
    "スパイクを設置",
    "ドローン展開",
    # 系统播报
    "残り1名",
    "敵残り1名",
    "最後の一人だ",
    "最後の1人だ",
    "お遊びはここまでだ",
    "対戦を確認",
    "マッチポイント",
    "オーバータイムだ",
    "ディフェンダーの勝利",
    "アタッカーの勝利",
    "アルティメットいけるぞ",
    "スティムビーコンだ",
    # 击杀播报：X でキャリアダウン（X = 区域 / 出生点）
    "aでキャリアダウン",
    "bでキャリアダウン",
    "cでキャリアダウン",
    "中央でキャリアダウン",
    "スポーンでキャリアダウン",
)

# 预归一化后的词表（模块加载时构建一次）
VALORANT_CALLOUTS: frozenset[str] = frozenset(_normalize(p) for p in _RAW_CALLOUTS)


def is_game_callout(text: str, callouts: frozenset[str] = VALORANT_CALLOUTS) -> bool:
    """判断整条字幕（归一化后）是否恰好等于某条游戏播报词。"""
    return _normalize(text) in callouts


def filter_sentences(
    sentences: list[Sentence], callouts: frozenset[str] = VALORANT_CALLOUTS
) -> tuple[list[Sentence], int]:
    """剔除整条匹配游戏播报词的字幕。

    返回 (保留的句子列表, 被删条数)。被删条不补时间轴，渲染 SRT 时序号自动重新连续。
    """
    kept: list[Sentence] = []
    removed = 0
    for s in sentences:
        if is_game_callout(s["text"], callouts):
            removed += 1
        else:
            kept.append(s)
    return kept, removed


# ---- 游戏注册与名称解析 ----

# 规范游戏名 -> 已归一化的播报词表。新增游戏在此登记。
GAME_CALLOUTS: dict[str, frozenset[str]] = {
    "valorant": VALORANT_CALLOUTS,
}

# 用户可输入的游戏名（全小写，含简写）-> 规范名。忽略大小写匹配。
_GAME_ALIASES: dict[str, str] = {
    "valorant": "valorant",
    "valo": "valorant",
}


def supported_games() -> list[str]:
    """返回所有可识别的游戏名输入（含简写），已排序，用于帮助 / 报错提示。"""
    return sorted(_GAME_ALIASES)


def resolve_games(names: list[str]) -> frozenset[str]:
    """把游戏名列表合并成一个播报词表。

    忽略大小写与首尾空白，支持简写（valo -> valorant），可重复传入并去重合并。
    遇到未知游戏名抛 ValueError 并列出当前支持的名字；空列表返回空词表。
    """
    merged: set[str] = set()
    for name in names:
        key = _GAME_ALIASES.get(name.strip().lower())
        if key is None:
            raise ValueError(
                f"不支持的游戏: {name!r}。当前支持: {', '.join(supported_games())}"
            )
        merged |= GAME_CALLOUTS[key]
    return frozenset(merged)
