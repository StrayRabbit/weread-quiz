"""PK 结算判读：优先本地奖励证据，胜负只看比赛比分。"""

from __future__ import annotations

import re

from PIL import Image

from .errors import AnswerUnavailable
from .usage import safe_log

# 只接受已经确认过的奖励档位，不把对战比分当作奖励。
_REWARD_RE = re.compile(r"(?:获得积分\s*)?\+\s*(\d+)")
_KNOWN_REWARD_TIERS = {"1", "3", "4", "5"}
_EXPLICIT_OUTCOMES = {
    "VICTORY": "胜利", "胜利": "胜利",
    "DEFEAT": "落败", "失败": "落败", "落败": "落败",
}


def verdict_from_score(score) -> str:
    """按比分判胜负：分数高的一方赢。比分未知返回空串。

    **奖励积分档位不能用来判胜负** —— 实测同一星级（1 星）既有 +2 也有
    +5，而 +1 的那一局是 125:60 的大胜。积分由别的东西决定，跟输赢无关。
    """
    if not score:
        return ""
    mine, theirs = score
    if mine is None or theirs is None:
        return ""
    if mine > theirs:
        return "胜利"
    if mine < theirs:
        return "落败"
    return "平局"


def read_result(image: Image.Image, vision=None, *, ocr=None, recorder=None,
                score=None) -> dict:
    """返回 {result, title, score}；本地证据可靠时不再请求图片模型。

    score 是最终比分 (我方, 对手) —— 胜负由它决定。
    """
    try:
        local = _parse_detail(_read_local_result(image, ocr))
    except Exception as exc:  # noqa: BLE001 - 本地不可用仍可走图片路径
        safe_log(recorder, f"结算本地 OCR 不可用（{type(exc).__name__}），回退图片模型")
        local = _parse_detail({})
    if local["score"] or local["result"]:
        safe_log(recorder, f"结算本地 OCR 命中：奖励积分 {local['score'] or '未读到'}，"
                 f"结果 {local['result'] or '未知'}")
        return _apply_score(local, score)
    if vision is None:
        return _apply_score(local, score)
    safe_log(recorder, "结算本地无可靠证据，使用图片模型")
    try:
        detail = vision.read_result_detail(image)
    except AnswerUnavailable as exc:
        # 决策模型读不了图 —— 结算页就这么读不出来。说清楚，别只丢个异常类型。
        safe_log(recorder, f"结算页读不了：{exc}（没有能读图的模型）")
        return _apply_score(local, score)
    except Exception as exc:  # noqa: BLE001 - 不记录可能带敏感上下文的异常正文
        safe_log(recorder, f"结算图片识别失败（{type(exc).__name__}）")
        return _apply_score(local, score)
    info = _parse_detail(detail)
    if not info["score"] and not info["result"]:
        safe_log(recorder, "结算图片结果未包含有效胜负或已知奖励积分")
    return _apply_score(info, score)


def _apply_score(info: dict, score) -> dict:
    """用比分定胜负；页面文字和比分打架时不硬报。"""
    verdict = verdict_from_score(score)
    if not verdict:
        return info
    if info["result"] and info["result"] != verdict:
        # 页面写的和比分不一致 —— 两个都不硬报，别猜。
        info["result"] = ""
        return info
    info["result"] = verdict
    return info


def _read_local_result(image: Image.Image, ocr) -> dict:
    """要求高置信度奖励标签及同一行右侧的 +N，避免误读比分或星级。"""
    reader = getattr(ocr, "_engine_run", None)
    if reader is None:
        return {}
    rows = []
    for box, text, confidence in reader(image) or []:
        if not 0.7 <= float(confidence) <= 1.0:
            continue
        text = re.sub(r"\s+", "", text)
        if not text:
            continue
        x0, x1 = min(p[0] for p in box), max(p[0] for p in box)
        y0, y1 = min(p[1] for p in box), max(p[1] for p in box)
        rows.append((text, x0, x1, y0, y1))
    if any(re.search(r"第\d+题|最后一题", row[0]) for row in rows):
        return {}

    rewards = set()
    for label, _, right, top, bottom in rows:
        if label.startswith("获得积分"):
            combined = _REWARD_RE.fullmatch(label)
            if combined and combined.group(1) in _KNOWN_REWARD_TIERS:
                rewards.add(combined.group(1))
            elif label == "获得积分":
                for text, x0, _, y0, y1 in rows:
                    reward = _REWARD_RE.fullmatch(text)
                    same_row = abs((top + bottom - y0 - y1) / 2) <= max(bottom - top, y1 - y0) * 0.75
                    adjacent = right - 4 <= x0 <= right + image.width * 0.25
                    if (reward and same_row and adjacent
                            and reward.group(1) in _KNOWN_REWARD_TIERS):
                        rewards.add(reward.group(1))
    verdicts = {_EXPLICIT_OUTCOMES[text.upper()] for text, *_ in rows
                if text.upper() in _EXPLICIT_OUTCOMES}
    if len(rewards) > 1 or len(verdicts) > 1:
        return {}
    return {
        "result": next(iter(verdicts), ""),
        "title": next((text for text, *_ in rows if re.fullmatch(r".+[·・]\d+等", text)), ""),
        "score": f"+{next(iter(rewards))}" if rewards else "",
    }


def _parse_detail(detail) -> dict:
    """两条识别路径共用相同的档位与冲突规则。"""
    info: dict = {"result": "", "title": "", "score": ""}
    if not isinstance(detail, dict):
        return info
    info["title"] = str(detail.get("title") or "").strip()
    verdict = str(detail.get("result") or "").strip().upper()
    info["result"] = _EXPLICIT_OUTCOMES.get(verdict, "")
    score = str(detail.get("score") or "").strip()
    reward = _REWARD_RE.fullmatch(score)
    tier = reward.group(1) if reward else ""
    if tier in _KNOWN_REWARD_TIERS:
        info["score"] = f"+{tier}"
    return info
