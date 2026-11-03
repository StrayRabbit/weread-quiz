"""逐层调试读题链路：像素层 → 按钮 → OCR 两路 → 门控 → 模型。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

# src/ 在 sys.path 里 —— 这个脚本从项目根用 .venv/Scripts/python.exe 直接跑，
# 不是从已安装的 wheel（安装的包里没有 tools/）。与 tools/ 下其他脚本一致。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weread_quiz import buttons, daily_buttons, daily_screen, screen  # noqa: E402
from weread_quiz.config import AppConfig  # noqa: E402
from weread_quiz.daily_ocr import (  # noqa: E402
    DailyOcrReader,
    find_option_bands as daily_option_bands,
    has_highlighted_button,
)
from weread_quiz.daily_vision import DAILY_PROMPT, build_daily_vision_service  # noqa: E402
from weread_quiz.errors import WqError  # noqa: E402
from weread_quiz.models import Rect  # noqa: E402
from weread_quiz.ocr_core import OcrDecision, OcrRead  # noqa: E402
from weread_quiz.pk_ocr import TEXT_PROMPT, RapidOcrReader  # noqa: E402
from weread_quiz.vision import PROMPT_PK, build_vision_service  # noqa: E402


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _ms(seconds: float) -> float:
    """秒转毫秒，保留一位小数。"""
    return round(seconds * 1000, 1)


def _jsonable(value: Any) -> Any:
    """把 dataclass / Rect / 异常转成能 json.dump 的结构。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Rect):
        return value.as_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _timed(call, *args, **kwargs):
    """跑一个函数，返回 (结果, 毫秒)。异常照常上抛给调用方处理。"""
    started = time.monotonic()
    result = call(*args, **kwargs)
    return result, _ms(time.monotonic() - started)


def _exc(exc: BaseException) -> dict:
    """把异常转成结构化记录。"""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=6),
    }


def _read_to_dict(read: OcrRead | None) -> dict | None:
    """OcrRead → dict。读不出是 None（不是空 read）。"""
    if read is None:
        return None
    data = dataclasses.asdict(read)
    # min_confidence 是 property，asdict 里没有 —— 单独补上。
    data["min_confidence"] = read.min_confidence
    return data


def _decision_to_dict(decision: OcrDecision) -> dict:
    return {
        "accepted": decision.accepted,
        "reason": decision.reason,
        "needs_strip": decision.needs_strip,
        "read": _read_to_dict(decision.read),
    }


# --------------------------------------------------------------------------
# 像素层
# --------------------------------------------------------------------------


def probe_pixels(image: Image.Image) -> dict:
    """PK 侧像素判据。全部不抛异常（模板判据本身是三态）。"""
    out: dict[str, Any] = {}
    probes = (
        ("is_question_screen", screen.is_question_screen),
        ("has_rendered_stem", screen.has_rendered_stem),
        ("has_answer_reveal", screen.has_answer_reveal),
        ("stem_text_pixels", screen.stem_text_pixels),
        ("reveal_fill_count", screen.reveal_fill_count),
        # **三态**：None = 判不了（模板资源缺失 / 画面比例不可信 / 匹配出错）
        ("looks_like_question_template", screen.looks_like_question_template),
    )
    for name, fn in probes:
        try:
            out[name] = fn(image)
        except Exception as exc:  # noqa: BLE001 - 像素层不该崩，记录即可
            out[name] = _exc(exc)
    return out


def probe_daily_pixels(image: Image.Image) -> dict:
    """每日一答侧像素判据。"""
    out: dict[str, Any] = {}
    probes = (
        ("is_question_screen", daily_screen.is_question_screen),
        ("has_progress_line", daily_screen.has_progress_line),
        ("has_rendered_stem", daily_screen.has_rendered_stem),
        ("has_answer_reveal", daily_screen.has_answer_reveal),
        ("stem_text_pixels", daily_screen.stem_text_pixels),
    )
    for name, fn in probes:
        try:
            out[name] = fn(image)
        except Exception as exc:  # noqa: BLE001
            out[name] = _exc(exc)
    return out


# --------------------------------------------------------------------------
# 按钮定位
# --------------------------------------------------------------------------


def probe_buttons(image: Image.Image, mode: str) -> dict:
    """按钮定位。"""
    out: dict[str, Any] = {}
    finder = buttons.find_option_buttons if mode == "pk" else daily_buttons.find_option_buttons
    try:
        boxes = finder(image)
        out["count"] = len(boxes)
        out["boxes"] = [b.as_dict() for b in boxes]
        out["center_ys"] = [b.center_y for b in boxes]
    except WqError as exc:
        out["count"] = None
        out["boxes"] = None
        out["center_ys"] = None
        out["error"] = _exc(exc)

    try:
        out["bands"] = (
            buttons.option_bands(image) if mode == "pk" else daily_option_bands(image)
        )
    except Exception as exc:  # noqa: BLE001
        out["bands"] = _exc(exc)
    return out


def detect_mode(image: Image.Image) -> dict:
    """自动判断这是 PK 还是每日一答。"""
    pk_bands = len(buttons.option_bands(image))
    daily_bands = len(daily_option_bands(image))
    if pk_bands >= 2 and daily_bands == 0:
        verdict, confidence = "pk", "high"
    elif daily_bands >= 2 and pk_bands == 0:
        verdict, confidence = "daily", "high"
    elif pk_bands >= 2 and daily_bands >= 2:
        verdict, confidence = "pk", "low"
    else:
        verdict, confidence = "unknown", "none"
    return {
        "detected": verdict,
        "confidence": confidence,
        "pk_bands": pk_bands,
        "daily_bands": daily_bands,
        "size": list(image.size),
    }


# --------------------------------------------------------------------------
# OCR 层
# --------------------------------------------------------------------------


def probe_ocr_pk(reader: RapidOcrReader, image: Image.Image) -> dict:
    """PK 的两路 OCR + 门控。"""
    out: dict[str, Any] = {}

    whole, ms = _timed(reader.read, image)
    out["whole"] = _read_to_dict(whole)
    out["whole_ms"] = ms

    # **总是传 whole** —— whole=None 时 `pk_ocr.read_strip` 会自己跑
    # read_card_text（额外一次裁图 OCR），而且题干读不出就返回 None，
    # 把已经读出的选项也丢掉。whole 非 None 时传进去省掉这次重复。
    strip, ms = _timed(reader.read_strip, image, whole)
    out["strip"] = _read_to_dict(strip)
    out["strip_ms"] = ms

    # 门控用**子类**的 decide_merged —— 它额外加了题号行判据，
    # 基类 `ocr_core.decide_merged` 没有那一条。
    out["decision"] = _decision_to_dict(reader.decide_merged(whole, strip))
    # 单路门控（只跑整图路），用来对比「两路合并」相对「只跑一路」的增益。
    out["decision_whole_only"] = _decision_to_dict(reader.decide(whole))

    # **三态**：True=读到题号行 / False=确凿否定 / None=不知道
    progress, ms = _timed(reader.read_progress_text, image)
    out["progress_text"] = {
        "has_progress": progress[0],
        "text": progress[1],
        "ms": ms,
    }

    # 题号行 + 题干的合并带
    card, ms = _timed(reader.read_card_text, image)
    out["card_text"] = {
        "has_progress": card[0],
        "stem": card[1],
        "ms": ms,
    }
    return out


def probe_ocr_daily(reader: DailyOcrReader, image: Image.Image) -> dict:
    """每日一答的两路 OCR + 门控 + 终态识别。"""
    out: dict[str, Any] = {}

    whole, ms = _timed(reader.read, image)
    out["whole"] = _read_to_dict(whole)
    out["whole_ms"] = ms

    # **总是传 whole** —— whole=None 时 `daily_ocr.read_strip` 内部会重跑
    # read()，等于把整图 OCR 的成本翻倍。
    strip, ms = _timed(reader.read_strip, image, whole)
    out["strip"] = _read_to_dict(strip)
    out["strip_ms"] = ms

    out["decision"] = _decision_to_dict(reader.decide_merged(whole, strip))
    out["decision_whole_only"] = _decision_to_dict(reader.decide(whole))

    # 终态页识别。**返回 None = "不确定"**，不是"不是终态页"。
    # 前提：引擎已加载且 warm_error 为空 —— 否则直接返回 None
    # （daily_ocr.py 的 read_terminal 有这个前置守卫）。
    terminal, ms = _timed(reader.read_terminal, image)
    out["terminal"] = (
        None if terminal is None
        else {"kind": terminal.kind, "evidence": terminal.evidence}
    )
    out["terminal_ms"] = ms

    # 高亮帧：每日一答的深蓝灰检测器看不见被染色的按钮
    out["has_highlighted_button"] = has_highlighted_button(image)
    return out


# --------------------------------------------------------------------------
# 模型层
# --------------------------------------------------------------------------


def probe_model(service, image: Image.Image, mode: str, ocr_result: dict,
                *, model_path: str) -> dict:
    """跑模型路径。"""
    out: dict[str, Any] = {}
    prompt = PROMPT_PK if mode == "pk" else DAILY_PROMPT

    if model_path in ("text", "both"):
        ocr_error = ocr_result.get("error")
        decision = ocr_result.get("decision") or {}
        read = _read_from_dict(decision.get("read"))
        if ocr_error:
            # OCR 整层就失败了 —— 门控结论都不存在，说"门控未通过"是撒谎。
            out["text"] = {
                "skipped": True,
                "reason": f"OCR 层失败（{ocr_error.get('message')}），无选项可问",
            }
        elif not decision.get("accepted") or read is None:
            out["text"] = {
                "skipped": True,
                "reason": (
                    f"门控未通过（{decision.get('reason') or '未知原因'}），"
                    "选项列表不可信，不发文本请求"
                ),
            }
        else:
            out["text"] = _run_text_path(service, read)

    if model_path in ("image", "both"):
        out["image"] = _run_image_path(service, image, prompt)
    return out


def _read_from_dict(data: dict | None) -> OcrRead | None:
    """从 JSON 化的读数里还原 OcrRead（文本路径要用）。"""
    if not data:
        return None
    return OcrRead(
        stem=data.get("stem", ""),
        options=list(data.get("options") or []),
        confidences=list(data.get("confidences") or []),
        button_count=data.get("button_count", 0),
        has_progress=data.get("has_progress", False),
    )


def _run_text_path(service, read: OcrRead) -> dict:
    """OCR 文本路径：只发题干+选项，模型只回下标。"""
    options_block = "\n".join(
        f"{index}. {text}" for index, text in enumerate(read.options)
    )
    body = TEXT_PROMPT % (read.stem, options_block)
    started = time.monotonic()
    usage_before = service.usage.snapshot()
    try:
        index = service.answer_from_text(body, len(read.options))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _exc(exc), "ms": _ms(time.monotonic() - started)}
    elapsed = _ms(time.monotonic() - started)
    result: dict[str, Any] = {
        "ok": index is not None,
        "answer_index": index,
        "options": list(read.options),
        "stem": read.stem,
        "ms": elapsed,
        "usage": dataclasses.asdict(service.usage.since(usage_before)),
    }
    if index is None:
        result["reason"] = "模型没给出合法下标（或调用失败）"
    else:
        result["chosen"] = read.options[index]
    return result


def _run_image_path(service, image: Image.Image, prompt: str) -> dict:
    """图片路径：整张图发给模型。"""
    started = time.monotonic()
    usage_before = service.usage.snapshot()
    try:
        question = service.read_question(image, prompt=prompt)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _exc(exc), "ms": _ms(time.monotonic() - started)}
    elapsed = _ms(time.monotonic() - started)
    return {
        "ok": True,
        "stem": question.stem,
        "options": list(question.options),
        "answer_index": question.answer_index,
        "raw_text": question.raw_text,
        "screen": question.screen,
        "bare_index": question.bare_index,
        "progress_no": question.progress_no,
        "ms": elapsed,
        "usage": dataclasses.asdict(service.usage.since(usage_before)),
    }


# --------------------------------------------------------------------------
# 单张图的完整流程
# --------------------------------------------------------------------------


def debug_one(path: Path, mode: str, ocr_reader, vision_service, *,
              use_model: bool, model_path: str, min_confidence: float) -> dict:
    """跑完一张图的所有层，返回一条 JSON 记录。"""
    record: dict[str, Any] = {
        "file": str(path),
        "mode": mode,
        "min_confidence": min_confidence,
    }
    overall = time.monotonic()

    try:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        record["error"] = _exc(exc)
        record["total_ms"] = _ms(time.monotonic() - overall)
        return record

    record["size"] = list(image.size)

    # 自动判别 —— 即便显式指定了 mode 也记下来，方便核对判别准不准。
    try:
        record["detection"] = detect_mode(image)
    except Exception as exc:  # noqa: BLE001
        record["detection"] = _exc(exc)

    # 像素层
    started = time.monotonic()
    if mode == "pk":
        record["pixels"] = probe_pixels(image)
    else:
        record["pixels"] = probe_daily_pixels(image)
    record["pixels_ms"] = _ms(time.monotonic() - started)

    # 按钮定位
    record["buttons"] = probe_buttons(image, mode)

    # OCR 层
    if ocr_reader is None:
        record["ocr"] = {"skipped": True, "reason": "OCR 引擎不可用"}
    else:
        started = time.monotonic()
        try:
            if mode == "pk":
                record["ocr"] = probe_ocr_pk(ocr_reader, image)
            else:
                record["ocr"] = probe_ocr_daily(ocr_reader, image)
        except Exception as exc:  # noqa: BLE001
            record["ocr"] = {"error": _exc(exc)}
        record["ocr_ms"] = _ms(time.monotonic() - started)

    # 模型层
    if not use_model:
        record["model"] = {"skipped": True, "reason": "未加 --model，不联网"}
    elif vision_service is None:
        record["model"] = {"skipped": True, "reason": "模型服务不可用"}
    else:
        try:
            record["model"] = probe_model(
                vision_service, image, mode, record.get("ocr") or {},
                model_path=model_path,
            )
        except Exception as exc:  # noqa: BLE001
            record["model"] = {"error": _exc(exc)}

    record["total_ms"] = _ms(time.monotonic() - overall)
    return record


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def summarize(records: list[dict]) -> dict:
    """批量结束后的统计。"""
    stats: dict[str, Any] = {"total": len(records)}
    if not records:
        return stats

    stats["load_failed"] = sum(
        1 for r in records if "error" in r and "size" not in r
    )

    for label in ("pk", "daily"):
        subset = [r for r in records if r.get("mode") == label]
        if not subset:
            continue
        gate_ok = whole_ok = strip_only = 0
        model_text_ok = model_image_ok = model_any = 0
        reasons: dict[str, int] = {}
        for r in subset:
            ocr = r.get("ocr") or {}
            decision = ocr.get("decision") or {}
            if decision.get("accepted"):
                gate_ok += 1
            else:
                reason = (
                    decision.get("reason")
                    or (ocr.get("error") or {}).get("message")
                    or "未知原因"
                )
                reasons[reason] = reasons.get(reason, 0) + 1
            if (ocr.get("decision_whole_only") or {}).get("accepted"):
                whole_ok += 1
            # 整图路没通过、两路合并才通过的 = 拼接路救回来的
            if decision.get("accepted") and not (
                ocr.get("decision_whole_only") or {}
            ).get("accepted"):
                strip_only += 1
            model = r.get("model") or {}
            text = model.get("text") or {}
            image = model.get("image") or {}
            if text.get("ok"):
                model_text_ok += 1
            if image.get("ok"):
                model_image_ok += 1
            if text.get("ok") or image.get("ok"):
                model_any += 1
        stats[label] = {
            "count": len(subset),
            "gate_accepted": gate_ok,
            "gate_rejected": len(subset) - gate_ok,
            "gate_accepted_pct": round(100 * gate_ok / len(subset), 1),
            "whole_path_accepted": whole_ok,
            "strip_rescued": strip_only,
            "model_text_answered": model_text_ok,
            "model_image_answered": model_image_ok,
            "model_answered": model_any,
            "reject_reasons": reasons,
        }
    return stats


# --------------------------------------------------------------------------
# 打印
# --------------------------------------------------------------------------


def _print_record(record: dict) -> None:
    """逐层打印一张图的结论。"""
    name = Path(record["file"]).name
    print(f"\n--- {name}  mode={record.get('mode')}  {record.get('total_ms')}ms")

    if "error" in record and "size" not in record:
        print(f"  读图失败：{record['error']['message']}")
        return

    detection = record.get("detection") or {}
    if isinstance(detection, dict):
        print(
            f"  自动判别：{detection.get('detected')}"
            f"（置信 {detection.get('confidence')}，"
            f"pk_bands={detection.get('pk_bands')}，"
            f"daily_bands={detection.get('daily_bands')}）"
        )

    for key, value in (record.get("pixels") or {}).items():
        print(f"  像素 {key} = {value}")

    buttons_info = record.get("buttons") or {}
    if buttons_info.get("error"):
        print(f"  按钮：定位失败 —— {buttons_info['error']['message']}")
    else:
        print(f"  按钮：{buttons_info.get('count')} 个  "
              f"center_ys={buttons_info.get('center_ys')}")

    ocr = record.get("ocr") or {}
    if ocr.get("skipped"):
        print(f"  OCR：{ocr['reason']}")
    elif ocr.get("error"):
        print(f"  OCR 失败：{ocr['error']['message']}")
    else:
        whole = ocr.get("whole")
        strip = ocr.get("strip")
        print(f"  OCR 整图路（{ocr.get('whole_ms')}ms）：")
        if whole is None:
            print("      未读出")
        else:
            print(f"      题干={whole.get('stem')!r}")
            print(f"      选项={whole.get('options')}")
            print(f"      置信度="
                  f"{[round(c, 3) for c in whole.get('confidences') or []]}")
            print(f"      最低={whole.get('min_confidence')}  "
                  f"按钮数={whole.get('button_count')}")
        print(f"  OCR 拼接路（{ocr.get('strip_ms')}ms）：")
        if strip is None:
            print("      未读出")
        else:
            print(f"      题干={strip.get('stem')!r}")
            print(f"      选项={strip.get('options')}")
            print(f"      最低={strip.get('min_confidence')}  "
                  f"按钮数={strip.get('button_count')}")
        decision = ocr.get("decision") or {}
        print(f"  门控：accepted={decision.get('accepted')}  "
              f"needs_strip={decision.get('needs_strip')}")
        if decision.get("reason"):
            print(f"      理由：{decision['reason']}")
        if ocr.get("progress_text"):
            pt = ocr["progress_text"]
            print(f"  题号行：has_progress={pt.get('has_progress')}  "
                  f"原文={pt.get('text')!r}")
        if ocr.get("card_text"):
            ct = ocr["card_text"]
            print(f"  合并带：has_progress={ct.get('has_progress')}  "
                  f"题干={ct.get('stem')!r}")
        if "terminal" in ocr:
            print(f"  终态：{ocr.get('terminal')}")
        if "has_highlighted_button" in ocr:
            print(f"  已作答高亮：{ocr.get('has_highlighted_button')}")

    model = record.get("model") or {}
    if model.get("skipped"):
        print(f"  模型：{model['reason']}")
    elif model.get("error"):
        print(f"  模型失败：{model['error']['message']}")
    else:
        text = model.get("text") or {}
        if text:
            if text.get("skipped"):
                print(f"  模型文本路径：跳过 —— {text.get('reason')}")
            elif not text.get("ok"):
                # **失败要显示原因** —— 打印成「下标=None」会让人以为是
                # 「模型答不出来」，而实际是超时/网络错误，两回事。
                print(f"  模型文本路径（{text.get('ms')}ms）：失败 —— "
                      f"{(text.get('error') or {}).get('message', '未知原因')}")
            else:
                print(f"  模型文本路径（{text.get('ms')}ms）："
                      f"下标={text.get('answer_index')}  "
                      f"选项={text.get('chosen')}")
        image = model.get("image") or {}
        if image:
            if not image.get("ok"):
                print(f"  模型图片路径（{image.get('ms')}ms）：失败 —— "
                      f"{(image.get('error') or {}).get('message', '未知原因')}")
            else:
                print(f"  模型图片路径（{image.get('ms')}ms）："
                      f"screen={image.get('screen')}  "
                      f"下标={image.get('answer_index')}")
                print(f"      题干={image.get('stem')!r}")
                print(f"      选项={image.get('options')}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def collect_images(args) -> list[Path]:
    """收集要处理的图片。"""
    if args.image:
        path = Path(args.image)
        if not path.is_file():
            print(f"图片不存在：{path}", file=sys.stderr)
            return []
        return [path]
    if args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            print(f"目录不存在：{directory}", file=sys.stderr)
            return []
        files = sorted(directory.glob(args.glob))
        if not files:
            print(f"{directory} 里没有匹配 {args.glob} 的图片", file=sys.stderr)
        return files
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="逐层调试读题链路（像素层 → 按钮 → OCR 两路 → 门控 → 模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 离线跑 PK（不联网、不花钱）
  python tools/debug_read.py --image 答题页.png --mode pk

  # 批量每日一答，输出 JSON 供 diff
  python tools/debug_read.py --dir 每日一答截图目录 --mode daily --out out/daily.json

  # 带模型：只测图片路径
  python tools/debug_read.py --image 答题页.png --mode pk --model --model-path image

  # 带模型：文本路径 + 图片路径都测
  python tools/debug_read.py --image 答题页.png --mode pk --model --model-path both
""",
    )
    source = parser.add_argument_group("输入")
    source.add_argument("--image", help="单张图片路径")
    source.add_argument("--dir", help="图片目录（批量）")
    source.add_argument(
        "--mode",
        choices=("pk", "daily", "auto"),
        default="auto",
        help="链路：pk / daily / auto（auto 按按钮检测器判断，判不出就跳过）。"
             "默认 auto",
    )
    source.add_argument(
        "--glob",
        default="*.png",
        help="--dir 时的文件匹配模式。默认 *.png",
    )

    model = parser.add_argument_group("模型（默认不调，不联网不花钱）")
    model.add_argument(
        "--model",
        action="store_true",
        help="调用模型。**要花钱且慢**，默认关闭",
    )
    model.add_argument(
        "--model-path",
        choices=("image", "text", "both"),
        default="both",
        help="模型走哪条路径。默认 both（图片路径 + OCR 文本路径）",
    )
    model.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="最多处理几张（0 = 不限）。批量调模型时限量花销",
    )

    output = parser.add_argument_group("输出")
    output.add_argument("--out", help="JSON 输出文件。批量为数组，单张为对象")
    output.add_argument(
        "--quiet",
        action="store_true",
        help="只写 JSON 文件，不打印逐层明细",
    )
    return parser


def _make_reader(cfg: AppConfig, mode: str):
    """造对应链路的 OCR 读取器。"""
    if mode == "pk":
        return RapidOcrReader(min_confidence=cfg.ocr_min_confidence)
    return DailyOcrReader(min_confidence=cfg.daily_ocr_min_confidence)


def _warm_ocr(reader, sample: Image.Image | None) -> tuple[bool, float]:
    """预热 OCR 引擎，返回 (是否可用, 毫秒)。"""
    started = time.monotonic()
    ok = reader.warmup(sample)
    return ok, _ms(time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.image and not args.dir:
        print("必须给 --image 或 --dir", file=sys.stderr)
        return 2

    files = collect_images(args)
    if not files:
        return 2
    if args.max_images and len(files) > args.max_images:
        files = files[: args.max_images]
        print(f"按 --max-images 只处理前 {len(files)} 张")

    # ---- 终端编码 ----
    # Windows 的 stdout 默认是 gbk，中文全是乱码。项目里的 .bat 启动器靠
    # PYTHONIOENCODING=utf-8 解决；这里检测并提示，不自行改环境变量。
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
    if encoding not in ("utf8", "utf16", "utf32"):
        print(
            "⚠ 终端编码不是 UTF-8，中文可能显示为乱码。\n"
            "  用 PYTHONIOENCODING=utf-8 重新运行，或改用 PowerShell。",
            file=sys.stderr,
        )

    # ---- 配置 ----
    try:
        cfg = AppConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        print(f"配置加载失败：{exc}", file=sys.stderr)
        return 2

    # ---- OCR 引擎 ----
    # 预热用的样例图：**两张链路都要能读**，所以用第一张图。
    warm_sample = None
    try:
        with Image.open(files[0]) as raw:
            warm_sample = raw.convert("RGB")
    except Exception:  # noqa: BLE001 - 读不到就只加载模型，不预热
        warm_sample = None

    # 两套 reader 都要 —— auto 模式下判出哪条链路就用哪套。
    # 但只预热「会用上的」那套：加载 onnx 模型要约 1 秒。
    readers: dict[str, Any] = {}
    warm_ms: dict[str, float] = {}
    for mode in ("pk", "daily"):
        if args.mode != "auto" and args.mode != mode:
            continue
        reader = _make_reader(cfg, mode)
        ok, ms = _warm_ocr(reader, warm_sample)
        warm_ms[mode] = ms
        print(f"OCR 预热（{mode}）：{'成功' if ok else '失败'} {ms}ms")
        if not ok:
            print(f"  ⚠ 引擎不可用：{reader.load_error}", file=sys.stderr)
            print("  该链路的 OCR 与门控层会缺失，其余照常", file=sys.stderr)
            continue
        if reader.warm_error:
            print(f"  ⚠ 预热失败（不影响识别）：{reader.warm_error}", file=sys.stderr)
        readers[mode] = reader

    # ---- 模型服务 ----
    vision_service = None
    if args.model:
        try:
            cfg.validate_model()
            # 两条链路的超时与严格度不同，用各自的 builder：
            # daily 的是 11.5 秒且思考块未闭合直接抛错，PK 的是 11 秒。
            # auto 模式先用 PK 的（超时更短，判别不出时也不会白等太久）。
            vision_service = (
                build_daily_vision_service(cfg)
                if args.mode == "daily"
                else build_vision_service(cfg)
            )
            started = time.monotonic()
            vision_service.warmup_client()
            print(f"模型连接预热：{_ms(time.monotonic() - started)}ms")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠ 模型服务不可用：{exc}", file=sys.stderr)
            print("  只跑本地层", file=sys.stderr)

    # ---- 主循环 ----
    records: list[dict] = []
    for path in files:
        mode = args.mode
        if mode == "auto":
            # 先判别，再决定用哪套 reader —— **不双跑**。
            try:
                with Image.open(path) as raw:
                    probe = raw.convert("RGB")
                detected = detect_mode(probe)
            except Exception as exc:  # noqa: BLE001
                print(f"{path.name}  读图失败：{exc}", file=sys.stderr)
                records.append({
                    "file": str(path),
                    "mode": "unknown",
                    "error": _exc(exc),
                })
                continue
            mode = detected["detected"]
            if mode not in ("pk", "daily"):
                records.append({
                    "file": str(path),
                    "mode": "unknown",
                    "size": detected["size"],
                    "detection": detected,
                    "skipped": (
                        "自动判别不出链路"
                        f"（pk_bands={detected['pk_bands']}，"
                        f"daily_bands={detected['daily_bands']}）"
                    ),
                })
                if not args.quiet:
                    print(f"{path.name}  跳过：自动判别不出链路")
                continue

        reader = readers.get(mode)
        record = debug_one(
            path,
            mode,
            reader,
            vision_service,
            use_model=args.model,
            model_path=args.model_path,
            min_confidence=(
                cfg.ocr_min_confidence if mode == "pk" else cfg.daily_ocr_min_confidence
            ),
        )
        records.append(record)
        if not args.quiet:
            _print_record(record)

    # ---- 汇总 ----
    stats = summarize(records)
    print()
    print("=" * 64)
    print(f"共 {stats['total']} 张"
          f"（读图失败 {stats.get('load_failed', 0)}）")
    for label in ("pk", "daily"):
        if label not in stats:
            continue
        s = stats[label]
        print(f"  {label}: {s['count']} 张  "
              f"门控通过 {s['gate_accepted']}（{s['gate_accepted_pct']}%）  "
              f"拼接路救回 {s['strip_rescued']}  "
              f"模型答出 {s['model_answered']}")
        for reason, count in sorted(s["reject_reasons"].items(),
                                    key=lambda kv: -kv[1]):
            print(f"      判否 {count} 次：{reason}")
    skipped = [r for r in records if r.get("skipped")]
    if skipped:
        print(f"  跳过 {len(skipped)} 张（自动判别不出链路）")
    if args.model:
        print("  （模型已启用 —— 本次调用产生了费用）")

    # ---- 落盘 ----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": args.mode,
            "min_confidence": {
                "pk": cfg.ocr_min_confidence,
                "daily": cfg.daily_ocr_min_confidence,
            },
            "model_enabled": args.model,
            "model_path": args.model_path if args.model else None,
            "ocr_warmup_ms": warm_ms,
            "summary": stats,
            "records": records,
        }
        out_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, default=_jsonable),
            encoding="utf-8",
        )
        print(f"JSON -> {out_path}")

    if vision_service is not None:
        vision_service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
