"""PK OCR 快路径：读取题干、选项和题号证据，使用置信度及按钮数门控。"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from .blank import TextSpan, compose_stem, find_blank_lines
from .buttons import option_bands
from .errors import WqError
from .ocr_core import (
    OcrDecision,
    OcrRead,
    RapidOcrEngine,
    decide_merged,
    decide_read,
)

# 置信度门控阈值。低于它就回退图片路径。
#
DEFAULT_MIN_CONFIDENCE = 0.7

# 题干区要剔除的非题目文字。
#
_NOISE_PATTERNS = (
    re.compile(r"^VS$", re.IGNORECASE),
    re.compile(r"^\d+$"),
    re.compile(r"^\d{1,2}:\d{2}$"),
    re.compile(r"^第\s*\d+\s*题"),
    re.compile(r"共\s*\d+\s*题"),
    re.compile(r"^最后一题"),
)

# OCR 行与按钮的纵向匹配容差（像素）。按钮高 43~50px，OCR 文字行
# 高度 17~25px 且居中，取按钮高度的一半足够宽松又不串行。
_CENTER_TOLERANCE_RATIO = 0.5

_X_TOLERANCE = 12

_PROGRESS_BAND = (248, 312)

_PROGRESS_PATTERNS = (
    re.compile(r"第\s*\d+\s*题"),
    re.compile(r"共\s*\d+\s*题"),
    re.compile(r"最后一题"),
)

_BUTTON_PAD_V = 6     # 按钮内文字区的上下留白
_BUTTON_PAD_H = 8     # 左右留白，切掉青色边框和尖角

_CARD_BAND = (248, 368)

_CARD_SCALE = 1


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    return any(pattern.search(stripped) for pattern in _NOISE_PATTERNS)


def _is_progress(text: str) -> bool:
    """这行文字是不是题号行？"""
    stripped = text.strip()
    return any(pattern.search(stripped) for pattern in _PROGRESS_PATTERNS)


class RapidOcrReader(RapidOcrEngine):
    """PK 的 RapidOCR 读取器。"""

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        super().__init__(min_confidence=min_confidence)

    def read(self, image: Image.Image) -> OcrRead | None:
        """识别一张截图。引擎不可用或读不出按钮时返回 None。"""
        bands = self._bands(image)
        if not bands:
            return None
        result = self._engine_run(image)
        if result is None:
            return None
        return self._map_lines(result, bands, stem_from_bands=True, image=image)

    def read_card_text(self, image: Image.Image) -> tuple[bool | None, str]:
        """保留公开二元接口；内部详细结果同时携带题号原文。"""
        progress, stem, _raw = self._read_card_text_detail(image)
        return progress, stem

    def _read_card_text_detail(self, image: Image.Image) -> tuple[bool | None, str, str]:
        """一次裁图读取题干及题号是否存在，分别返回类型证据与题号原文。"""
        if self._engine is None and not self.warmup():
            return None, "", ""
        top, bottom = _CARD_BAND
        band = image.crop((0, top, image.size[0], bottom))
        if band.size[1] < 8:
            return None, "", ""
        band = band.resize((band.size[0] * _CARD_SCALE,
                            band.size[1] * _CARD_SCALE), Image.LANCZOS)
        result = self._engine_run(band)
        if result is None:
            return None, "", ""
        has_progress = False
        progress_parts = []
        stem_parts: list[TextSpan] = []
        spans: list[TextSpan] = []
        for box, text, _confidence in result or []:
            stripped = text.strip()
            if not stripped:
                continue
            center_y = top + (sum(point[1] for point in box) / len(box)) / _CARD_SCALE
            span = TextSpan(
                x0=min(p[0] for p in box) / _CARD_SCALE,
                x1=max(p[0] for p in box) / _CARD_SCALE,
                y0=top + min(p[1] for p in box) / _CARD_SCALE,
                y1=top + max(p[1] for p in box) / _CARD_SCALE,
                text=stripped,
            )
            spans.append(span)
            if (_PROGRESS_BAND[0] <= center_y <= _PROGRESS_BAND[1]
                    and _is_progress(stripped)):
                has_progress = True
                progress_parts.append((center_y, min(p[0] for p in box), stripped))
                continue
            if _is_noise(stripped):
                continue
            stem_parts.append(span)
        if not result:
            # 整条带一个字都没读到 —— 这是"不知道"，不是"没有题号行"。
            return None, "", ""
        stem = compose_stem(
            stem_parts,
            find_blank_lines(image, (top, bottom), spans),
        )
        return has_progress, stem, "".join(
            text for _, _, text in sorted(progress_parts))

    def read_strip(self, image: Image.Image,
                   whole: OcrRead | None = None) -> OcrRead | None:
        """兼容补读入口：逐按钮裁出文字并批量识别，不拼接不同按钮。"""
        bands = self._bands(image)
        if not bands:
            return None
        stem = whole.stem if whole is not None else ""
        has_progress = whole.has_progress if whole is not None else False
        progress_text = whole.progress_text if whole is not None else ""
        if whole is None:
            progress, stem, progress_text = self._read_card_text_detail(image)
            has_progress = progress is True
        if not stem:
            return None

        count = len(bands)
        aligned = (whole is not None and whole.button_count == count
                   and len(whole.options) == count
                   and len(whole.confidences) == count)
        options = list(whole.options) if aligned else [""] * count
        confidences = list(whole.confidences) if aligned else [0.0] * count
        crops = []
        indices = []
        for index, band in enumerate(bands):
            if options[index].strip() and confidences[index] >= self.min_confidence:
                continue
            lines = self._button_text_crops(image, band)
            crops.extend(lines)
            indices.extend([index] * len(lines))

        if crops and self.warmup():
            try:
                # 已有按钮和文字区域，无需再跑文字检测；批量输出与输入逐项对应。
                output = self._engine.recognize_txt(crops)
                texts = output.txts
                scores = output.scores
                if (texts is None or scores is None
                        or len(texts) != len(crops) or len(scores) != len(crops)):
                    raise ValueError("逐按钮识别结果数量不符")
                recovered = [[] for _ in bands]
                for index, text, score in zip(indices, texts, scores, strict=True):
                    score = float(score)
                    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError("逐按钮识别置信度无效")
                    recovered[index].append((str(text).strip(), score))
                for index, lines in enumerate(recovered):
                    if not lines or any(not text for text, _ in lines):
                        continue
                    text = "".join(text for text, _ in lines)
                    confidence = min(score for _, score in lines)
                    if options[index].strip():
                        # 同一模糊字可能重复误读，补读高分不能抹去已有低置信证据。
                        confidence = min(confidence, confidences[index])
                    options[index] = text
                    confidences[index] = confidence
            except Exception:  # noqa: BLE001 - 补读失败保留原槽位，交门控回退
                pass
        return OcrRead(
            stem=stem,
            options=options,
            confidences=confidences,
            button_count=count,
            has_progress=has_progress,
            progress_text=progress_text,
            countdown_seconds=getattr(whole, "countdown_seconds", None),
        )

    def _bands(self, image: Image.Image) -> list[tuple[int, int]] | None:
        try:
            bands = option_bands(image)
        except WqError:
            return None
        return bands or None

    def read_progress(self, image: Image.Image) -> bool | None:
        """这一帧有没有题号行？读不出返回 None。"""
        progress, _text = self.read_progress_text(image)
        return progress

    def read_progress_text(self, image: Image.Image) -> tuple[bool | None, str]:
        """读题号行，同时把**原始文字**带回来。"""
        if self._engine is None and not self.warmup():
            return None, ""
        band = image.crop((0, _PROGRESS_BAND[0], image.size[0], _PROGRESS_BAND[1]))
        if band.size[1] < 8:
            return None, ""
        result = self._engine_run(band)
        if result is None:
            return None, ""
        text = "".join(
            line[1].strip() for line in (result or []) if line[1].strip()
        )
        for _box, line_text, _confidence in result:
            if _is_progress(line_text):
                return True, text
        # 读到文字但不是题号行 —— 这是**确凿的否定**（不是"不知道"）。
        return (False if result else None), text

    def _button_text_crops(self, image: Image.Image,
                           band: tuple[int, int]) -> list[np.ndarray]:
        """只在一个按钮内按白字行裁图；返回顺序即该按钮内的阅读顺序。"""
        top, bottom = band
        left = _BUTTON_LEFT_MIN + _BUTTON_PAD_H
        right = min(image.width, _BUTTON_LEFT_MIN + _BUTTON_WIDTH - _BUTTON_PAD_H)
        upper, lower = max(0, top + _BUTTON_PAD_V), min(image.height, bottom - _BUTTON_PAD_V)
        if upper >= lower or left >= right:
            return []
        array = np.asarray(image.crop((left, upper, right, lower)).convert("RGB"))
        low = array.min(axis=2)
        high = array.max(axis=2).astype(int)
        white = (low > 160) & (high - low < 80)
        rows = np.flatnonzero(white.any(axis=1))
        if not len(rows):
            return []
        # 相邻文字行不能送给单行识别器；小于三像素的字内空行仍属同一行。
        groups = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1)
        crops = []
        for group in groups:
            y0, y1 = int(group[0]), int(group[-1]) + 1
            if y1 - y0 < 3:
                continue
            columns = np.flatnonzero(white[y0:y1].any(axis=0))
            x0, x1 = int(columns[0]), int(columns[-1]) + 1
            crops.append(array[max(0, y0 - 4):min(array.shape[0], y1 + 4),
                               max(0, x0 - 4):min(array.shape[1], x1 + 4)].copy())
        return crops

    def _map_lines(self, result, bands: list[tuple[int, int]], *,
                   stem_from_bands: bool, image: Image.Image) -> OcrRead | None:
        first_top = min(top for top, _ in bands)
        option_parts = [[] for _ in bands]
        stem_parts: list[TextSpan] = []
        # 图上所有文字段。横线判据靠它排除「一」这类横笔。
        spans: list[TextSpan] = []
        has_progress = False
        progress_parts = []
        countdowns = set()

        for line in result or []:
            box, text, confidence = line[0], line[1], float(line[2])
            if not text.strip():
                continue
            center_y = sum(point[1] for point in box) / len(box)
            x0 = min(point[0] for point in box)
            x1 = max(point[0] for point in box)
            span = TextSpan(
                x0=x0,
                x1=x1,
                y0=min(point[1] for point in box),
                y1=max(point[1] for point in box),
                text=text.strip(),
            )
            spans.append(span)
            # 复用整图 OCR，只信倒计时固定位置的高置信度完整读数。
            timer = re.fullmatch(r"00[:：](\d{2})", re.sub(r"\s+", "", text))
            if (210 <= center_y <= 240 and 160 <= (x0 + x1) / 2 <= 270
                    and 0.90 <= confidence <= 1.0 and timer
                    and int(timer.group(1)) <= 12):
                countdowns.add(int(timer.group(1)))
            # 题号行：在题号带内，且文字形态是「第 N 题」「共 M 题」
            # 或「最后一题」。它是界面类型的判据（匹配页没有这一行）。
            if (_PROGRESS_BAND[0] <= center_y <= _PROGRESS_BAND[1]
                    and _is_progress(text)):
                has_progress = True
                progress_parts.append((center_y, x0, text.strip()))
            for index, (top, bottom) in enumerate(bands):
                height = bottom - top
                tolerance = height * _CENTER_TOLERANCE_RATIO
                center = (top + bottom) / 2
                if (
                    abs(center_y - center) <= tolerance
                    and x0 >= _BUTTON_LEFT_MIN - _X_TOLERANCE
                    and x1 <= _BUTTON_LEFT_MIN + _BUTTON_WIDTH + _X_TOLERANCE
                ):
                    option_parts[index].append((center_y, x0, text.strip(), confidence))
                    break
            else:
                # 只收答题卡内、首个按钮上方的文字；卡片外的比分和 VS 不属于正文。
                if (stem_from_bands and _CARD_BAND[0] <= center_y < first_top
                        and not _is_noise(text)):
                    stem_parts.append(span)

        options = ["".join(part[2] for part in sorted(parts)) for parts in option_parts]
        confidences = [min((part[3] for part in parts), default=0.0) for parts in option_parts]
        progress_text = "".join(text for _, _, text in sorted(progress_parts))
        # 题干按**画面上的阅读顺序**拼（先分行，行内从左到右），
        # 填空横线按位置补在中间。
        stem = compose_stem(
            stem_parts,
            find_blank_lines(image, (_CARD_BAND[0], first_top), spans),
        )
        if stem_from_bands and not stem:
            return None
        return OcrRead(
            stem=stem,
            options=options,
            confidences=confidences,
            button_count=len(bands),
            has_progress=has_progress,
            progress_text=progress_text,
            countdown_seconds=next(iter(countdowns)) if len(countdowns) == 1 else None,
        )

    def decide(self, read: OcrRead | None) -> OcrDecision:
        """PK 门控显式拒绝空槽；题号证据与选项完整性分开处理。"""
        if read is not None:
            missing = [index for index, text in enumerate(read.options) if not text.strip()]
            if missing:
                return OcrDecision(
                    False, f"未读出按钮下标 {missing} 的文字（下标从 0 开始）", read,
                    needs_strip=True,
                )
            if len(read.confidences) != len(read.options):
                return OcrDecision(False, "选项与置信度数量不符", read, needs_strip=True)
        decision = decide_read(read, self.min_confidence)
        if not decision.accepted:
            # 旧式压缩读数也允许重读，但补读时不能复用其错误下标。
            decision.needs_strip = True
            return decision
        return self._reject_without_progress(decision)

    def decide_merged(self, whole: OcrRead | None,
                      strip: OcrRead | None) -> OcrDecision:
        """合并整图与逐按钮补读，拒绝空槽且保留可定位的缺失下标。"""
        whole_decision = self.decide(whole)
        strip_decision = self.decide(strip)
        if whole_decision.accepted and strip_decision.accepted:
            decision = decide_merged(whole, strip, self.min_confidence)
            decision.reason = decision.reason.replace("拼接路", "逐按钮补读")
            return decision
        if whole_decision.accepted:
            return whole_decision
        if strip_decision.accepted:
            return OcrDecision(True, "逐按钮补读", strip)
        recovery = "未运行" if strip is None else f"：{strip_decision.reason}"
        return OcrDecision(
            False, f"本地 OCR：{whole_decision.reason}；逐按钮补读{recovery}",
            whole if whole is not None else strip,
            needs_strip=strip is None and whole_decision.needs_strip,
        )

    @staticmethod
    def _reject_without_progress(decision: OcrDecision) -> OcrDecision:
        """缺少题号证据时拒绝本地快路径，但保留已有读数供上层区分界面和门控。"""
        read = decision.read
        if read is not None and not read.has_progress:
            return OcrDecision(
                False,
                "没读到题号行（可能不是答题卡片）",
                read,
                needs_strip=False,
            )
        return decision


# 按钮横向范围，与 buttons._BUTTON_LEFT/_BUTTON_WIDTH 保持一致。
# 这里独立抄一份而不是 import 私有常量：buttons.py 把它们当内部实现，
# 且 option_bands 只返回纵向区间、不暴露横向信息。
_BUTTON_LEFT_MIN = 60
_BUTTON_WIDTH = 310


# 纯文本路径的 prompt。
#
# **这里用 % 格式化而不是 .format()**：JSON 里的 {"answer_index": ...}
# 会被 str.format 当成占位符而抛 KeyError。选项文字里也可能出现 { }，
# 用 %s 占位同样能安全代入。
TEXT_PROMPT = """回答下面这道选择题。只输出一个 JSON，不要解释：
{"answer_index": 正确选项的编号，从 0 开始}

题目：%s

选项：
%s
"""
