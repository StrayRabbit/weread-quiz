"""PK OCR 文本路径的 thinking 开/关 A/B 实测。"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from weread_quiz.config import AppConfig
from weread_quiz.pk.ocr import RapidOcrReader
from weread_quiz.vision import VisionService
from weread_quiz.pk.ocr import TEXT_PROMPT


def collect(run_dir: Path) -> list[tuple[str, str, list[str]]]:
    """从一局运行目录里取出 (题干, 拼接后的选项块, 选项列表)。"""
    from PIL import Image

    ocr = RapidOcrReader(min_confidence=0.0)
    items = []
    for path in sorted(run_dir.glob("*_q*_before.png")):
        image = Image.open(path).convert("RGB")
        read = ocr.read(image)
        if read is None or not read.options:
            continue
        block = "\n".join(f"{i}. {t}" for i, t in enumerate(read.options))
        items.append((read.stem, block, list(read.options)))
    return items


def run(vision: VisionService, stem: str, block: str, count: int):
    prompt = TEXT_PROMPT % (stem, block)
    started = time.monotonic()
    index = vision.answer_from_text(prompt, count)
    return index, (time.monotonic() - started) * 1000


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) > 1 else 1)
    run_dir = Path(sys.argv[1])

    cfg = AppConfig.from_env()
    cfg.validate_model()

    items = collect(run_dir)
    if not items:
        print("没读到题目 —— 检查截图目录")
        return
    print(f"样本 {len(items)} 道题（来自 {run_dir.name}）\n")

    results = {}
    for label, thinking in (("thinking 开", True), ("thinking 关", False)):
        with VisionService(
            cfg.model_base_url, cfg.model_api_key, cfg.model_name,
            thinking=thinking,
        ) as vision:
            times, indices = [], []
            for stem, block, options in items:
                index, elapsed = run(vision, stem, block, len(options))
                times.append(elapsed)
                indices.append(index)
            results[label] = (times, indices)
            print(f"--- {label} ---")
            print(f"  耗时 中位 {statistics.median(times):.0f}ms  "
                  f"均值 {statistics.mean(times):.0f}ms  "
                  f"区间 {min(times):.0f}~{max(times):.0f}ms")
            print(f"  下标 {indices}")
            print(f"  解析失败/越界 {sum(1 for i in indices if i is None)} 次\n")

    on_times, on_idx = results["thinking 开"]
    off_times, off_idx = results["thinking 关"]
    diff = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(on_idx, off_idx), 1)
        if a != b
    ]
    print("=== 结论 ===")
    print(f"耗时：中位 {statistics.median(on_times):.0f}ms → "
          f"{statistics.median(off_times):.0f}ms "
          f"(省 {statistics.median(on_times) - statistics.median(off_times):.0f}ms)")
    if diff:
        print(f"答案不一致 {len(diff)}/{len(items)} 道：")
        for i, a, b in diff:
            print(f"  第{i}题：开={a} 关={b}")
    else:
        print(f"答案完全一致（{len(items)}/{len(items)}）")


if __name__ == "__main__":
    main()
