"""拿真实截图验证视觉识别 + 按钮定位。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

from weread_quiz.pk.buttons import find_option_buttons
from weread_quiz.config import AppConfig
from weread_quiz.model_service import build_model_service


def main() -> int:
    shot_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/watch")
    shots = sorted(shot_dir.glob("*.png"))
    if not shots:
        print(f"{shot_dir} 里没有截图")
        return 1

    # 走门面 —— 主模型是决策模型时，读图的是视觉模型（VISION_*）。
    service = build_model_service(AppConfig.from_env())
    try:
        _verify_shots(shots, service)
    finally:
        service.close()
    return 0


def _verify_shots(shots, service) -> None:
    ok = 0
    for shot in shots:
        image = Image.open(shot)
        try:
            boxes = find_option_buttons(image)
            centers = [b.center_y for b in boxes]
        except Exception as exc:  # noqa: BLE001
            print(f"{shot.name}  定位失败: {exc}")
            continue

        started = time.time()
        try:
            question = service.read_question(image)
        except Exception as exc:  # noqa: BLE001
            print(f"{shot.name}  识别失败: {exc}")
            continue
        elapsed = time.time() - started

        answer = (
            "?" if question.answer_index is None
            else f"[{question.answer_index}] {question.options[question.answer_index]}"
        )
        match = "OK" if len(question.options) == len(boxes) else "选项数不符!"
        print(
            f"{shot.name}  {elapsed:.1f}s  {match}\n"
            f"    题干: {question.stem}\n"
            f"    选项: {question.options}\n"
            f"    按钮y: {centers}\n"
            f"    答案: {answer}\n"
        )
        if match == "OK":
            ok += 1

    print(f"\n共 {len(shots)} 张，按钮数与选项数一致 {ok} 张")


if __name__ == "__main__":
    sys.exit(main())
