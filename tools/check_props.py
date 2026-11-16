"""道具判读的真模型验证：用真实截图跑一遍本地 OCR，核对数量与颜色。

离线回归测试不加载 onnxruntime（项目约定，见 pyproject.toml 的
"src" 说明和 tests/ 里"全程使用本地替身"的惯例）—— 真模型验证放在
这里，需要时手动跑：

    uv run python tools/check_props.py

它会遍历 data/runs/daily/*/ 下的真实帧，逐张打印道具判读结果，
并与期望一致时退出码为 0。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weread_quiz.daily.ocr import DailyOcrReader  # noqa: E402
from weread_quiz.daily.props import choose_prop, read_props  # noqa: E402

# 期望判读：(跳题卡数量, 跳题卡亮色, 换题卡数量, 换题卡亮色)
# 历史真实帧里的换题卡**恒为 0 且暗色** —— 没有换题卡可用的样本，
# 所以换题分支的点击落点与生效判据仍未经实机验证。
EXPECTED = {
    "20261103-082525/001_q01_before.png": (0, False, 0, False),
    "20261104-081333/001_q01_before.png": (1, True, 0, False),
    "20261105-083157/019_q12_before.png": (1, False, 0, False),
    "20261107-095729/001_q01_before.png": (2, True, 0, False),
    "20261107-205026/021_q12_before.png": (2, False, 0, False),
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reader = DailyOcrReader()
    sample = Image.open(root / "tests/fixtures/daily_props_skip_enabled.png")
    if not reader.warmup(sample):
        print("RapidOCR 不可用：", reader.warm_error or reader.load_error)
        return 2
    failures = 0
    checked = 0
    for path in sorted(glob.glob(str(root / "data/runs/daily/*/*before.png"))):
        key = "/".join(Path(path).parts[-2:])
        if key not in EXPECTED:
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
        states = read_props(image, reader)
        actual = tuple(
            value
            for kind in ("skip", "swap")
            for value in ((next(s.count for s in states if s.kind == kind),
                           next(s.enabled_color for s in states if s.kind == kind)))
        )
        expected = EXPECTED[key]
        picked = choose_prop(states, last_question=False)
        status = "OK " if actual == expected else "差异"
        if actual != expected:
            failures += 1
        checked += 1
        print(f"{status} {key}: 跳题卡 {actual[0]}/{'亮' if actual[1] else '暗'} "
              f"换题卡 {actual[2]}/{'亮' if actual[3] else '暗'} "
              f"→ 选中 {picked.kind if picked else '无'}")
    print(f"\n核对 {checked} 张真实帧，{failures} 处不一致")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
