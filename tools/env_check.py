"""环境自检：把「为什么没自动点『自动匹配』」需要的证据一次打印出来。

在出问题的机器上跑：

    .venv\\Scripts\\python.exe tools\\env_check.py

打印四段：系统与依赖、OCR 引擎、微信窗口（程序会选哪一个）、当前窗口里 OCR
读到的全部文字（程序就是靠这些文字找按钮的）。文字那几行最关键 ——
「自动匹配」不在里面，就说明是白名单/读数问题；在里面却没点上，
就说明是点击坐标（显示缩放）问题。
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weread_quiz.main import _force_utf8_output  # noqa: E402


def _section(title: str) -> None:
    print()
    print(f"===== {title} =====")


def _env() -> None:
    _section("系统与依赖")
    print(f"系统：{platform.platform()}")
    print(f"Python：{sys.version.split()[0]}（{sys.executable}）")
    for name in ("rapidocr", "onnxruntime", "mss", "pyautogui", "httpx", "numpy", "PIL"):
        try:
            module = __import__(name)
            print(f"  {name}: {getattr(module, '__version__', '已安装')}")
        except Exception as exc:  # noqa: BLE001 - 缺依赖也要继续往下查
            print(f"  {name}: 不可用（{type(exc).__name__}: {exc}）")


def _scale() -> None:
    from weread_quiz.dpi import get_scale_for_window, init_dpi_awareness

    try:
        init_dpi_awareness()
    except Exception as exc:  # noqa: BLE001 - 感知失败要报出来
        print(f"DPI 感知初始化失败：{type(exc).__name__}: {exc}")
        return
    print(f"主显示器缩放：{get_scale_for_window(None):.2f}（1.00 = 100%，1.50 = 150%）")


def _ocr() -> None:
    _section("OCR 引擎")
    from weread_quiz.ocr_core import build_rapid_ocr

    try:
        engine = build_rapid_ocr()
    except Exception as exc:  # noqa: BLE001 - 装不上/加载失败是常见故障
        print(f"RapidOCR 加载失败：{type(exc).__name__}: {exc}")
        print("→ 推进阶段全靠本地 OCR 认按钮，引擎不可用就一个按钮都点不到。")
        return
    print("RapidOCR 加载成功")
    try:
        import numpy as np
        from PIL import Image

        canvas = Image.new("RGB", (320, 80), "white")
        canvas.paste(Image.new("RGB", (120, 40), "black"), (20, 20))
        engine(np.asarray(canvas.convert("RGB")), use_cls=False)
        print("首次推理成功")
    except Exception as exc:  # noqa: BLE001 - 推理失败也是环境问题
        print(f"首次推理失败：{type(exc).__name__}: {exc}")


def _window() -> object | None:
    _section("微信窗口")
    from weread_quiz.window.finder import find_wechat_window, list_windows, pick_largest_wechat

    windows = list_windows()
    print("可见的微信相关窗口（按程序自己的候选规则过滤）：")
    candidates = [
        info for info in windows
        if info.title
        and len(info.title.strip()) <= 12
        and ("微信" in info.title or "WeChat" in info.title)
    ]
    if not candidates:
        print("  （没有 —— 微信 PC 客户端没开，或小程序窗口没打开）")
    for info in candidates:
        print(f"  {info.title!r} {info.width}×{info.height} @({info.left},{info.top})")
    picked = pick_largest_wechat(windows)
    if picked is not None and picked.title.strip() != "微信读书":
        print(f"⚠ 程序会退回按面积挑，选中的是 {picked.title!r}"
              f"（不是「微信读书」小程序窗口 —— 抓错窗口整局作废）")
    try:
        host = find_wechat_window()
    except Exception as exc:  # noqa: BLE001 - 找不到窗口就是找不到
        print(f"find_wechat_window() 失败：{exc}")
        return None
    print(f"程序会选中：{host.title!r} {host.width}×{host.height} @({host.left},{host.top})")
    if host.title.strip() != "微信读书":
        print("⚠ 选中的不是「微信读书」小程序窗口 —— 抓错窗口整局作废。")
    from weread_quiz.dpi import get_scale_for_window

    print(f"该窗口缩放：{get_scale_for_window(host.hwnd):.2f}")
    return host


def _texts(host) -> None:
    _section("当前窗口里 OCR 读到的文字")
    from weread_quiz.capture import grab_screen
    from weread_quiz.ocr_core import build_rapid_ocr

    region = host.to_rect()
    try:
        image = grab_screen(region)
    except Exception as exc:  # noqa: BLE001 - 抓屏失败也是环境问题
        print(f"抓屏失败：{type(exc).__name__}: {exc}")
        return
    print(f"物理窗口：{region.width}×{region.height} @({region.left},{region.top})"
          f" → 归一化后截图 {image.width}×{image.height}"
          f"（像素判据按 {image.width} 宽标定）")
    try:
        engine = build_rapid_ocr()
        output = engine(__import__("numpy").asarray(image.convert("RGB")), use_cls=False)
    except Exception as exc:  # noqa: BLE001 - 读不出就报读不出
        print(f"OCR 读图失败：{type(exc).__name__}: {exc}")
        return
    lines = [(text, float(score)) for text, score in zip(output.txts or [], output.scores or [])]
    if not lines:
        print("（一个字都没读到 —— 窗口可能被遮挡，或 OCR 对这张图不适用）")
        return
    for text, score in lines:
        print(f"  {score:.3f}  {text!r}")
    names = {text for text, _ in lines}
    print()
    for target in ("问答PK", "自动匹配"):
        print(f"白名单目标 {target!r}：{'读到' if target in names else '没读到（精确相等）'}")
    print()
    _click_probe(host, output)


def _click_probe(host, output) -> None:
    """验算点击点：按标定坐标换算出的屏幕位置，落在按钮上吗？

    不移动鼠标、不点击 —— 只在那个物理位置抓一小块图，看看是不是目标文字。
    """
    import mss
    from PIL import Image

    from weread_quiz.dpi import box_center_to_screen
    from weread_quiz.models import Rect
    from weread_quiz.ocr_core import build_rapid_ocr

    region = host.to_rect()
    targets = ("问答PK", "自动匹配")
    hits = []
    for box, text in zip(output.boxes or [], output.txts or []):
        if text not in targets:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        hits.append((text, Rect(left=int(min(xs)), top=int(min(ys)),
                                width=int(max(xs) - min(xs)),
                                height=int(max(ys) - min(ys)))))
    if not hits:
        print("（页面上没有白名单目标，跳过点击点验算）")
        return

    engine = build_rapid_ocr()
    for text, rect in hits:
        x, y = box_center_to_screen(region, rect)
        print(f"「{text}」标定坐标 ({rect.left},{rect.top}) {rect.width}×{rect.height}"
              f" → 屏幕物理点 ({x:.0f},{y:.0f})，窗口 {region.as_dict()}")
        left, top = int(x) - 60, int(y) - 20
        with mss.MSS() as sct:
            raw = sct.grab({"left": left, "top": top, "width": 120, "height": 40})
        patch = Image.frombytes("RGB", raw.size, raw.rgb)
        try:
            result = engine(__import__("numpy").asarray(patch.convert("RGB")),
                            use_cls=False)
        except Exception as exc:  # noqa: BLE001 - 读不出就报读不出
            print(f"    点击点附近读不出文字：{type(exc).__name__}: {exc}")
            continue
        near = [t for t in (result.txts or []) if t.strip()]
        if not near:
            print("    点击点附近没有文字 —— 点不到目标")
        elif text in near:
            print(f"    ✓ 点击点命中：{near}")
        else:
            print(f"    ✗ 点击点读到的是 {near}，不是 {text!r} —— 坐标换算不对")
        print("    " + _cursor_roundtrip(x, y))


def _cursor_roundtrip(x: float, y: float) -> str:
    """把光标移到那个点再读回来 —— 验证 SetCursorPos 与 GetCursorPos 同一坐标系。

    鼠标会被移动一下（随后放回原处），不会点任何东西。
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return "读不到光标位置（GetCursorPos 失败）"
    origin = (point.x, point.y)
    user32.SetCursorPos(int(x), int(y))
    user32.GetCursorPos(ctypes.byref(point))
    landed = (point.x, point.y)
    user32.SetCursorPos(origin[0], origin[1])
    if landed == (int(x), int(y)):
        return f"✓ 光标往返一致：({landed[0]},{landed[1]})（已放回原位 {origin}）"
    return f"✗ 光标落到 {landed}，不是 ({int(x)},{int(y)}) —— 点击坐标系不对"


def main() -> int:
    _force_utf8_output()
    _env()
    _scale()
    _ocr()
    host = _window()
    if host is not None:
        _texts(host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
