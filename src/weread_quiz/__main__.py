"""`python -m weread_quiz` 入口。启动器以 ASCII/CRLF 保存，中文提示由 Python 输出。"""

from __future__ import annotations

import os
import sys

from .main import _force_utf8_output, main

# 启动器提示文案。键是启动器通过 WEREQ_BANNER 传进来的 ASCII 标识。
#
# 开头给的是微信小程序的直达链接 —— 双击启动器时人还没打开小程序，
# 这是最容易卡住的一步。#小程序:// 是微信私有协议，浏览器里点不开，
# 要复制到微信「文件传输助手」再点。
BANNERS = {
    "pk": (
        "请先在微信里打开微信读书小程序：\n"
        "  #小程序://微信读书/bJK5TaSZ3rUpu1D\n"
        "（复制到微信「文件传输助手」再点开；浏览器里点不开）\n"
        "\n"
        "程序会控制鼠标自动点击作答。\n"
        "想中止：按 Esc 键。"
    ),
    "daily-days": (
        "⚠ 正式答题：答错即当日机会用完，不可恢复。\n"
        "⚠ 不会使用跳题卡/换题卡/续命卡等任何道具。\n"
        "\n"
        "请先在微信里打开微信读书小程序：\n"
        "  #小程序://微信读书/lI9387D9hnbmJnk\n"
        "（复制到微信「文件传输助手」再点开；浏览器里点不开）\n"
        "\n"
        "程序会控制鼠标自动点击作答。\n"
        "想中止：按 Esc 键。"
    ),
    "daily-coins": (
        "⚠ 正式答题：答错即当日机会用完，不可恢复。\n"
        "⚠ 不会使用跳题卡/换题卡/续命卡等任何道具。\n"
        "\n"
        "请先在微信里打开微信读书小程序：\n"
        "  #小程序://微信读书/lI9387D9hnbmJnk\n"
        "（复制到微信「文件传输助手」再点开；浏览器里点不开）\n"
        "\n"
        "程序会控制鼠标自动点击作答。\n"
        "想中止：按 Esc 键。"
    ),
}


def print_banner() -> None:
    """打印启动器提示。环境变量没设就什么都不打。"""
    key = os.environ.get("WEREQ_BANNER", "").strip()
    text = BANNERS.get(key)
    if not text:
        return
    for line in text.split("\n"):
        print(line)
    print()


if __name__ == "__main__":
    # banner 里有 ⚠ —— 先把输出编码定成 UTF-8，再打印。
    _force_utf8_output()
    print_banner()
    sys.exit(main())
