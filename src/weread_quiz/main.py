"""进程入口。"""

from __future__ import annotations

import sys


def _force_utf8_output() -> None:
    """把终端输出统一成 UTF-8。

    中文 Windows 的默认输出编码是 GBK，那里 ✓/⚠ 这类符号编不出来 ——
    一旦把输出重定向到文件（`... > out.txt`），print 会抛
    UnicodeEncodeError，把答题循环打断在半路（鼠标还按着下一题）。
    真实控制台不受影响：Windows 控制台走 WriteConsoleW，按 UTF-16 渲染。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 换不了编码就按原样跑
            pass


def main() -> int:
    # 关键：先设 DPI 感知，再导入任何碰屏幕的模块
    from .dpi import init_dpi_awareness

    init_dpi_awareness()

    from .cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
