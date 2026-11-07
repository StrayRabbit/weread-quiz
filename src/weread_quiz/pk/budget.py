"""PK 单题识别预算：倒计时锚定原截图，始终留出复核及点击余量。"""
from dataclasses import dataclass
import time


_CLICK_RESERVE_SECONDS = 1.5

# 文本阶段（本地 OCR 后判下标）的配置上限，可用 PK_TEXT_TIMEOUT 调整。
# 有图片能力时还要按当前余量预留图片预算及超时退出余量，不能只看此上限。
_TEXT_PATH_TIMEOUT = 5.0

# 图片路径至少要剩这么多预算才值得试。
#
# 图片路径要把整张图编码后发给模型读，实测 1.5~3 秒。留 3.5 秒是
# 「切掉慢的文本路径之后还够跑一次图片路径」的底线。
_IMAGE_PATH_MIN_BUDGET = 3.5
_TEXT_TIMEOUT_EXIT_MARGIN_SECONDS = 0.3


def text_path_timeout(timeout: float, deadline: float | None, *,
                      supports_image: bool = True) -> float:
    """按当前余量限制文本阶段；能回退时预留图片预算及超时退出余量。"""
    if deadline is None:
        return timeout
    remaining = max(0.0, deadline - time.monotonic())
    if supports_image and remaining >= _IMAGE_PATH_MIN_BUDGET:
        remaining = max(0.0, remaining - _IMAGE_PATH_MIN_BUDGET
                        - _TEXT_TIMEOUT_EXIT_MARGIN_SECONDS)
    return min(timeout, remaining)


@dataclass
class PKQuestionBudget:
    captured_at: float
    deadline: float
    countdown_seconds: int | None = None

    def observe(self, read) -> None:
        """仅有题号证据且倒计时可信时更新；同张截图重复补读不能延长预算。"""
        seconds = getattr(read, "countdown_seconds", None)
        if (not getattr(read, "has_progress", False)
                or type(seconds) is not int or not 0 <= seconds <= 12):
            return
        if self.countdown_seconds is not None:
            seconds = min(seconds, self.countdown_seconds)
        self.countdown_seconds = seconds
        self.deadline = self.captured_at + seconds - _CLICK_RESERVE_SECONDS
