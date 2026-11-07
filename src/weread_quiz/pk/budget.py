"""PK 单题识别预算：倒计时锚定原截图，始终留出复核及点击余量。"""
from dataclasses import dataclass
import time


_CLICK_RESERVE_SECONDS = 0.8

# 文本优先：默认允许用足题目剩余预算，PK_TEXT_TIMEOUT 仍可设置更短上限。
_TEXT_PATH_TIMEOUT = 12.0

# 图片路径至少要剩这么多预算才值得试。仅用于 OCR 判否或文本提前无效时的回退；
# 不再从正常文本请求的预算里预扣这段时间。
_IMAGE_PATH_MIN_BUDGET = 3.5


def text_path_timeout(timeout: float, deadline: float | None) -> float:
    """文本使用剩余识别预算，不为超时后的图片路径预留时间。"""
    if deadline is None:
        return timeout
    return min(timeout, max(0.0, deadline - time.monotonic()))


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
