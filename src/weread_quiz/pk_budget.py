"""PK 单题识别预算：倒计时锚定原截图，始终留出复核及点击余量。"""
from dataclasses import dataclass


_CLICK_RESERVE_SECONDS = 1.5

# 文本路径（本地 OCR 读题 + 模型只判下标）的单次上限。
#
# 实测同一局的 9 次成功请求耗时 0.8~4.6 秒，另有一次 10 秒没回。给
# 5 秒：慢请求被切掉，但还剩 ~4.5 秒走图片路径 —— 两条路都慢才是真
# 的没救。
#
# **不给它单独设限的后果**：文本路径会吃光整题预算（倒计时 12 秒减去
# 点击预留就是 10.5 秒），超时后只能蒙答 —— 而二选一蒙对也只有一半
# 机会，白白丢一题。
_TEXT_PATH_TIMEOUT = 5.0

# 图片路径至少要剩这么多预算才值得试。
#
# 图片路径要把整张图编码后发给模型读，实测 1.5~3 秒。留 3.5 秒是
# 「切掉慢的文本路径之后还够跑一次图片路径」的底线。
_IMAGE_PATH_MIN_BUDGET = 3.5


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
