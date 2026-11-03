"""项目异常：区分本地门控判否、模型调用失败、识别超时及非答题页面。"""


class WqError(Exception):
    """所有本项目异常的基类。"""


class AnswerUnavailable(WqError):
    """模型调用失败、超时，或返回了非法选项。"""


class RecognitionTimeout(AnswerUnavailable):
    """识别阶段把单题的时间预算耗光了。"""


class OcrRejected(AnswerUnavailable):
    """本地 OCR 门控**判定这张图读不出来**（不是调用失败）。"""

    def __init__(self, message: str, *, read=None) -> None:
        super().__init__(message)
        # 被判否时**已经读到的内容**；读不出整体读数时为 None。
        self.read = read


# ---- 必须立即停止 ----


class WindowNotFound(WqError):
    """未找到微信宿主窗口。"""


class NotAQuestionScreen(WqError):
    """当前界面不是答题页。"""
