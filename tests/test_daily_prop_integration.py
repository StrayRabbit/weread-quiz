"""保留真实逐题处理、道具点击及确认函数的流程集成测试。"""
from weread_quiz.daily import runner as daily
from weread_quiz.daily.props import PropState
from weread_quiz.models import Rect
from weread_quiz.vision import VisionQuestion
from test_daily_pipeline import clock, desktop  # noqa: F401


def test_swap_then_answer_uses_new_model_budget_and_only_counts_one_slot(desktop, clock, monkeypatch):
    class Vision:
        def __init__(self):
            self.calls = []

        def read_question(self, image, prompt=None, *, timeout=None, deadline=None):
            self.calls.append(deadline)
            clock.sleep(0.2)
            index = len(self.calls)
            return VisionQuestion(stem="换前题" if index == 1 else "换后题", options=["甲", "乙", "丙"],
                                  answer_index=None if index == 1 else index - 1,
                                  screen="question", progress_no=1 if index <= 2 else 2)

        def read_progress_number(self, image):
            return 1

    vision = Vision()
    config = daily.DailyConfig(Rect(0, 0, 430, 788), vision, verbose=False,
                               use_props_on_failure=True, max_questions=2)
    # 读数顺序：点击前判读 1、点击前复核 1、点击后确认 0。
    counts = iter([1, 1, 0])
    monkeypatch.setattr(daily, "read_props", lambda *a: [
        PropState("swap", next(counts), True, Rect(224, 630, 163, 70))])
    monkeypatch.setattr(daily, "wait_for_next_question", lambda *a, **k:
                        daily._WaitOutcome("next", desktop[0], captured_at=clock.now))
    # 换题的证据之一是"题干变了" —— 这里替身让像素判据成立。
    monkeypatch.setattr(daily, "is_new_question", lambda a, b: True)
    settlements = []
    monkeypatch.setattr(daily, "_read_final", lambda *a, **k: settlements.append(len(vision.calls)) or "success")

    report = daily.run_daily(config)

    assert report.outcome == "success"
    assert len(vision.calls) == 3 and vision.calls[1] > vision.calls[0]
    assert report.answered == 2 and len(report.questions) == 2
    assert report.progress_numbers == [1, 2]
    assert len(report.prop_uses) == 1 and report.prop_uses[0]["confirmed"]
    assert desktop[1] == [665, 390, 460]
    assert settlements == [3]
