"""每日一答的终端逐题行：每题一行，答后反馈合成进去；明细只落盘。

与 PK 共用 record.format_question_line —— 两边的终端格式必须一样。
"""

from weread_quiz import daily
from weread_quiz.daily_hybrid import DailyHybridReader
from weread_quiz.models import Rect
from test_daily_pipeline import Ocr, Vision, clock, desktop, read  # noqa: F401


class _Feedback:
    """假的答后反馈：按 _FEEDBACK_SEQUENCE 依次给结果。"""

    def __init__(self, mode):
        self.last_question = False
        self.status = "unknown"

    def start(self, before, box):
        pass

    def observe_now(self, grab):
        pass

    def observe(self, image):
        pass

    def sample_window(self, grab, **kwargs):
        pass

    def finish(self, result, *, recorder=None, verbose=True, label="本题",
               terminal=False):
        result["answer_feedback"] = (_FEEDBACK_SEQUENCE.pop(0)
                                      if _FEEDBACK_SEQUENCE else "unknown")


# 每题一个实例，状态得由外部队列喂 —— 用例按需预置。
_FEEDBACK_SEQUENCE = []


class _Recorder:
    echo = False

    def __init__(self):
        self.messages = []

    def log(self, message):
        self.messages.append(message)

    def shot(self, name, image=None):
        pass


def _config(reader, recorder, **kwargs):
    return daily.DailyConfig(Rect(0, 0, 430, 788), 1.0, reader, verbose=True,
                             recorder=recorder, **kwargs)


def _question_lines(output: str) -> list[str]:
    return [line for line in output.splitlines()
            if line.startswith("  ") and "→" in line]


def test_one_line_per_question_with_feedback_mark(desktop, monkeypatch, capsys):
    vision = Vision()
    reader = DailyHybridReader(vision, Ocr(read()), verbose=True)
    _FEEDBACK_SEQUENCE[:] = ["correct", "wrong"]
    monkeypatch.setattr(daily, "AnswerFeedback", _Feedback)
    monkeypatch.setattr(daily, "wait_for_next_question",
                        lambda *a, **k: daily._WaitOutcome("next", desktop[0],
                                                          captured_at=100.25))
    monkeypatch.setattr(daily, "_read_final", lambda cfg, **kwargs: "success")

    report = daily.run_daily(_config(reader, _Recorder(), max_questions=2))
    assert report.outcome == "success" and report.answered == 2

    lines = _question_lines(capsys.readouterr().out)
    assert len(lines) == 2
    assert lines[0].startswith("  1. ") and lines[0].endswith("✓")
    assert lines[1].startswith("  1. ") and lines[1].endswith("✗")
    # 逐题行里不再单独打一行「答后反馈」。
    assert "答后反馈" not in capsys.readouterr().out


def test_per_question_token_only_goes_to_log(desktop, monkeypatch, capsys):
    vision = Vision()
    reader = DailyHybridReader(vision, Ocr(read()), verbose=True)
    recorder = _Recorder()
    monkeypatch.setattr(daily, "AnswerFeedback", _Feedback)
    monkeypatch.setattr(daily, "wait_for_next_question",
                        lambda *a, **k: daily._WaitOutcome("next", desktop[0],
                                                          captured_at=100.25))
    monkeypatch.setattr(daily, "_read_final", lambda cfg, **kwargs: "success")

    daily.run_daily(_config(reader, recorder, max_questions=2))
    output = capsys.readouterr().out
    assert "第 1 题 token" not in output
    assert any("第 1 题 token" in message for message in recorder.messages)


def test_dry_run_line_marks_not_clicked(desktop, capsys):
    vision = Vision()
    reader = DailyHybridReader(vision, Ocr(read()), verbose=True)

    report = daily.run_daily(_config(reader, _Recorder(), dry_run=True))
    assert report.outcome == "dry_run"

    lines = _question_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert lines[0].endswith("（未点击）")


def test_skipped_question_is_not_reported_as_answered(desktop, monkeypatch, capsys):
    vision = Vision()
    reader = DailyHybridReader(vision, Ocr(read()), verbose=True)
    monkeypatch.setattr(daily, "AnswerFeedback", _Feedback)
    monkeypatch.setattr(daily, "wait_for_next_question",
                        lambda *a, **k: daily._WaitOutcome("next", desktop[0],
                                                          captured_at=100.25))
    monkeypatch.setattr(daily, "_read_final", lambda cfg, **kwargs: "success")

    real = daily.answer_one_question

    def skipped(*args, **kwargs):
        result = real(*args, **kwargs)
        result["skipped"] = True
        result["error"] = "screen_changed"
        return result

    monkeypatch.setattr(daily, "answer_one_question", skipped)
    report = daily.run_daily(_config(reader, _Recorder(), max_questions=1))

    lines = _question_lines(capsys.readouterr().out)
    assert lines and "跳过（screen_changed），未点击" in lines[0]
    assert lines[0].endswith("-")
    assert report.skipped == 1 and report.answered == 0
