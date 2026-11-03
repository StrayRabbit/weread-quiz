"""逐题行：题号、题干、答案与答后反馈合成一行；反馈未知不当作答对。"""

import pytest

from weread_quiz import cli, main as wq_main, pk


def question(**changes):
    result = {"stem": "题目", "options": ["甲", "乙"], "answer": 1,
              "ui_number": 1, "answer_feedback": "correct"}
    result.update(changes)
    return result


def line(result, *, ui_number=1, question_no=1) -> str:
    return pk.format_question_line(result, ui_number=ui_number,
                                   question_no=question_no)


@pytest.mark.parametrize("feedback, mark", [
    ("correct", "✓"), ("wrong", "✗"), ("unknown", "?"), (None, "?"),
])
def test_line_shows_observed_feedback(feedback, mark):
    assert line(question(answer_feedback=feedback)) == f"  1. 题目 → 乙  {mark}"


def test_missing_feedback_is_not_assumed_correct():
    result = question()
    del result["answer_feedback"]
    assert line(result).endswith("?")


def test_unknown_number_is_marked_as_round():
    result = question(ui_number=None)
    assert line(result, ui_number=None, question_no=3).startswith("  第3轮.")


def test_last_question_uses_round_number_as_label():
    result = question(ui_number=None, ui_number_text="最后一题·分数翻倍")
    assert line(result, ui_number=None, question_no=10).startswith("  10.")


def test_guess_still_shows_feedback():
    text = line(question(guessed=True, answer_feedback="wrong", error="识别超时"))
    assert "蒙第一个" in text and "识别超时" in text
    assert text.endswith("✗")


def test_skipped_question_is_not_reported_as_answered():
    text = line(question(skipped=True, reason="screen_changed", answer=None))
    assert "跳过（screen_changed）" in text and "未点击" in text
    assert "✓" not in text and "答对" not in text


def test_bare_index_question_shows_the_ordinal():
    text = line(question(options=[], answer=1, bare_index=True))
    assert "模型只回序号 2" in text


class _FakeStream:
    def __init__(self):
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


def test_terminal_output_is_forced_to_utf8(monkeypatch):
    """GBK 环境把输出重定向到文件时，✓/⚠ 编不出来 —— 入口统一改 UTF-8。

    不改的话 print 会抛 UnicodeEncodeError，把答题循环打断在半路。
    """
    out, err = _FakeStream(), _FakeStream()
    monkeypatch.setattr(wq_main.sys, "stdout", out)
    monkeypatch.setattr(wq_main.sys, "stderr", err)
    wq_main._force_utf8_output()
    assert out.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert err.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_cli_entry_point_forces_utf8(monkeypatch):
    """cli.main() 也要改编码 —— 它可以不经 main 模块被直接调用。"""

    class _Stop(Exception):
        pass

    def _stop():
        raise _Stop

    monkeypatch.setattr(cli, "_force_utf8_output", _stop)
    with pytest.raises(_Stop):
        cli.main(["pk"])
