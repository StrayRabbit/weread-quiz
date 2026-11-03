"""逐题记录行：题号、题干、答案与答后反馈合成一行。

PK 和每日一答共用同一套终端逐题输出 —— **每题只打一行**，
置信度、走的是哪条识别路径、分段耗时这些细节留在 run.log 里。

**没确认就是没确认** —— 反馈没观察到就标「?」，不能因为没标错就当成答对。
"""

from __future__ import annotations

# 逐题行的反馈标记：✓ 答对、✗ 答错、? 没观察到反馈、- 没作答（跳过）。
FEEDBACK_MARKS = {"correct": "✓", "wrong": "✗"}


def question_label(result: dict, ui_number: int | None, question_no: int) -> str:
    """题号标签：优先界面自己写的题号，读不出才退回处理轮次。"""
    if ui_number is not None:
        return f"{ui_number}."
    if "最后一题" in (result.get("ui_number_text") or ""):
        # 末题没有数字题号 —— 用处理轮次占位。
        return f"{question_no}."
    return f"第{question_no}轮."


def question_choice(result: dict) -> str:
    """逐题行里「→」后面的内容：选了哪个 / 蒙了 / 跳过了。"""
    if result.get("skipped"):
        # reason 是 PK 的字段名，error 是每日一答的字段名 —— 两个都认。
        reason = result.get("reason") or result.get("error") or "界面已变"
        return f"跳过（{reason}），未点击"
    if result.get("guessed"):
        error = (result.get("error") or "识别失败")[:40]
        return f"蒙第一个（{error}）"
    options = result.get("options") or []
    answer = result.get("answer")
    if result.get("bare_index"):
        return (f"模型只回序号 {answer + 1}"
                if isinstance(answer, int) else "模型只回序号 ?")
    if isinstance(answer, int) and 0 <= answer < len(options):
        return options[answer]
    return "?"


def format_question_line(result: dict, *, ui_number: int | None,
                         question_no: int, mark: str | None = None) -> str:
    """把一道题压成一行：题号. 题干 → 答案 ✓。

    标记：✓ 答对、✗ 答错、? 没观察到反馈、- 没作答（跳过）。
    mark 可以显式指定（dry-run 没点击，用它说明「未点击」）。
    """
    stem = (result.get("stem") or "").strip() or "(模型未给题干)"
    if mark is None:
        mark = ("-" if result.get("skipped")
                else FEEDBACK_MARKS.get(result.get("answer_feedback"), "?"))
    return (f"  {question_label(result, ui_number, question_no)} {stem}"
            f" → {question_choice(result)}  {mark}")
