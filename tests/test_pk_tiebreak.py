"""PK 平局加时赛：常规 10 题之后界面继续出题，程序要接着答。"""

from __future__ import annotations

from PIL import Image

from weread_quiz.pk import runner as pk


def progress_plan(total_rounds: int, *, tiebreak_total: int | None = None):
    """构造一局的题号行序列：1~9 数字题、10 是「最后一题」，之后是加时赛。"""
    plan = []
    for number in range(1, 10):
        plan.append((number, 10, f"第{number}题·共10题"))
    plan.append((None, 10, "最后一题·分数翻倍"))
    for extra in range(11, total_rounds + 1):
        total = tiebreak_total if tiebreak_total is not None else 10
        plan.append((extra, total, f"第{extra}题·共{total}题"))
    return plan


class FakeGame:
    """按脚本回放题号行；answers_after 之后界面变成结算页。"""

    def __init__(self, plan, answers_after: int):
        self.plan = plan
        self.answers_after = answers_after
        self.index = -1
        self.image = Image.new("RGB", (430, 788), (20, 35, 70))
        self.finals = 0

    def answer(self, *args, **kwargs):
        self.index += 1
        number, total, text = self.plan[self.index]
        return {
            "stem": f"第{self.index + 1}轮题干",
            "options": ["甲", "乙"],
            "answer": 0,
            "guessed": False,
            "skipped": False,
            "bare_index": False,
            "error": "",
            "ui_number": number,
            "ui_total": total,
            "ui_number_text": text,
        }

    def read_number(self, image, ocr=None):
        number, total, text = self.plan[self.index]
        return number, total, text

    def wait(self, region, vision, *, capture_context, **kwargs):
        capture_context.captured_at = 100.0
        return self.image, None

    def next_question(self, region, current, *, capture_context, **kwargs):
        capture_context.reset()
        capture_context.captured_at = 100.0
        if self.index + 1 >= self.answers_after:
            return None, None
        return self.image, None

    def read_final(self, *args, **kwargs):
        self.finals += 1


def run(monkeypatch, game, *, max_questions=None, verbose=False):
    monkeypatch.setattr(pk, "wait_for_question", game.wait)
    monkeypatch.setattr(pk, "wait_for_next_question", game.next_question)
    monkeypatch.setattr(pk, "answer_one_question", game.answer)
    monkeypatch.setattr(pk, "read_question_number", game.read_number)
    monkeypatch.setattr(pk, "_read_final", game.read_final)
    if max_questions is None:
        # 不传上限 —— 走 run_pk 的默认值（启动器也是这条路径）。
        return pk.run_pk((0, 0, 430, 788), object(), verbose=verbose)
    return pk.run_pk((0, 0, 430, 788), object(), max_questions=max_questions,
                     verbose=verbose)


def test_default_round_cap_leaves_room_for_tiebreak():
    """默认轮次上限必须大于常规的 10 题 —— 否则加时赛一题都答不到。"""
    assert pk._MAX_QUESTIONS > 10
    assert pk._NORMAL_QUESTIONS == 10


def test_cli_default_matches_the_run_default():
    """启动器不传 --max-questions，走的就是这个默认值。"""
    from weread_quiz import cli

    args = cli._build_parser().parse_args(["pk"])
    assert args.max_questions == pk._MAX_QUESTIONS


def test_tiebreak_questions_are_answered_after_the_tenth(monkeypatch):
    """常规 10 题之后界面继续出题 —— 加时赛那几题也要答。"""
    game = FakeGame(progress_plan(12), answers_after=12)
    report = run(monkeypatch, game)
    assert report.answered == 12, "平局加时赛的题不能停在第 10 轮"
    assert report.ui_numbers == [1, 2, 3, 4, 5, 6, 7, 8, 9, None, 11, 12]
    assert game.finals == 1, "结算页只应读一次"


def test_tiebreak_does_not_report_the_last_question_as_missing(monkeypatch):
    """「最后一题」没有数字题号，但它答过了 —— 不能误报漏答第 10 题。"""
    game = FakeGame(progress_plan(12), answers_after=12)
    report = run(monkeypatch, game)
    assert report.missing_questions() == []
    assert report.duplicate_questions() == []


def test_tiebreak_with_grown_total_still_has_no_missing(monkeypatch):
    """加时赛把总题数改大了，已答题号要照着新的总数算。"""
    game = FakeGame(progress_plan(12, tiebreak_total=12), answers_after=12)
    report = run(monkeypatch, game)
    assert report.ui_total() == 12
    assert report.missing_questions() == []


def test_normal_game_stops_at_ten_questions(monkeypatch):
    """没进加时赛时，结算页出现就停 —— 不能多答。"""
    game = FakeGame(progress_plan(10), answers_after=10)
    report = run(monkeypatch, game)
    assert report.answered == 10
    assert report.error == ""


def test_tiebreak_notice_is_printed_once(monkeypatch, capsys):
    """加时赛提示只打一次，别把终端刷满。"""
    game = FakeGame(progress_plan(12), answers_after=12)
    run(monkeypatch, game, verbose=True)
    assert capsys.readouterr().out.count("之后界面仍出题") == 1


def test_no_tiebreak_notice_in_a_normal_game(monkeypatch, capsys):
    """正常一局不该出现加时赛提示。"""
    game = FakeGame(progress_plan(10), answers_after=10)
    run(monkeypatch, game, verbose=True)
    assert "之后界面仍出题" not in capsys.readouterr().out


def test_settlement_with_buttons_does_not_start_eleventh_answer(monkeypatch, capsys):
    """跑真实换题等待逻辑：结算双按钮不能产生加时提示或第 11 次答题调用。"""
    wait_next = pk.wait_for_next_question
    game = FakeGame(progress_plan(10), answers_after=10)
    settlement = Image.new("RGB", (430, 788))
    now = [100.0]
    checks = []

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(pk.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pk.time, "sleep", sleep)
    monkeypatch.setattr(pk, "grab_screen",
                        lambda _region: settlement if game.index >= 9 else game.image)
    monkeypatch.setattr(pk, "is_question_screen", lambda _image: True)
    monkeypatch.setattr(pk, "looks_like_question_template",
                        lambda image: image is not settlement)
    monkeypatch.setattr(pk, "has_score_row", lambda image: image is not settlement)
    monkeypatch.setattr(pk, "has_answer_reveal", lambda _image: False)
    monkeypatch.setattr(pk, "is_new_question", lambda *_args: True)

    def confirm(image, _ocr):
        checks.append(image)
        return image is settlement

    monkeypatch.setattr(pk, "_is_settlement_page", confirm)
    game.next_question = wait_next
    report = run(monkeypatch, game, verbose=True)
    assert report.answered == 10
    assert game.index == 9
    assert game.finals == 1
    assert checks
    assert "之后界面仍出题" not in capsys.readouterr().out


def test_round_cap_reached_while_still_answering(monkeypatch):
    """轮次用尽时界面还是答题页 —— 要说明对局没结束，别当成答完了。"""
    game = FakeGame(progress_plan(20), answers_after=20)
    report = run(monkeypatch, game, max_questions=12)
    assert report.answered == 12
    assert "对局未结束" in report.error


def test_missing_questions_with_tiebreak_rounds():
    """加时赛轮次接在后面时，「最后一题」仍算已答，不能报漏答第 10 题。"""
    report = pk.PKReport()
    report.questions = [
        {"ui_number": n, "ui_total": 10, "ui_number_text": f"第{n}题·共10题"}
        for n in range(1, 10)
    ] + [
        {"ui_number": None, "ui_total": 10, "ui_number_text": "最后一题·分数翻倍"},
        {"ui_number": 11, "ui_total": 10, "ui_number_text": "第11题·共10题"},
        {"ui_number": 12, "ui_total": 10, "ui_number_text": "第12题·共10题"},
    ]
    assert report.missing_questions() == []


def test_renumbered_tiebreak_is_not_a_duplicate():
    """加时赛若从头编号，同号不算重复处理。"""
    report = pk.PKReport()
    report.questions = [{"ui_number": n} for n in (1, 2, 3, 1, 2)]
    assert report.duplicate_questions() == []


def test_same_number_twice_is_still_a_duplicate():
    """同一题号被处理两次仍然要报出来。"""
    report = pk.PKReport()
    report.questions = [{"ui_number": 6}, {"ui_number": 6}]
    assert report.duplicate_questions() == [6]


def test_skipped_last_question_is_reported_as_missing():
    """「最后一题」被跳过时，它占的题号仍算漏答。"""
    report = pk.PKReport()
    report.questions = [
        {"ui_number": n, "ui_total": 10, "ui_number_text": f"第{n}题·共10题"}
        for n in range(1, 10)
    ] + [
        {"ui_number": None, "ui_total": 10, "ui_number_text": "最后一题·分数翻倍",
         "skipped": True},
    ]
    assert report.missing_questions() == [10]
