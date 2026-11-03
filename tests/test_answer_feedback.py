"""局部答后取色：不重新抓屏，不把其他选项高亮或下一题归到本题。"""
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from weread_quiz.models import Rect


BOX = Rect(60, 400, 310, 44)
BLUE = (13, 193, 255)
RED = (230, 65, 70)


def frame(color=(35, 55, 70), *, mode="pk", changed=False):
    image = Image.new("RGB", (430, 788), (10, 20, 40))
    draw = ImageDraw.Draw(image)
    y = 310 if mode == "pk" else 240
    draw.rectangle((100, y, 240 if not changed else 290, y + 10), fill="white")
    draw.rectangle((BOX.left, BOX.top, BOX.right - 1, BOX.bottom - 1), fill=color,
                   outline=BLUE, width=2)
    draw.rectangle((190, 415, 230, 425), fill="white")
    return image


def observer(mode="pk"):
    from weread_quiz.answer_feedback import AnswerFeedback
    feedback = AnswerFeedback(mode)
    feedback.start(frame(mode=mode), BOX)
    return feedback


@pytest.mark.parametrize("mode", ["pk", "daily"])
@pytest.mark.parametrize("color, expected", [(BLUE, "correct"), (RED, "wrong")])
def test_selected_button_fill(mode, color, expected):
    feedback = observer(mode)
    feedback.observe(frame(color, mode=mode))
    assert feedback.status == expected


def test_blue_border_is_not_correct():
    feedback = observer()
    feedback.observe(frame())
    assert feedback.status == "unknown"


def test_other_option_blue_does_not_hide_selected_red():
    image = frame(RED)
    ImageDraw.Draw(image).rectangle((60, 480, 369, 523), fill=BLUE)
    feedback = observer()
    feedback.observe(image)
    assert feedback.status == "wrong"


def test_other_option_blue_is_not_our_correct_answer():
    image = frame()
    ImageDraw.Draw(image).rectangle((60, 480, 369, 523), fill=BLUE)
    feedback = observer()
    feedback.observe(image)
    assert feedback.status == "unknown"


def test_new_question_expires_observer():
    feedback = observer()
    feedback.observe(frame(BLUE, changed=True))
    feedback.observe(frame(RED))
    assert feedback.status == "unknown"


def test_size_change_expires_observer():
    feedback = observer()
    feedback.observe(frame(BLUE).resize((440, 800)))
    feedback.observe(frame(RED))
    assert feedback.status == "unknown"


def test_result_is_latched():
    feedback = observer()
    feedback.observe(frame(RED))
    feedback.observe(frame(BLUE))
    assert feedback.status == "wrong"


def test_preexisting_blue_is_not_click_feedback():
    from weread_quiz.answer_feedback import AnswerFeedback
    feedback = AnswerFeedback("pk")
    feedback.start(frame(BLUE), BOX)
    feedback.observe(frame(BLUE))
    assert feedback.status == "unknown"


def test_small_blue_patch_is_not_enough():
    image = frame()
    ImageDraw.Draw(image).rectangle((100, 410, 130, 430), fill=BLUE)
    feedback = observer()
    feedback.observe(image)
    assert feedback.status == "unknown"


def test_unstarted_observer_does_not_add_result_for_skip_or_dry_run():
    from weread_quiz.answer_feedback import AnswerFeedback
    result = {"skipped": True}
    AnswerFeedback("daily").finish(result, verbose=False)
    assert "answer_feedback" not in result


def test_finish_records_once_without_changing_game_outcome():
    messages = []
    feedback = observer()
    feedback.observe(frame(RED))
    result = {"answer": 0}
    recorder = SimpleNamespace(log=messages.append)
    feedback.finish(result, recorder=recorder, verbose=False, label="第 1 题")
    feedback.finish(result, recorder=recorder, verbose=False, label="第 1 题")
    assert result == {"answer": 0, "answer_feedback": "wrong"}
    assert len(messages) == 1 and "错误" in messages[0]


def test_pk_wait_reuses_existing_frame(monkeypatch):
    from weread_quiz import pk
    feedback = observer()
    images = iter([frame(RED), frame(changed=True)])
    grabs = []
    monkeypatch.setattr(pk.time, "sleep", lambda _: None)
    monkeypatch.setattr(pk, "grab_screen", lambda _: grabs.append(True) or next(images))
    monkeypatch.setattr(pk, "has_answer_reveal", lambda im: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    current, _ = pk.wait_for_next_question(None, frame(), delay=0, feedback=feedback)
    assert feedback.status == "wrong"
    assert len(grabs) == 2
    assert current is not None


def test_daily_wait_reuses_existing_frame(monkeypatch):
    from weread_quiz import daily, daily_buttons
    from test_daily_pipeline import Clock
    feedback = observer("daily")
    clock = Clock()
    monkeypatch.setattr(daily.time, "monotonic", clock)
    monkeypatch.setattr(daily.time, "sleep", clock.sleep)
    grabs = []
    monkeypatch.setattr(daily, "grab_screen", lambda _: grabs.append(True) or frame(RED, mode="daily"))
    monkeypatch.setattr(daily_buttons, "_detect_any_buttons", lambda im: True)
    monkeypatch.setattr(daily, "_looks_like_revive_page", lambda im: False)
    monkeypatch.setattr(daily, "has_answer_reveal", lambda im: True)
    cfg = daily.DailyConfig(Rect(0, 0, 430, 788), 1.0, SimpleNamespace(), verbose=False)
    outcome = daily.wait_for_next_question(cfg, frame(mode="daily"), feedback=feedback)
    assert feedback.status == "wrong"
    assert outcome.status == "interrupted"  # 颜色不改变原来的终态判定
    assert len(grabs) > 0


def test_timeout_guess_uses_actual_fresh_frame_and_button(monkeypatch):
    from weread_quiz import pk
    from weread_quiz.answer_feedback import AnswerFeedback
    feedback = AnswerFeedback("pk")
    fresh = frame(changed=True)
    monkeypatch.setattr(pk, "grab_screen", lambda _: fresh)
    monkeypatch.setattr(pk, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "read_question_number", lambda *a, **k: (2, 10, "第2题·共10题"))
    result = pk._guess_after_timeout(None, 1.0, frame(), None, None, True, None,
                                     "timeout", feedback=feedback)
    feedback.observe(frame(RED, changed=True))
    feedback.finish(result, verbose=False)
    assert result["answer_feedback"] == "wrong"
    assert result["ui_number"] == 2


def test_daily_last_question_samples_the_settle_window(monkeypatch):
    """末题：结算等待期间连取几帧 —— 等 3 秒再去抓只剩渐隐的暗色。

    取帧时点与 PK 末题共用一份（answer_feedback.SAMPLE_OFFSETS）。
    """
    from weread_quiz import daily
    from test_daily_pipeline import Clock
    feedback = observer("daily")
    feedback.last_question = True
    clock = Clock()
    grabs = []
    sleeps = []

    def grab(region):
        grabs.append(True)
        return frame(BLUE, mode="daily")

    def sleep(seconds):
        sleeps.append(seconds)
        clock.sleep(seconds)

    monkeypatch.setattr(daily.time, "monotonic", clock)
    monkeypatch.setattr(daily.time, "sleep", sleep)
    monkeypatch.setattr(daily, "grab_screen", grab)
    monkeypatch.setattr(daily, "read_daily_result", lambda *a: "无")
    cfg = daily.DailyConfig(Rect(0, 0, 430, 788), 1.0, SimpleNamespace(), verbose=False)
    assert daily._read_final(cfg, feedback=feedback) == "final_pending"
    assert feedback.status == "correct"
    assert len(grabs) == len(daily._FINAL_SAMPLE_OFFSETS) + 1
    assert sum(sleeps) == pytest.approx(daily._FINAL_SETTLE_SECONDS), "取帧不能拉长结算等待"


def test_daily_click_starts_feedback_from_revalidated_button(monkeypatch):
    from weread_quiz import daily
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.vision import VisionQuestion
    feedback = AnswerFeedback("daily")
    monkeypatch.setattr(daily, "is_question_screen", lambda im: True)
    monkeypatch.setattr(daily, "is_same_question", lambda *a: True)
    monkeypatch.setattr(daily, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(daily, "grab_screen", lambda _: frame(mode="daily"))
    clicks = []
    monkeypatch.setattr(daily, "click_box", lambda box, *a, **k: clicks.append(box))
    cfg = daily.DailyConfig(Rect(0, 0, 430, 788), 1.0, SimpleNamespace(), verbose=False)
    question = VisionQuestion("题目", ["甲"], answer_index=0)
    result = daily._answer_one_question_inner(cfg, frame(mode="daily"), question,
                                              feedback=feedback)
    feedback.observe(frame(RED, mode="daily"))
    feedback.finish(result, verbose=False)
    assert clicks == [BOX]
    assert result["answer_feedback"] == "wrong"


def test_daily_last_question_captures_right_after_click(monkeypatch):
    """末题点击后立刻补一帧，才能赶上只亮一瞬的反馈色。"""
    from weread_quiz import daily
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.vision import VisionQuestion
    feedback = AnswerFeedback("daily")
    feedback.last_question = True
    frames = [frame(mode="daily"), frame(BLUE, mode="daily")]
    grabs = []
    monkeypatch.setattr(daily, "is_question_screen", lambda im: True)
    monkeypatch.setattr(daily, "is_same_question", lambda *a: True)
    monkeypatch.setattr(daily, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(daily, "grab_screen",
                        lambda _: grabs.append(True) or frames[min(len(grabs) - 1, 1)])
    monkeypatch.setattr(daily, "click_box", lambda *a, **k: None)
    cfg = daily.DailyConfig(Rect(0, 0, 430, 788), 1.0, SimpleNamespace(), verbose=False)
    question = VisionQuestion("题目", ["甲"], answer_index=0)
    daily._answer_one_question_inner(cfg, frame(mode="daily"), question,
                                      feedback=feedback)
    assert feedback.status == "correct"
    assert len(grabs) == 2


def test_daily_normal_click_does_not_capture_extra(monkeypatch):
    """非末题不多抓一帧 —— 换题等待阶段本来就有取帧。"""
    from weread_quiz import daily
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.vision import VisionQuestion
    feedback = AnswerFeedback("daily")
    feedback.last_question = False
    grabs = []
    monkeypatch.setattr(daily, "is_question_screen", lambda im: True)
    monkeypatch.setattr(daily, "is_same_question", lambda *a: True)
    monkeypatch.setattr(daily, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(daily, "grab_screen", lambda _: grabs.append(True) or frame(mode="daily"))
    monkeypatch.setattr(daily, "click_box", lambda *a, **k: None)
    cfg = daily.DailyConfig(Rect(0, 0, 430, 788), 1.0, SimpleNamespace(), verbose=False)
    question = VisionQuestion("题目", ["甲"], answer_index=0)
    daily._answer_one_question_inner(cfg, frame(mode="daily"), question,
                                      feedback=feedback)
    assert len(grabs) == 1, "非末题只应有点击前复核那一次抓屏"
    assert feedback.status == "unknown"


def test_daily_last_question_dumps_the_sample_window():
    """末题抓不到时要把整段取帧窗口落盘 —— 一帧看不出颜色有没有出现过。"""
    from weread_quiz.answer_feedback import AnswerFeedback
    feedback = AnswerFeedback("daily")
    feedback.last_question = True
    feedback.start(frame(mode="daily"), BOX)
    feedback.observe(frame(mode="daily"))
    feedback.observe(frame(mode="daily"))
    shots = []
    recorder = SimpleNamespace(log=lambda message: None,
                               shot=lambda label, image: shots.append((label, image)))
    feedback.finish({}, recorder=recorder, verbose=False, terminal=True)
    assert [label for label, _ in shots] == ["answer_feedback_window_1",
                                             "answer_feedback_window_2"]


def test_run_daily_marks_the_last_question_for_early_capture(monkeypatch):
    """末题标记必须跟着 max_questions 走 —— 标错了就抓不到反馈色。"""
    from weread_quiz import daily

    created = []

    class FakeFeedback:
        def __init__(self, mode):
            self.mode = mode
            self.last_question = False
            created.append(self)

        def start(self, before, box):
            pass

        def observe(self, image):
            pass

        def observe_now(self, grab):
            pass

        def finish(self, result, **kwargs):
            result["answer_feedback"] = "correct"

    def fake_answer(cfg, image, question, question_no, watch, *,
                    capture_context=None, feedback=None):
        return {"stem": "题目", "options": ["甲"], "answer": 0,
                "progress_no": question_no}

    monkeypatch.setattr(daily, "AnswerFeedback", FakeFeedback)
    monkeypatch.setattr(daily, "wait_for_question",
                        lambda cfg, capture_context=None: (frame(mode="daily"), None))
    monkeypatch.setattr(daily, "answer_one_question", fake_answer)
    monkeypatch.setattr(daily, "wait_for_next_question",
                        lambda cfg, current, watch=None, feedback=None:
                        daily._WaitOutcome("next", frame(mode="daily")))
    monkeypatch.setattr(daily, "_read_final", lambda cfg, feedback=None: "success")
    cfg = daily.DailyConfig(Rect(0, 0, 430, 788), 1.0, SimpleNamespace(),
                             verbose=False, max_questions=3)
    report = daily.run_daily(cfg)
    assert report.outcome == "success"
    assert [f.last_question for f in created] == [False, False, True]
    assert all(q["answer_feedback"] == "correct" for q in report.questions)


def test_unknown_terminal_feedback_keeps_first_existing_frame(capsys):
    messages, shots = [], []
    feedback = observer()
    rejected = frame(BLUE, changed=True)
    feedback.observe(rejected)
    feedback.observe(frame())
    result = {}
    recorder = SimpleNamespace(log=messages.append,
                               shot=lambda label, image: shots.append((label, image)))
    feedback.finish(result, recorder=recorder, verbose=False, terminal=True)
    assert result["answer_feedback"] == "unknown"
    assert any("题干变化" in line and "蓝点18/18" in line for line in messages)
    assert len(shots) == 1 and shots[0][1] is rejected
    feedback.finish(result, recorder=recorder, verbose=False, terminal=True)
    assert len(shots) == 1


def test_unknown_nonterminal_feedback_does_not_write_images():
    feedback = observer()
    feedback.observe(frame())
    messages = []
    recorder = SimpleNamespace(log=messages.append,
                               shot=lambda *a, **k: pytest.fail("正常换题不新增落盘"))
    feedback.finish({}, recorder=recorder, verbose=False)
    assert any("颜色未明确变化" in line for line in messages)


def test_diagnostic_image_failure_does_not_change_result():
    feedback = observer()
    feedback.observe(frame(BLUE, changed=True))

    def broken_shot(*a, **k):
        raise OSError("disk unavailable")

    messages = []
    result = {}
    feedback.finish(result, recorder=SimpleNamespace(log=messages.append, shot=broken_shot),
                    verbose=False, terminal=True)
    assert result["answer_feedback"] == "unknown"
    assert any("留证失败" in line for line in messages)


def test_dimmed_frame_is_not_mistaken_for_wrong():
    """整页渐隐会把蓝底压暗，但暗不等于红 —— 不能判成答错。"""
    dim = frame()
    pixels = dim.load()
    for y in range(dim.size[1]):
        for x in range(dim.size[0]):
            r, g, b = pixels[x, y]
            pixels[x, y] = (r // 4, g // 4, b // 4)
    feedback = observer()
    feedback.observe(dim)
    assert feedback.status == "unknown"


def test_partially_faded_reveal_is_detected_by_hue_shift():
    """颜色渐显到一半时绝对亮度不够，但相对基线明显偏蓝 —— 仍应判答对。"""
    faded = frame()
    pixels = faded.load()
    for x, y in [(BOX.left + int(BOX.width * fx), BOX.top + int(BOX.height * fy))
                 for fx in (0.15, 0.25, 0.35, 0.65, 0.75, 0.85)
                 for fy in (0.25, 0.5, 0.75)]:
        r, g, b = pixels[x, y]
        pixels[x, y] = (r + 6, g + 60, b + 70)
    feedback = observer()
    feedback.observe(faded)
    assert feedback.status == "correct"


def test_uniform_brighten_is_not_click_feedback():
    """按下高亮只是整体提亮，色相不变 —— 不能当成答对。"""
    brighter = frame()
    pixels = brighter.load()
    for x, y in [(BOX.left + int(BOX.width * fx), BOX.top + int(BOX.height * fy))
                 for fx in (0.15, 0.25, 0.35, 0.65, 0.75, 0.85)
                 for fy in (0.25, 0.5, 0.75)]:
        r, g, b = pixels[x, y]
        pixels[x, y] = (r + 30, g + 30, b + 30)
    feedback = observer()
    feedback.observe(brighter)
    assert feedback.status == "unknown"


def test_pk_wait_captures_feedback_before_the_fixed_delay(monkeypatch):
    """末题：固定延迟走完之前就要取到那一帧 —— 之后只剩渐隐的暗色。"""
    from weread_quiz import pk
    from test_daily_pipeline import Clock
    feedback = observer()
    feedback.last_question = True
    # 揭示帧出现在固定延迟之前；再往后抓到的都是换题后的画面。
    frames = [frame(), frame(BLUE), frame(changed=True)]
    grabs = []
    sleeps = []
    state = {"i": 0}
    clock = Clock()

    def grab(region):
        grabs.append(True)
        image = frames[min(state["i"], len(frames) - 1)]
        state["i"] += 1
        return image

    def sleep(seconds):
        sleeps.append(seconds)
        clock.sleep(seconds)

    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk.time, "sleep", sleep)
    monkeypatch.setattr(pk, "grab_screen", grab)
    monkeypatch.setattr(pk, "has_answer_reveal", lambda im: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "is_new_question", lambda a, b: True)
    pk.wait_for_next_question(None, frame(), delay=1.0, feedback=feedback)
    assert feedback.status == "correct"
    assert len(grabs) >= 2
    assert sum(sleeps) == pytest.approx(1.0), "提前取帧不能拉长总等待时间"


def test_pk_last_question_samples_the_whole_window(monkeypatch):
    """末题要在渐隐之前连着取几帧 —— 揭示色要等服务端回包才出现。"""
    from weread_quiz import pk
    from test_daily_pipeline import Clock
    feedback = observer()
    feedback.last_question = True
    grabs = []
    sleeps = []
    clock = Clock()

    def sleep(seconds):
        sleeps.append(seconds)
        clock.sleep(seconds)

    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk.time, "sleep", sleep)
    monkeypatch.setattr(pk, "grab_screen", lambda r: grabs.append(True) or frame())
    monkeypatch.setattr(pk, "has_answer_reveal", lambda im: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "is_new_question", lambda a, b: True)
    pk.wait_for_next_question(None, frame(), delay=1.0, feedback=feedback)
    assert len(grabs) == len(pk._FEEDBACK_SAMPLE_OFFSETS) + 1
    assert sum(sleeps) == pytest.approx(1.0), "总等待时间不变"


def test_pk_wait_for_normal_question_does_not_capture_extra(monkeypatch):
    """非末题：不多抓一帧 —— 开销和改之前一样。"""
    from weread_quiz import pk
    feedback = observer()
    feedback.last_question = False
    grabs = []
    sleeps = []
    monkeypatch.setattr(pk.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(pk, "grab_screen", lambda r: grabs.append(True) or frame())
    monkeypatch.setattr(pk, "has_answer_reveal", lambda im: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "is_new_question", lambda a, b: True)
    pk.wait_for_next_question(None, frame(), delay=1.0, feedback=feedback)
    assert len(grabs) == 1
    assert sum(sleeps) == 1.0


def test_last_question_is_detected_from_progress_text():
    from weread_quiz.pk import _is_last_question
    assert _is_last_question("最后一题·分数翻倍")
    assert not _is_last_question("第9题·共10题")
    assert not _is_last_question("")


def test_pk_normal_question_click_does_not_capture_extra(monkeypatch):
    """非末题点击后不多抓一帧。"""
    from weread_quiz import pk
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.vision import VisionQuestion
    feedback = AnswerFeedback("pk")
    grabs = []
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "is_same_question", lambda *a: True)
    monkeypatch.setattr(pk, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(pk, "grab_screen", lambda _: grabs.append(True) or frame())
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    question = VisionQuestion("题目", ["甲"], answer_index=0)
    pk.answer_one_question(None, 1.0, SimpleNamespace(usage=None), image=frame(),
                            question=question, captured_at=pk.time.monotonic(),
                            verbose=False, feedback=feedback)
    assert len(grabs) == 1, "非末题只应有点击前复核那一次抓屏"


def test_pk_last_question_click_captures_right_after_click(monkeypatch):
    """末题点击后立刻补一帧，才能赶上只亮一瞬的反馈色。"""
    from weread_quiz import pk
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.ocr_core import OcrRead
    from weread_quiz.vision import VisionQuestion
    feedback = AnswerFeedback("pk")
    frames = [frame(), frame(BLUE)]
    grabs = []
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "is_same_question", lambda *a: True)
    monkeypatch.setattr(pk, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(pk, "grab_screen",
                        lambda _: grabs.append(True) or frames[min(len(grabs) - 1, 1)])
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    ocr_read = OcrRead(stem="题目", options=["甲"], confidences=[0.99],
                        button_count=1, has_progress=True,
                        progress_text="最后一题·分数翻倍")
    question = VisionQuestion("题目", ["甲"], answer_index=0)
    pk.answer_one_question(None, 1.0, SimpleNamespace(usage=None), image=frame(),
                            question=question, ocr_read=ocr_read,
                            captured_at=pk.time.monotonic(), verbose=False,
                            feedback=feedback)
    assert feedback.status == "correct"
    assert len(grabs) == 2


def test_pk_wait_without_feedback_keeps_single_capture(monkeypatch):
    from weread_quiz import pk
    grabs = []
    sleeps = []
    monkeypatch.setattr(pk.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(pk, "grab_screen", lambda r: grabs.append(True) or frame())
    monkeypatch.setattr(pk, "has_answer_reveal", lambda im: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "is_new_question", lambda a, b: True)
    pk.wait_for_next_question(None, frame(), delay=1.0)
    assert len(grabs) == 1
    assert sum(sleeps) == 1.0


def test_pk_normal_click_starts_feedback(monkeypatch):
    from weread_quiz import pk
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.vision import VisionQuestion
    feedback = AnswerFeedback("pk")
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "is_same_question", lambda *a: True)
    monkeypatch.setattr(pk, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(pk, "grab_screen", lambda _: frame())
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    question = VisionQuestion("题目", ["甲"], answer_index=0)
    result = pk.answer_one_question(None, 1.0, SimpleNamespace(usage=None), image=frame(),
                                    question=question, captured_at=pk.time.monotonic(),
                                    verbose=False, feedback=feedback)
    feedback.observe(frame(BLUE))
    feedback.finish(result, verbose=False)
    assert result["answer_feedback"] == "correct"


def test_pk_last_question_detected_when_progress_read_during_model_call(monkeypatch):
    """题号行常常是**模型调用期间**才被本地 OCR 读到的。

    末题判定必须用调用之后的原文 —— 开头取一次快照的话，末题永远判不出来，
    点击后就不会提前取帧，唯一一次取色落在固定延迟之后（那时页面已渐隐）。
    """
    from weread_quiz import pk
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.errors import RecognitionTimeout

    class Vision:
        usage = None
        last_question_progress_text = ""
        last_question_local_finished_at = None
        last_question_model_started_at = None

        def read_question_with_budget(self, image, prompt=None, *, budget):
            # 本地 OCR 在这一步读出题号行；随后模型超时。
            self.last_question_progress_text = "最后一题·分数翻倍"
            raise RecognitionTimeout("deadline_exceeded")

    feedback = AnswerFeedback("pk")
    grabs = []
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(pk, "grab_screen", lambda r: grabs.append(True) or frame())
    monkeypatch.setattr(pk, "click_box", lambda *a, **k: None)
    monkeypatch.setattr(pk, "read_question_number",
                        lambda *a, **k: (None, None, "最后一题·分数翻倍"))
    context = pk.PKCaptureContext()
    pk.answer_one_question(None, 1.0, Vision(), image=frame(), question=None,
                            captured_at=pk.time.monotonic(), verbose=False,
                            feedback=feedback, capture_context=context)
    assert feedback.last_question is True, "模型调用期间读到的题号行必须参与末题判定"
    assert len(grabs) == 2, "末题点击后应立刻补一帧"


def test_unknown_terminal_feedback_keeps_the_most_colorful_frame():
    """留证要留**颜色证据最强**的那一帧 —— 第一帧通常还没变色。"""
    from weread_quiz.answer_feedback import AnswerFeedback
    feedback = AnswerFeedback("pk")
    feedback.start(frame(), BOX)
    plain = frame()
    feedback.observe(plain)
    colored = frame(BLUE, changed=True)
    feedback.observe(colored)
    shots = []
    recorder = SimpleNamespace(log=lambda message: None,
                               shot=lambda label, image: shots.append((label, image)))
    feedback.finish({}, recorder=recorder, verbose=False, terminal=True)
    assert len(shots) == 1
    assert shots[0][1] is colored


def test_unknown_terminal_last_question_dumps_the_sample_window():
    """末题抓不到时要把整段取帧窗口落盘 —— 一帧看不出颜色有没有出现过。"""
    from weread_quiz.answer_feedback import AnswerFeedback
    feedback = AnswerFeedback("pk")
    feedback.last_question = True
    feedback.start(frame(), BOX)
    feedback.observe(frame())
    feedback.observe(frame())
    shots = []
    recorder = SimpleNamespace(log=lambda message: None,
                               shot=lambda label, image: shots.append((label, image)))
    feedback.finish({}, recorder=recorder, verbose=False, terminal=True)
    assert [label for label, _ in shots] == ["answer_feedback_window_1",
                                             "answer_feedback_window_2"]


TEAL = (38, 186, 171)


def test_last_question_reveal_is_captured_end_to_end(monkeypatch):
    """末题端到端：揭示色在点击后 0.35 秒出现、0.75 秒后页面渐隐 —— 必须抓到。

    揭示色要等服务端回包才出现，末题页面却很快就开始渐隐转场。
    这条用例把「点击 → 末题判定 → 提前取帧 → 取色」整条链路走一遍。
    """
    from weread_quiz import pk
    from weread_quiz.answer_feedback import AnswerFeedback
    from weread_quiz.errors import RecognitionTimeout
    from test_daily_pipeline import Clock

    clock = Clock()
    clicked = {"at": None}

    def screen(elapsed):
        if elapsed < 0.35:
            return frame()        # 还没回包，按钮是原色
        if elapsed < 0.75:
            return frame(TEAL)    # 揭示色（青绿）
        return frame((7, 17, 32))  # 页面已渐隐，只剩暗色

    def grab(region):
        if clicked["at"] is None:
            return frame()
        return screen(clock.now - clicked["at"])

    class Vision:
        usage = None
        last_question_progress_text = ""
        last_question_local_finished_at = None
        last_question_model_started_at = None

        def read_question_with_budget(self, image, prompt=None, *, budget):
            # 本地 OCR 在这一步读出题号行；随后模型超时。
            self.last_question_progress_text = "最后一题·分数翻倍"
            raise RecognitionTimeout("deadline_exceeded")

    def click(box, *args, **kwargs):
        clicked["at"] = clock.now
        clock.sleep(0.15)  # 真点击会停一会儿

    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk.time, "sleep", clock.sleep)
    monkeypatch.setattr(pk, "grab_screen", grab)
    monkeypatch.setattr(pk, "click_box", click)
    monkeypatch.setattr(pk, "is_question_screen", lambda im: True)
    monkeypatch.setattr(pk, "find_option_buttons", lambda im: [BOX])
    monkeypatch.setattr(pk, "read_question_number",
                        lambda *a, **k: (None, None, "最后一题·分数翻倍"))
    monkeypatch.setattr(pk, "has_answer_reveal", lambda im: False)
    monkeypatch.setattr(pk, "is_new_question", lambda a, b: True)

    feedback = AnswerFeedback("pk")
    context = pk.PKCaptureContext()
    result = pk.answer_one_question(None, 1.0, Vision(), image=frame(), question=None,
                                    captured_at=clock.now, verbose=False,
                                    feedback=feedback, capture_context=context)
    pk.wait_for_next_question(None, frame(), delay=1.0, feedback=feedback)
    feedback.finish(result, verbose=False)
    assert result["answer_feedback"] == "correct"
