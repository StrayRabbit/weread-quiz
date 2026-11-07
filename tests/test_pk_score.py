"""PK 比分读取：左右分数、VS 被并进同一行、读不出就是读不出。"""

from PIL import Image
import pytest

from weread_quiz.pk.score import read_score


class Ocr:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def _engine_run(self, image):
        self.calls += 1
        return self.rows


def row(text, x, y=130, width=30):
    return ([[x, y], [x + width, y], [x + width, y + 20], [x, y + 20]], text, 0.99)


def frame():
    return Image.new("RGB", (430, 788))


def test_reads_the_two_scores_around_vs():
    ocr = Ocr([row("125", 114), row("VS", 170, width=20), row("60", 272)])
    assert read_score(frame(), ocr) == (125, 60)


def test_handles_score_merged_with_vs():
    """OCR 偶尔把分数和 VS 合成一行（"120 VS"）。"""
    ocr = Ocr([row("120 VS", 108, width=90), row("65", 276)])
    assert read_score(frame(), ocr) == (120, 65)


def test_ignores_rows_outside_the_score_band():
    ocr = Ocr([row("125", 114, y=300), row("VS", 170, y=300, width=20),
               row("60", 272, y=300)])
    assert read_score(frame(), ocr) == (None, None)


def test_missing_score_reads_as_unknown():
    ocr = Ocr([row("获得积分", 150, y=440), row("+1", 240, y=440)])
    assert read_score(frame(), ocr) == (None, None)


def test_no_ocr_engine_reads_as_unknown():
    assert read_score(frame(), object()) == (None, None)


def test_ocr_failure_reads_as_unknown():
    class Broken:
        def _engine_run(self, image):
            raise RuntimeError("boom")

    assert read_score(frame(), Broken()) == (None, None)


def test_wait_keeps_the_last_card_frame_for_the_score(monkeypatch):
    """结算页上没有比分 —— 结算前最后一张答题卡片要留着。

    胜负只能靠那张卡片上的「我 VS 对手」判。
    """
    from weread_quiz.pk import runner as pk

    card = object()
    settle = object()
    grabbed = []

    def grab(region):
        grabbed.append(True)
        return card if len(grabbed) == 1 else settle

    monkeypatch.setattr(pk.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pk, "grab_screen", grab)
    monkeypatch.setattr(pk, "has_answer_reveal", lambda image: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda image: image is card)
    monkeypatch.setattr(pk, "has_score_row", lambda image: False)
    monkeypatch.setattr(pk, "is_new_question", lambda a, b: False)
    context = pk.PKCaptureContext()
    image, _ = pk.wait_for_next_question(None, object(), delay=0,
                                          capture_context=context)
    assert image is None, "结算页 → 没有下一题"
    assert context.final_frame is card, "最后那张答题卡片要留着判比分"


@pytest.mark.parametrize("score_card", [True, False, None])
def test_settlement_buttons_do_not_overwrite_score_frame(monkeypatch, score_card):
    """卡片模板失配/不可用时，仍可用比分行留帧；结算按钮不能覆盖它。"""
    from weread_quiz.pk import runner as pk

    score_frame, settlement = object(), object()
    frames = iter([score_frame, settlement, settlement])
    monkeypatch.setattr(pk.time, "sleep", lambda _: None)
    monkeypatch.setattr(pk, "grab_screen", lambda _: next(frames, settlement))
    monkeypatch.setattr(pk, "has_answer_reveal", lambda _: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda _: True)
    monkeypatch.setattr(pk, "looks_like_question_template",
                        lambda image: score_card if image is score_frame else False)
    monkeypatch.setattr(pk, "has_score_row", lambda image: image is score_frame)
    monkeypatch.setattr(pk, "is_new_question", lambda *_: False)
    monkeypatch.setattr(pk, "_is_settlement_page", lambda image, _: image is settlement)
    context = pk.PKCaptureContext()
    current, _ = pk.wait_for_next_question(None, object(), delay=0,
                                          capture_context=context)
    assert current is None
    assert context.final_frame is score_frame


def test_starting_on_settlement_does_not_save_buttons_as_score_frame(monkeypatch):
    from weread_quiz.pk import runner as pk

    settlement = object()
    monkeypatch.setattr(pk.time, "sleep", lambda _: None)
    monkeypatch.setattr(pk, "grab_screen", lambda _: settlement)
    monkeypatch.setattr(pk, "has_answer_reveal", lambda _: False)
    monkeypatch.setattr(pk, "is_question_screen", lambda _: True)
    monkeypatch.setattr(pk, "looks_like_question_template", lambda _: False)
    monkeypatch.setattr(pk, "has_score_row", lambda _: False)
    monkeypatch.setattr(pk, "is_new_question", lambda *_: True)
    monkeypatch.setattr(pk, "_is_settlement_page", lambda *_: True)
    context = pk.PKCaptureContext()
    current, _ = pk.wait_for_next_question(None, object(), delay=0,
                                          capture_context=context)
    assert current is None
    assert context.final_frame is None, "没有有效比分帧时保持未知，不能拿结算按钮凑数"


def _last_question_transition(monkeypatch, *, fade_at=0.75, settle_at=1.2):
    from PIL import ImageDraw
    from weread_quiz.answer_feedback import AnswerFeedback, SAMPLE_OFFSETS
    from weread_quiz.models import Rect
    from weread_quiz.pk import runner as pk
    from test_pk_wait_next_question import Clock

    clock = Clock()
    images = {}
    for name, score in (("before", (110, 55)), ("early", (110, 55)),
                        ("latest", (140, 55)), ("fading", (140, 85)),
                        ("settlement", None)):
        images[name] = Image.new("RGB", (430, 788), (10, 20, 40))
        images[name].info.update(state=name, score=score)
    box = Rect(60, 400, 310, 44)
    ImageDraw.Draw(images["latest"]).rectangle(
        (box.left, box.top, box.right - 1, box.bottom - 1), fill=(13, 193, 255))
    feedback = AnswerFeedback("pk")
    feedback.last_question = True
    feedback.start(images["before"], box)
    grabs = []

    def grab(_region):
        elapsed = clock.now - 100.0
        grabs.append(elapsed)
        state = ("settlement" if elapsed >= settle_at else "fading" if elapsed >= fade_at
                 else "latest" if elapsed >= 0.35 else "early")
        return images[state]

    def visible(image):
        return image.info["state"] in {"before", "early", "latest"}

    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk.time, "sleep", clock.sleep)
    monkeypatch.setattr(pk, "grab_screen", grab)
    monkeypatch.setattr(pk, "has_answer_reveal", lambda _: False)
    monkeypatch.setattr(pk, "has_score_row", visible)
    monkeypatch.setattr(pk, "looks_like_question_template", visible)
    monkeypatch.setattr(pk, "is_question_screen",
                        lambda image: image.info["state"] != "fading")
    monkeypatch.setattr(pk, "is_new_question", lambda *_: False)
    monkeypatch.setattr(pk, "_is_settlement_page",
                        lambda image, _: image.info["state"] == "settlement")
    context = pk.PKCaptureContext()
    current, _ = pk.wait_for_next_question(
        None, images["before"], delay=1.0, feedback=feedback, capture_context=context)
    assert current is None
    assert len([at for at in grabs if at < 1.0 - 1e-6]) == len(SAMPLE_OFFSETS)
    assert grabs[len(SAMPLE_OFFSETS)] == pytest.approx(1.0), "固定采样窗口不能被拉长"
    expected_wait = 1.2 if settle_at < 1.0 else 2.3
    assert clock.now - 100.0 == pytest.approx(expected_wait), "不改变既有结算确认等待"
    return context, images, feedback


class FrameScoreOcr:
    def _engine_run(self, image):
        score = image.info.get("score")
        if score is None:
            return []
        return [row(str(score[0]), 114), row("VS", 170, width=20),
                row(str(score[1]), 272)]


def test_last_question_keeps_latest_score_frame_from_early_samples(monkeypatch):
    context, images, feedback = _last_question_transition(monkeypatch)
    assert feedback.status == "correct"
    assert context.final_frame is images["latest"], "反馈确认后仍要保留后续比分采样"


@pytest.mark.parametrize("fade_at", [0.0, 0.75])
def test_final_score_reads_fading_frame_even_when_pixel_gates_reject_it(monkeypatch, fade_at):
    from weread_quiz.pk import runner as pk

    context, _images, _feedback = _last_question_transition(monkeypatch, fade_at=fade_at)
    assert pk._read_final_score(context, FrameScoreOcr()) == (140, 85), (
        "渐隐帧里有更新后的对手分数，不能丢失或改用答前比分")


def test_final_score_falls_back_when_transition_ocr_has_no_score(monkeypatch):
    from weread_quiz.pk import runner as pk

    context, images, _feedback = _last_question_transition(monkeypatch)
    images["fading"].info["score"] = None
    assert pk._read_final_score(context, FrameScoreOcr()) == (140, 55)


def test_final_score_stays_unknown_without_readable_post_click_frames(monkeypatch):
    from weread_quiz.pk import runner as pk

    context, images, _feedback = _last_question_transition(monkeypatch, fade_at=0.0)
    images["fading"].info["score"] = None
    assert pk._read_final_score(context, FrameScoreOcr()) is None


def test_transition_during_early_sampling_is_not_overwritten_by_settlement(monkeypatch):
    from weread_quiz.pk import runner as pk

    context, images, _feedback = _last_question_transition(monkeypatch, settle_at=0.9)
    assert context.final_transition_frame is images["fading"]
    assert pk._read_final_score(context, FrameScoreOcr()) == (140, 85)


def test_recovered_card_discards_an_earlier_transition_frame(monkeypatch):
    from weread_quiz.pk import runner as pk
    from test_pk_wait_next_question import _install, Clock

    clock = Clock()
    monkeypatch.setattr(pk.time, "monotonic", clock)
    monkeypatch.setattr(pk.time, "sleep", clock.sleep)
    frames = _install(monkeypatch, ["blank", "answered", "blank", "settlement_buttons"])
    context = pk.PKCaptureContext()
    current, _ = pk.wait_for_next_question(None, frame(), delay=0, capture_context=context)
    assert current is None
    assert context.final_frame is frames[1]
    assert context.final_transition_frame is frames[2], "只能保留最后一次转场的起点"


def test_capture_reset_clears_transition_score_evidence():
    from weread_quiz.pk import runner as pk

    context = pk.PKCaptureContext()
    context.final_frame = object()
    context.final_transition_frame = object()
    context.reset()
    assert context.final_frame is None
    assert context.final_transition_frame is None
