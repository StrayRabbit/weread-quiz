"""PK 比分读取：左右分数、VS 被并进同一行、读不出就是读不出。"""

from PIL import Image

from weread_quiz.pk_score import read_score


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
    from weread_quiz import pk

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
