"""命令行入口。三个子命令：pk / daily / clean。"""

from __future__ import annotations

import argparse
import sys
import traceback
import time

from pyautogui import FailSafeException

from . import abort
from .advance import advance_to_question
from .capture import RunRecorder  # _cmd_pk 用
from .config import AppConfig
from .errors import WqError
from .main import _force_utf8_output as force_utf8_output
from .pk.runner import PKCaptureContext, _answer_from_pre_read, run_pk, wait_for_question  # 被 _run_pk_with_vision 引用，模块级 import
from .pk.hybrid import HybridQuestionReader  # PK 的 OCR 快路径（门控不通过回退图片）
from .pk.ocr import RapidOcrReader  # 同上
from .route import ROUTE_PK, get_route
from .usage import emit_usage, safe_log, usage_since, usage_snapshot
from .model_service import build_model_service
from .vision import PROMPT_PK  # 同上


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weread-quiz",
        description="微信读书每日一答辅助工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pk = sub.add_parser("pk", help="自动答「问答 PK」对局")
    pk.add_argument(
        "--dry-run",
        action="store_true",
        help="只识别题目并打印答案，不点击",
    )
    pk.add_argument(
        "--trace",
        action="store_true",
        help="详细日志：把 run.log 的分段耗时、请求时延同步打到终端，"
             "并额外输出等待阶段的耗时归因（时间花在哪一类操作上）",
    )
    pk.add_argument(
        "--max-questions",
        type=int,
        default=30,
        help="最多处理几轮（含跳过；默认 30 —— 常规一局 10 题，平局加时赛会"
             "继续出题，界面出现结算页才停）",
    )
    pk.add_argument(
        "--keep-days",
        type=int,
        default=None,
        help="运行记录保留天数（默认 30）。启动时自动清理更早的截图。",
    )
    pk.add_argument(
        "--no-clean",
        action="store_true",
        help="跳过启动清理",
    )

    clean_cmd = sub.add_parser("clean", help="清理临时截图（保留标定和最近几天的运行记录）")
    clean_cmd.add_argument(
        "--keep-days",
        type=int,
        default=None,
        help="保留最近几天的运行记录（默认 30）",
    )
    clean_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="只报告要删什么，不实际删除",
    )

    daily = sub.add_parser(
        "daily",
        help="自动答「每日一答」一局（不使用道具/复活卡）",
    )
    daily.add_argument(
        "--dry-run",
        action="store_true",
        help="只识别当前这一题的答案并打印，不点击（识别完即停）",
    )
    daily.add_argument(
        "--route",
        choices=("daily-days", "daily-coins"),
        default="daily-days",
        help="每日一答的入口路线：瓜分天数还是瓜分书币（默认 daily-days）",
    )
    return parser


def _keep_days(args) -> int:
    from .clean import CLEAN_KEEP_DAYS

    value = getattr(args, "keep_days", None)
    return CLEAN_KEEP_DAYS if value is None else value


def _warmup_client(vision) -> float:
    """预先创建 HTTP 客户端和 SSL context，不发送网络请求。"""
    warmup = getattr(vision, "warmup_client", None)
    if warmup is None:
        return 0.0
    try:
        return float(warmup())
    except Exception:  # noqa: BLE001 - 预热失败不该影响答题
        return 0.0


def _protocol_note(service) -> str:
    """启动时说一句这个模型走的是哪种协议、能不能读图。"""
    protocol = getattr(service, "protocol", "chat")
    if protocol == "systemone":
        if getattr(service, "supports_image", False):
            return "决策模型协议判下标，整图交给视觉模型"
        return "决策模型协议，只能判下标；整图兜底不可用"
    return "OpenAI 兼容协议，读图 + 判下标"


def _vision_model_note(cfg, service) -> str:
    """配了第二个模型就说清它管哪一段 —— 两个模型各管各的。"""
    if not getattr(cfg, "vision_model_configured", False):
        return ""
    decision = getattr(service, "protocol", "chat") == "systemone"
    role = "复核 + 整图兜底" if decision else "整图兜底"
    return f"；视觉模型 {cfg.vision_name}（{role}）"


def _probe_text_service(service, deadline: float):
    """发一个最小的判下标请求 —— 确认端点通、模型名对、连接也建好了。

    门面自带 probe（两种协议都认）；没有门面的老对象走原来的 prompt 路径。
    """
    probe = getattr(service, "probe", None)
    if callable(probe):
        return probe(deadline)
    return service.answer_from_text(
        '计算 1+1。选项：0. 2；1. 3。只输出 {"answer_index":0}。',
        2, max_tokens=512, timeout=8.0, deadline=deadline,
    )


def _warmup_pk_model(vision, recorder) -> None:
    """匹配前用同一客户端真实请求一次；失败只记类型，不重试或阻止开局。

    走文本路径实际用的那个服务 —— 门面自己认协议（决策模型或视觉模型）。
    """
    started = time.monotonic()
    status = "失败（未返回预期答案）"
    try:
        answer = _probe_text_service(vision, started + 8.0)
        if answer == 0:
            status = "成功"
    except Exception as exc:  # noqa: BLE001 - 预热失败不该影响答题，不记录异常正文
        status = f"失败（{type(exc).__name__}）"
    elapsed = (time.monotonic() - started) * 1000
    message = f"模型预热：{status}，耗时 {elapsed:.0f}ms（真实文本请求，不重试）"
    safe_log(recorder, message)
    print(f"✓ {message}" if status == "成功" else f"⚠ {message}，继续匹配")


def _make_pk_question_judge(ocr=None):
    """PK 推进就绪判据：模板明确放行，或用独立题号行证据确认。"""
    from .pk.screen import looks_like_question_template

    def is_question_page(image) -> bool:
        template = looks_like_question_template(image)
        if template is True:
            return True
        if template is None:
            # 判不了 —— 保守当"不是"，让题号行那条判据自己说话。
            return False
        reader_ref = getattr(ocr, "read_progress", None)
        if reader_ref is None:
            return False
        try:
            return reader_ref(image) is True
        except Exception:  # noqa: BLE001 - 读不出当"不知道"
            return False

    return is_question_page


def _advance_to_ready(route, region, reader, recorder, ocr=None, refresh=None) -> bool:
    """把界面推进到答题页。返回 False 表示推进失败（已截图落盘）。"""
    from .daily.screen import is_question_screen

    options = {}
    if route.mode == "pk":
        is_question_page = _make_pk_question_judge(ocr=ocr)
        warmed = False

        def before_click(text):
            nonlocal warmed
            if text == "自动匹配" and not warmed:
                warmed = True
                _warmup_pk_model(reader, recorder)

        options["before_click"] = before_click
    else:
        is_question_page = is_question_screen

    ok = advance_to_question(
        region=region,
        ocr=ocr,
        recorder=recorder,
        targets=route.advance_targets,
        is_question_page=is_question_page,
        refresh=refresh,
        **options,
    )
    if not ok:
        print()
        print("⚠ 没能自动推进到答题界面，已停下。")
        print("  截图和日志在上面的运行目录里 —— 对照截图确认程序停在哪一页，")
        print("  手动处理后重跑。程序不会在不确定的界面上乱点。")
    return ok


def _cmd_clean(keep_days: int, dry_run: bool) -> int:
    from .clean import clean
    from .config import DATA_DIR

    clean(DATA_DIR, keep_days=keep_days, dry_run=dry_run)
    return 0


def _cmd_daily(dry_run: bool, route_name: str = "daily-days") -> int:
    """运行指定每日一答路线；提前预热，推进失败也释放模型服务，不自动删除运行记录。"""
    from .capture import RunRecorder
    from .config import AppConfig
    from .daily.runner import (
        DailyConfig,
        _IDENTIFY_BUDGET_SECONDS,
        _IMAGE_PATH_MIN_BUDGET,
        _TEXT_TIMEOUT,
        run_daily,
    )
    from .daily.hybrid import DailyHybridReader
    from .daily.ocr import DailyOcrReader
    from .daily.vision import DAILY_VISION_TIMEOUT, build_daily_vision_service
    from .dpi import get_scale_for_window, init_dpi_awareness
    from .usage import emit_usage, safe_log, usage_since, usage_snapshot
    from .window.finder import find_wechat_window
    from .window.foreground import (
        bring_to_front,
        clear_topmost,
        move_console_aside,
        refresh_window,
        restore_console,
    )

    init_dpi_awareness()
    cfg = AppConfig.from_env()
    cfg.validate_model()

    if not abort.start():
        print("⚠ Esc 中止不可用（键盘钩子没装上）—— 只能把鼠标推到屏幕左上角中止。")

    host = find_wechat_window()
    region = host.to_rect()
    scale = get_scale_for_window(host.hwnd)
    print(f"✓ 微信窗口：{host.title!r} {host.width}×{host.height} @({host.left},{host.top})"
          f"（系统缩放 {scale * 100:.0f}%）")

    recorder = RunRecorder(mode="daily")
    safe_log(recorder, f"daily 启动，dry_run={dry_run} 窗口 {region.as_dict()}")
    print(f"✓ 截图与日志：{recorder.run_dir}")
    print()
    if not dry_run:
        print("开始答题。鼠标会被程序控制 —— 想中止就按 Esc 键。")
        print("注意：不会动任何道具/复活卡；答错今日机会用完。")
    else:
        print("dry-run 模式：只识别**当前这一题**并打印答案，不点击。")
        print("（识别完就停 —— 不点击的话界面不会换题。）")
        print("注意：不会动任何道具/复活卡。")

    vision = build_daily_vision_service(
        cfg, timing_sink=lambda message: safe_log(recorder, message)
    )
    print(f"✓ 模型：{cfg.model_name}"
          f"（{_protocol_note(vision)}{_vision_model_note(cfg, vision)}，"
          f"识别预算={_IDENTIFY_BUDGET_SECONDS:g}s，"
          f"文本上限={_TEXT_TIMEOUT:g}s，图片下限={_IMAGE_PATH_MIN_BUDGET:g}s，"
          f"服务默认 timeout={DAILY_VISION_TIMEOUT}s, retries=0）")
    start = usage_snapshot(vision)
    # 小程序窗口置顶 + 激活：抓屏按它在屏幕上的位置来，窗口被控制台盖住
    # 就抓错画面。激活是必须的 —— GPU 合成的窗口失去焦点会停止渲染，
    # 只置顶不激活会留在一块白屏上。
    if not bring_to_front(host.hwnd):
        print("⚠ 小程序窗口没能激活 —— 请确认它没有被别的窗口挡住")
    console_moved, console_rect = move_console_aside(region)
    if console_moved:
        print("（控制台压着小程序窗口，已挪开 —— 跑完还原）")
    try:
        warm_cost = _warmup_client(vision)
        safe_log(recorder, f"连接预热：{warm_cost * 1000:.0f}ms（建 SSL context 与长连接）")
        # OCR 快路径：门控不通过就自动回退图片路径，所以接上它没有正确性
        # 风险，只有收益。阈值靠跑真实局校准（见 daily_ocr 模块开头）。
        #
        from .daily.ocr import DEFAULT_MIN_CONFIDENCE

        reader = DailyHybridReader(
            vision,
            ocr=DailyOcrReader(min_confidence=getattr(
                cfg, "daily_ocr_min_confidence", DEFAULT_MIN_CONFIDENCE)),
            verbose=not dry_run,
            log_sink=lambda message: safe_log(recorder, message),
        )
        # 预热要一张真实尺寸的图：onnx 的首次推理成本与输入尺寸相关，
        # 用合成小图预热不能真正覆盖。这里抓一张当前窗口 —— 此时还没进入
        # 答题界面也没关系，warmup 只求"跑过一次推理"，不解析结果。
        from .capture import grab_screen

        warm_sample = None
        if not dry_run:
            try:
                warm_sample = grab_screen(region)
            except Exception:  # noqa: BLE001 - 抓不到就只加载模型，不预热
                warm_sample = None
        ocr_ok = reader.warmup(warm_sample)
        terminal_ocr = reader._ocr if ocr_ok and not reader._ocr.warm_error else None
        daily_threshold = getattr(cfg, "daily_ocr_min_confidence", DEFAULT_MIN_CONFIDENCE)
        safe_log(recorder, f"OCR 快路径：{'启用' if ocr_ok else '不可用，全程走图片路径'}"
                           f"，最低置信度阈值 {daily_threshold:.2f}")
        print(f"✓ OCR 快路径：{'已预热' if ocr_ok else '不可用（回退图片路径）'}"
              f"，置信度阈值 {daily_threshold:.2f}")
        print()

        daily_cfg = DailyConfig(
            region=region,
            vision=reader,
            recorder=recorder,
            verbose=True,
            dry_run=dry_run,
            ocr=terminal_ocr,
            refresh=lambda: refresh_window(host.hwnd),
        )
        if not dry_run:
            # 正式模式才推进 —— dry-run 需要用户已经手动停在答题页。
            # 推进失败时直接 return 2：下面那个 finally 会关掉 vision。
            route = get_route(route_name)
            if not _advance_to_ready(
                route, region, reader, recorder,
                ocr=getattr(reader, "_ocr", None),
                refresh=lambda: refresh_window(host.hwnd),
            ):
                return 2
        report = run_daily(daily_cfg)
    finally:
        clear_topmost(host.hwnd)
        restore_console(console_rect)
        vision.close()
        emit_usage("本局", usage_since(vision, start), recorder)
    print()
    text = report.summary()
    print(text)
    safe_log(recorder, text)
    progress_summary = getattr(report, "progress_summary", None)
    if callable(progress_summary):
        text = progress_summary()
        if text:
            print(text)
            safe_log(recorder, text)
    if report.skipped:
        text = f"另有 {report.skipped} 题因点击前校验判定界面已变而跳过未点。"
        print(text)
        safe_log(recorder, text)
    gate_summary = getattr(reader, "gate_summary", None)
    if gate_summary is not None and not dry_run:
        text = gate_summary()
        print(text)
        safe_log(recorder, text)
    print(f"截图与日志：{recorder.run_dir}")
    return _daily_exit_code(report)


def _daily_exit_code(report) -> int:
    """每日一答的退出码映射。"""
    return {
        "success": 0,
        "fail": 1,
        "dry_run": 0,
        "questions_exhausted": 0,
        "final_pending": 0,
        "wrong": 1,
        "interrupted": 130,
        "error": 2,
    }.get(report.outcome, 2)


def _cmd_pk(dry_run: bool, max_questions: int, keep_days: int | None = None,
            no_clean: bool = False, trace: bool = False) -> int:
    """自动答一局「问答 PK」。"""
    from .dpi import get_scale_for_window, init_dpi_awareness
    from .window.finder import find_wechat_window
    from .window.foreground import (
        bring_to_front,
        clear_topmost,
        move_console_aside,
        refresh_window,
        restore_console,
    )

    cfg = AppConfig.from_env()
    cfg.validate_model()
    init_dpi_awareness()

    if not abort.start():
        print("⚠ Esc 中止不可用（键盘钩子没装上）—— 只能把鼠标推到屏幕左上角中止。")

    if not no_clean:
        from .clean import clean
        from .config import DATA_DIR

        clean(DATA_DIR, keep_days=keep_days)

    host = find_wechat_window()
    region = host.to_rect()
    scale = get_scale_for_window(host.hwnd)
    print(f"✓ 微信窗口：{host.title!r} {host.width}×{host.height} @({host.left},{host.top})"
          f"（系统缩放 {scale * 100:.0f}%）")

    recorder = RunRecorder(mode="pk", echo=trace)
    safe_log(recorder, f"启动，窗口 {region.as_dict()}")
    print(f"✓ 截图与日志：{recorder.run_dir}")
    if trace:
        print("✓ 详细日志：分段耗时与请求时延会同步打到终端")

    vision = build_model_service(cfg, timing_sink=lambda message: safe_log(recorder, message))
    warm_cost = _warmup_client(vision)
    safe_log(recorder, f"客户端初始化：{warm_cost * 1000:.0f}ms（仅本地 HTTP 客户端与 SSL context，未请求模型）")
    # OCR 快路径：门控不通过就自动回退图片路径，所以接上它没有正确性
    # 风险，只有收益。阈值靠跑真实局校准（见 pk_ocr 模块开头）。
    reader = HybridQuestionReader(
        vision,
        ocr=RapidOcrReader(min_confidence=cfg.ocr_min_confidence),
        verbose=not dry_run,
        log_sink=lambda message: safe_log(recorder, message),
        text_timeout=cfg.pk_text_timeout,
    )
    # 预热要一张真实尺寸的图：onnx 的首次推理成本与输入尺寸相关，
    # 用合成小图预热不能真正覆盖。这里抓一张当前窗口 —— 此时还没进入
    # 答题界面也没关系，warmup 只求"跑过一次推理"，不解析结果。
    from .capture import grab_screen

    warm_sample = None
    if not dry_run:
        try:
            warm_sample = grab_screen(region)
        except Exception:  # noqa: BLE001 - 抓不到就只加载模型，不预热
            warm_sample = None
    ocr_ok = reader.warmup(warm_sample)
    safe_log(recorder, f"OCR 快路径：{'启用' if ocr_ok else '不可用，全程走图片路径'}"
                       f"，最低置信度阈值 {cfg.ocr_min_confidence:.2f}")
    print(f"✓ 模型：{cfg.model_name}（{_protocol_note(vision)}"
          f"{_vision_model_note(cfg, vision)}，"
          f"文本上限 {cfg.pk_text_timeout:g}s，"
          f"失败重试 {cfg.vision_retries} 次，"
          f"{'蒙一个' if cfg.guess_on_failure else '停下'}）")
    print(f"✓ OCR 快路径：{'已预热' if ocr_ok else '不可用（回退图片路径）'}"
          f"，置信度阈值 {cfg.ocr_min_confidence:.2f}")
    print()

    # 小程序窗口置顶 + 激活：抓屏按它在屏幕上的位置来，窗口被控制台盖住
    # 就抓错画面。激活是必须的 —— GPU 合成的窗口失去焦点会停止渲染，
    # 只置顶不激活会留在一块白屏上。
    if not bring_to_front(host.hwnd):
        print("⚠ 小程序窗口没能激活 —— 请确认它没有被别的窗口挡住")
    console_moved, console_rect = move_console_aside(region)
    if console_moved:
        print("（控制台压着小程序窗口，已挪开 —— 跑完还原）")
    try:
        if not dry_run:
            # 正式模式才推进 —— dry-run 的意义是"看看当前这题能不能识别
            # 对"，它需要用户已经手动停在答题页。
            #
            # OCR 引擎来自 reader._ocr（混合读取器持有的 OCR 实例），
            # **不是** vision._ocr —— VisionService 上没有 _ocr 属性。
            if not _advance_to_ready(
                ROUTE_PK, region, reader, recorder,
                ocr=getattr(reader, "_ocr", None),
                refresh=lambda: refresh_window(host.hwnd),
            ):
                return 2
        return _run_pk_with_vision(
            cfg, region, reader, recorder, dry_run, max_questions,
            refresh=lambda: refresh_window(host.hwnd),
        )
    finally:
        clear_topmost(host.hwnd)
        restore_console(console_rect)
        vision.close()


def _run_pk_with_vision(cfg, region, vision, recorder, dry_run, max_questions,
                        refresh=None) -> int:
    """真正跑 PK 的逻辑。单独拆出来是为了让 vision.close() 能用 finally 兜住。"""
    start = usage_snapshot(vision)
    try:
        if dry_run:
            print("dry-run 模式：只识别不点击。请先在小程序里点『自动匹配』或『邀请好友』。")
            context = PKCaptureContext()
            image, question = wait_for_question(
                region, vision, timeout=180.0, capture_context=context,
                # 传 OCR 引擎：等待阶段模板判否时用它读题号行做第二判据
                # （见 pk.wait_for_question）。
                ocr=getattr(vision, "_ocr", None),
            )
            if image is None:
                print("超时：没等到答题界面")
                return 1
            from .pk.buttons import find_option_buttons

            if question is None:
                deadline = None
                if context.captured_at is not None:
                    anchor = context.captured_at
                    if context.confirmed_at is not None and context.confirmed_at > anchor:
                        anchor = context.confirmed_at
                    deadline = anchor + 6.0
                if context.ocr_read is not None:
                    question = _answer_from_pre_read(vision, context.ocr_read, deadline=deadline)
                else:
                    question = vision.read_question(image, prompt=PROMPT_PK, deadline=deadline)
            if question is None:
                print("已确认答题页，但模型未取得答案；未点击。")
                recorder.shot("dry_run", image=image)
                return 0

            emit_usage("第 1 题", getattr(vision, "last_question_usage", None) or usage_since(vision), recorder)
            boxes = find_option_buttons(image)
            recorder.shot("dry_run", image=image)
            print()
            print(f"题干：{question.stem}")
            for i, opt in enumerate(question.options):
                marker = "  ← 答案" if i == question.answer_index else ""
                y = boxes[i].center_y if i < len(boxes) else "?"
                print(f"  [{i}] y={y}  {opt}{marker}")
            print(f"\n截图：{recorder.run_dir}")
            return 0

        print("开始答题。鼠标会被程序控制 —— 想中止就按 Esc 键。")
        print("过期或题面变化会跳过，不点击；记录按处理轮次编号，未必对应实际题号。")
        print()
        try:
            report = run_pk(
                region,
                vision,
                max_questions=max_questions,
                guess_on_failure=cfg.guess_on_failure,
                recorder=recorder,
                next_question_delay=getattr(cfg, "pk_next_question_delay", None),
                ocr_fallback_seconds=getattr(cfg, "pk_ocr_fallback_seconds", None),
                # 复用同一个 OCR 引擎读题号 —— 已经在上面预热过了。
                ocr=getattr(vision, "_ocr", None),
                refresh=refresh,
            )
        except BaseException:
            # 异常只打在控制台的话，run.log 里什么都没有 —— 下次只能靠猜。
            # 把完整 traceback 落进日志（用户自己的机器，不涉及外发）。
            if recorder:
                safe_log(recorder, f"运行异常：\n{traceback.format_exc()}")
            raise

        if report.questions:
            skipped_first = report.skipped_first_questions()
            if skipped_first:
                print(f"  ⚠ 漏答：第 {'、'.join(str(n) for n in skipped_first)} 题"
                      f"（程序进入答题界面时它已经开跑了）")
            missing = report.missing_questions()
            if missing:
                print(f"  ⚠ 漏答：第 {'、'.join(str(n) for n in missing)} 题"
                      f"（题号跳号了）")
            duplicates = report.duplicate_questions()
            if duplicates:
                print(f"  ⚠ 第 {'、'.join(str(n) for n in duplicates)} 题被重复处理")

        print()
        print(report.summary())

        # 门控命中率的唯一用途：校准 OCR_MIN_CONFIDENCE。全 0 说明阈值
        # 定得太高、全回退也说明 OCR 没帮上忙，都需要回头调。
        gate_summary = getattr(vision, "gate_summary", None)
        if gate_summary is not None and not dry_run:
            print(gate_summary())

        guessed = report.guessed
        if guessed:
            nums = ", ".join(str(n) for n in guessed)
            print(f"其中第 {nums} 轮模型没给出答案，是蒙的。"
                  f"实际由模型判定 {report.solid} 题。")
        print(f"截图与日志：{recorder.run_dir}")
        return 0 if report.answered else 1
    finally:
        emit_usage("本局", usage_since(vision, start), recorder)


def _force_utf8_output() -> None:
    """把终端输出统一成 UTF-8（实现见 main 模块）。"""
    force_utf8_output()


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "pk":
            return _cmd_pk(
                dry_run=args.dry_run,
                max_questions=args.max_questions,
                keep_days=_keep_days(args),
                no_clean=args.no_clean,
                trace=args.trace,
            )
        if args.command == "clean":
            return _cmd_clean(keep_days=_keep_days(args), dry_run=args.dry_run)
        if args.command == "daily":
            return _cmd_daily(dry_run=args.dry_run, route_name=args.route)
    except WqError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except abort.Aborted:
        print("\n已中止（按了 Esc）。", file=sys.stderr)
        return 130
    except FailSafeException:
        print("\n已中止（鼠标推到了屏幕左上角）。", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        # 缺标定文件 / 缺 .env
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        # 配置校验失败（缺环境变量、枚举值非法）
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130

    parser.error("未知命令")
    return 1
