"""API token 用量的整数累计、区间统计与非致命日志输出。"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class UsageSnapshot:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_known: int = 0
    completion_known: int = 0
    total_known: int = 0

    def since(self, start: UsageSnapshot) -> UsageSnapshot:
        return UsageSnapshot(*(a - b for a, b in zip(
            self.__dict__.values(), start.__dict__.values()
        )))


class UsageTracker:
    def __init__(self) -> None:
        self._usage = UsageSnapshot()

    def record(self, raw=None) -> None:
        raw = raw if isinstance(raw, dict) else {}
        values = [raw.get(key) for key in (
            "prompt_tokens", "completion_tokens", "total_tokens"
        )]
        values = [v if type(v) is int and v >= 0 else None for v in values]
        if raw.get("total_tokens") is None and all(v is not None for v in values[:2]):
            values[2] = values[0] + values[1]
        previous = self._usage
        self._usage = UsageSnapshot(
            previous.requests + 1,
            previous.prompt_tokens + (values[0] or 0),
            previous.completion_tokens + (values[1] or 0),
            previous.total_tokens + (values[2] or 0),
            previous.prompt_known + (values[0] is not None),
            previous.completion_known + (values[1] is not None),
            previous.total_known + (values[2] is not None),
        )

    def snapshot(self) -> UsageSnapshot:
        return self._usage

    def since(self, start: UsageSnapshot) -> UsageSnapshot:
        return self._usage.since(start)


def usage_snapshot(service) -> UsageSnapshot | None:
    tracker = getattr(service, "usage", None)
    return tracker.snapshot() if tracker is not None else None


def usage_since(service, start: UsageSnapshot | None = None) -> UsageSnapshot:
    current = usage_snapshot(service)
    if current is None:
        return UsageSnapshot(requests=1)
    return current.since(start) if start is not None else current


def format_tokens(value: int) -> str:
    if value < 1000:
        return str(value)
    divisor, suffix = (1000000, "M") if value >= 1000000 else (1000, "K")
    hundredths, remainder = divmod(value * 100, divisor)
    hundredths += remainder * 2 >= divisor
    whole, fraction = divmod(hundredths, 100)
    number = f"{whole}.{fraction:02d}".rstrip("0").rstrip(".")
    return number + suffix


def usage_message(label: str, usage: UsageSnapshot) -> str:
    def field(value, known):
        if usage.requests and not known:
            return "未知"
        text = format_tokens(value)
        return text + ("（统计不完整）" if known < usage.requests else "")

    return (f"{label} token：输入 {field(usage.prompt_tokens, usage.prompt_known)}，"
            f"输出 {field(usage.completion_tokens, usage.completion_known)}，"
            f"合计 {field(usage.total_tokens, usage.total_known)}")


def safe_log(recorder, message: str) -> None:
    if recorder is not None:
        try:
            recorder.log(message)
        except Exception:
            pass  # 日志失败不能触发请求重试或掩盖原异常。


def _usage_line(label: str, usage: UsageSnapshot) -> str:
    try:
        return usage_message(label, usage)
    except Exception:
        return f"{label} token：输入 未知，输出 未知，合计 未知"


def log_usage(label: str, usage: UsageSnapshot, recorder=None) -> None:
    """只落盘，不打终端。

    每题的 token 明细走这条 —— 终端只留本局合计，免得刷十几行。
    没有落盘对象时退回打终端（否则这条信息就没人看得到）。
    """
    message = _usage_line(label, usage)
    if recorder is None:
        try:
            print(message)
        except Exception:
            pass  # 控制台关闭也不能改变答题流程。
        return
    safe_log(recorder, message)


def emit_usage(label: str, usage: UsageSnapshot, recorder=None) -> None:
    message = _usage_line(label, usage)
    # **终端与落盘各写一次，但不该在同一行写两次**。
    #
    # 调试模式（recorder.echo）下 recorder.log 本身就会打到终端 ——
    # 那时再 print 一次，同一行会连着出现两遍。echo 时跳过 print。
    echoing = bool(getattr(recorder, "echo", False))
    if not echoing:
        try:
            print(message)
        except Exception:
            pass  # 控制台关闭也不能改变答题流程。
    safe_log(recorder, message)


class Stopwatch:
    """可重置的分段计时器；完整明细写日志，终端按 recorder 配置展示。"""

    def __init__(self, recorder=None) -> None:
        self._recorder = recorder
        self._start = time.perf_counter()
        self._last = self._start
        self._stages: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        """记录一个阶段结束。到 report 时会打印每个阶段的耗时。"""
        try:
            now = time.perf_counter()
            self._stages.append((name, now - self._last))
            self._last = now
        except Exception:
            pass

    def elapsed(self) -> float:
        """从创建到现在的总秒数。"""
        return time.perf_counter() - self._start

    def report(self, label: str = "") -> str:
        """把各阶段耗时拼成一行，落进 run.log。终端只输出总计。"""
        try:
            now = time.perf_counter()
            total = now - self._last
            self._last = now
            if total > 0.005:
                self._stages.append(("其它", total))
            detail = " ".join(f"{name} {value * 1000:.0f}ms" for name, value in self._stages)
            message = f"{label} 耗时 {sum(v for _, v in self._stages) * 1000:.0f}ms：{detail}"
            safe_log(self._recorder, message)
            if self._recorder is None:
                # 没有落盘时明细会丢，终端补一行，省得白测
                print(message)
            return message
        except Exception:
            return ""

    def reset(self, recorder=None) -> None:
        """清空分段，把计时基准挪到现在。下一题复用同一个计时器时用。"""
        self._recorder = recorder if recorder is not None else self._recorder
        now = time.perf_counter()
        self._start = self._last = now
        self._stages.clear()


class PhaseTotals:
    """按类别累计耗时与调用次数，供等待阶段归因；不改变业务流程。"""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._units: dict[str, str] = {}

    def add(self, name: str, seconds: float, count: int = 1,
            unit: str = "") -> None:
        """累计一次（或多次）操作的耗时。"""
        try:
            self._totals[name] = self._totals.get(name, 0.0) + seconds
            self._counts[name] = self._counts.get(name, 0) + count
            if unit:
                self._units.setdefault(name, unit)
        except Exception:
            pass

    def total(self) -> float:
        """已归因的总耗时（秒）。"""
        return sum(self._totals.values())

    def format(self, separator: str = "｜") -> str:
        """拼成一行：名称 次数单位 耗时。"""
        try:
            parts = []
            for name, seconds in self._totals.items():
                unit = self._units.get(name, "")
                count = self._counts.get(name, 0)
                parts.append(f"{name} {count}{unit} {seconds * 1000:.0f}ms")
            return separator.join(parts)
        except Exception:
            return ""
