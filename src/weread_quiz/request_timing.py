"""单次视觉请求的非致命时延诊断；只保存白名单阶段和时间，不读取 trace info。"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable


_LOCAL_STAGES = ("编码", "客户端准备", "HTTP总计", "JSON解析")
_TRACE_LABELS = {
    "connection.connect_tcp": "TCP连接",
    "connection.start_tls": "TLS握手",
    **{f"{protocol}.{event}": label
       for protocol in ("http11", "http2")
       for event, label in (
           ("send_request_headers", "发送请求头"),
           ("send_request_body", "发送请求体"),
           ("receive_response_headers", "等待响应头（非首token/非纯推理）"),
           ("receive_response_body", "接收响应体"),
       )},
}


@dataclass
class _Phase:
    elapsed: float = 0.0
    starts: deque[float] = field(default_factory=deque)
    completed: int = 0
    failed: int = 0
    unmatched: int = 0


class RequestTiming:
    """每个 _call_once 独立使用；诊断失败绝不影响请求或触发重试。"""

    def __init__(self, sink: Callable[[str], None] | None, request_no: int = 1):
        self._sink = sink
        self._request_no = request_no
        self._phases: dict[str, _Phase] = {}
        self._finished = False
        self._start = self._now()

    def _now(self) -> float | None:
        try:
            return time.perf_counter() if self._sink is not None else None
        except Exception:
            return None

    def _event(self, name: str, state: str) -> None:
        try:
            now = self._now()
            if now is None or self._finished:
                return
            phase = self._phases.setdefault(name, _Phase())
            if state == "started":
                phase.starts.append(now)
            elif phase.starts:
                phase.elapsed += now - phase.starts.popleft()
                if state == "complete":
                    phase.completed += 1
                else:
                    phase.failed += 1
            else:
                phase.unmatched += 1
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str):
        self._event(name, "started")
        try:
            yield
        except BaseException:
            self._event(name, "failed")
            raise  # 只记录状态，保留业务异常（包括中断）。
        else:
            self._event(name, "complete")

    def trace(self, event_name: str, info) -> None:
        # info 可能含 headers、URL、payload、异常正文；刻意不访问。
        try:
            event, state = event_name.rsplit(".", 1)
            if event in _TRACE_LABELS and state in ("started", "complete", "failed"):
                self._event(event, state)
        except Exception:
            pass

    def _field(self, name: str, label: str, now: float | None) -> str:
        phase = self._phases.get(name)
        if phase is None:
            return f"{label} 未观测"
        elapsed = phase.elapsed
        if now is not None:
            elapsed += sum(now - start for start in phase.starts)
        observed = phase.completed or phase.failed or phase.starts
        duration = (f"{elapsed * 1000:.0f}ms"
                    if now is not None and observed else "未观测")
        return (f"{label} {duration}（完成{phase.completed}，失败{phase.failed}，"
                f"未完成{len(phase.starts)}，未配对{phase.unmatched}）")

    def _message(self, now: float | None, success: bool) -> str:
        total = (f"{(now - self._start) * 1000:.0f}ms"
                 if now is not None and self._start is not None else "未观测")
        local = "；".join(self._field(name, name, now) for name in _LOCAL_STAGES)
        trace = "；".join(self._field(name, label, now)
                          for name, label in _TRACE_LABELS.items()
                          if not name.startswith("http2.") or name in self._phases)
        return (f"请求 #{self._request_no} 时延（{'完成' if success else '失败'}）："
                f"{local}；attempt总计 {total}；"
                f"HTTP子阶段（不重复相加，缺失仅表示未观测）：{trace}")

    def finish(self, success: bool = True) -> None:
        try:
            if self._finished or self._sink is None:
                return
            now = self._now()
            self._finished = True
            self._sink(self._message(now, success))
        except Exception:
            pass
