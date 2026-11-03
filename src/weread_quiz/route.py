"""PK 与两条每日一答路线的独立推进白名单。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    """一条答题路线。"""

    name: str
    mode: str
    advance_targets: tuple[str, ...]


ROUTE_PK = Route("pk", "pk", ("问答PK", "自动匹配"))

ROUTE_DAILY_DAYS = Route(
    "daily-days", "daily", ("每日一答", "瓜分100万天体验卡", "开始游戏"),
)

ROUTE_DAILY_COINS = Route(
    "daily-coins", "daily", ("每日一答", "瓜分100万书币", "开始游戏"),
)

ALL_ROUTES: tuple[Route, ...] = (ROUTE_PK, ROUTE_DAILY_DAYS, ROUTE_DAILY_COINS)

# 名字 → 路线的查表。daily 的两条路线靠这个区分。
_BY_NAME = {route.name: route for route in ALL_ROUTES}


def get_route(name: str) -> Route:
    """按名字取路线。名字不认识就报错 —— 不猜。"""
    try:
        return _BY_NAME[name]
    except KeyError:
        known = "、".join(_BY_NAME)
        raise ValueError(f"未知路线 {name!r}（可用：{known}）") from None
