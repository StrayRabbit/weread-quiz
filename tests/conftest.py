"""pytest 公共夹具。"""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch, request):
    """把 socket 连接打断 —— 任何真实网络访问都会炸。"""
    if request.node.get_closest_marker("allow_network"):
        real_connect = socket.create_connection

        def guarded(address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) else address
            if host in ("127.0.0.1", "::1", "localhost"):
                return real_connect(address, *args, **kwargs)
            raise AssertionError("allow_network 只放行 loopback")

        monkeypatch.setattr(socket, "create_connection", guarded)
        return

    def _blocked(*args, **kwargs):
        raise AssertionError("测试里不允许访问网络")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
