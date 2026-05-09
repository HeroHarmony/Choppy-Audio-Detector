"""OBS WebSocket connection helpers used by GUI actions."""

from __future__ import annotations

from .obs_websocket_service import ObsConnectionConfig, ObsWebSocketService
from .settings import AppSettings


def build_connection_config(settings: AppSettings) -> ObsConnectionConfig:
    return ObsConnectionConfig(
        host=settings.obs_websocket.host,
        port=settings.obs_websocket.port,
        password=settings.obs_websocket.password,
    )


def test_connection_once(settings: AppSettings) -> tuple[bool, str]:
    cfg = build_connection_config(settings)
    temp = ObsWebSocketService()
    ok, message = temp.connect(cfg)
    if ok:
        temp.disconnect()
        return True, f"Connection test successful to {cfg.host}:{cfg.port} (test socket closed)."
    return False, message
