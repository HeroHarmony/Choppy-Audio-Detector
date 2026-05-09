"""GUI tab builders."""

from .settings_tab import build_settings_tab
from .websocket_tab import build_websocket_tab

__all__ = ["build_settings_tab", "build_websocket_tab"]
