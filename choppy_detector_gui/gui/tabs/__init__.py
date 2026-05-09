"""GUI tab builders."""

from .advanced_tab import build_advanced_tab
from .console_tab import build_console_tab
from .main_tab import build_main_tab
from .responses_tab import build_responses_tab
from .settings_tab import build_settings_tab
from .support_tab import build_support_tab
from .websocket_tab import build_websocket_tab

__all__ = [
    "build_advanced_tab",
    "build_console_tab",
    "build_main_tab",
    "build_responses_tab",
    "build_settings_tab",
    "build_support_tab",
    "build_websocket_tab",
]
