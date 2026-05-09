"""Workflow service for unsaved tab checks and save prompts."""

from __future__ import annotations

from collections.abc import Callable

from choppy_detector_gui.advanced_controller import advanced_dirty
from choppy_detector_gui.responses_controller import templates_dirty
from choppy_detector_gui.settings_controller import settings_dirty
from choppy_detector_gui.websocket_settings_controller import websocket_dirty


def handle_tab_changed(window, new_index: int, confirm_save: Callable[[str, str], bool]) -> None:
    previous_index = window._last_tab_index
    window._last_tab_index = new_index
    previous_widget = window.tabs.widget(previous_index)

    checks: tuple[tuple[object, Callable[[object], bool], str, str, Callable[[], None]], ...] = (
        (
            window.settings_tab,
            settings_dirty,
            "Unsaved Settings",
            "You have unsaved changes in Settings. Save now?",
            window.save_all_settings,
        ),
        (
            window.templates_tab,
            templates_dirty,
            "Unsaved Templates",
            "You have unsaved changes in Responses. Save now?",
            window.save_templates,
        ),
        (
            window.advanced_tab,
            advanced_dirty,
            "Unsaved Advanced Changes",
            "You have unsaved changes in Advanced. Save now?",
            window.save_advanced_settings,
        ),
        (
            getattr(window, "websocket_tab", None),
            websocket_dirty,
            "Unsaved WebSocket Changes",
            "You have unsaved changes in WebSocket. Save now?",
            window.save_obs_settings,
        ),
    )

    for target_widget, dirty_fn, title, message, save_action in checks:
        if previous_widget is target_widget and dirty_fn(window):
            if confirm_save(title, message):
                save_action()
            return
