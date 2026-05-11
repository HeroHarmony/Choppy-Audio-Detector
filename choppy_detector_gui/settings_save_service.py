"""Workflow service for saving settings and applying side effects."""

from __future__ import annotations

from choppy_detector_gui.advanced_controller import collect_advanced_from_controls
from choppy_detector_gui.settings import save_settings
from choppy_detector_gui.settings_controller import collect_settings_from_controls
from choppy_detector_gui.websocket_settings_controller import collect_obs_from_controls


def save_all_settings(window) -> None:
    collect_settings_from_controls(window)
    collect_obs_from_controls(window)
    collect_advanced_from_controls(window)
    window.file_logger.settings = window.settings.log_settings
    save_settings(window.settings)
    window.apply_theme()
    window.update_meter_refresh_timer()
    window.update_auto_restart_timer()
    window.update_command_service()
    window.runtime.set_clip_capture_enabled(window.settings.enable_clip_capture_buffer)
    window.restart_meter_preview()
    window.update_meter_display_mode()
    window.update_twitch_status_from_settings()
    window.prune_log_windows()
    window.append_console("Settings saved.")
