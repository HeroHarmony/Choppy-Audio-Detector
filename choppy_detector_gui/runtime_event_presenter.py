"""Presenter for applying routed runtime events to GUI widgets."""

from __future__ import annotations

from datetime import datetime


class RuntimeEventPresenter:
    def __init__(self, window) -> None:
        self.window = window

    def apply_route(self, route, data: dict[str, object]) -> None:
        if route.mark_audio_level_received:
            self.window._last_audio_level_seen_at = datetime.now()
        if route.clear_audio_watchdog_warned:
            self.window._audio_watchdog_warned = False
        for line in route.append_console:
            self.window.append_console(line)
        for line in route.append_event:
            self.window.append_event(line)

        if route.audio_level is not None and not route.audio_level.skip_display:
            smooth_peak_dbfs, smooth_rms_dbfs = self.window._smooth_display_levels(
                route.audio_level.peak_dbfs,
                route.audio_level.rms_dbfs,
            )
            self.window.peak_meter.set_level_dbfs(smooth_peak_dbfs, peak_source=True)
            self.window.rms_meter.set_level_dbfs(smooth_rms_dbfs, peak_source=False)
            self.window.level_text.setText(
                f"Peak {route.audio_level.peak_dbfs:.1f} dBFS | RMS {route.audio_level.rms_dbfs:.1f} dBFS"
            )

        if route.trigger_obs_auto_refresh:
            self.window.maybe_trigger_obs_auto_refresh(data)

        if route.monitoring_ui_update is not None:
            self._apply_monitoring_ui_update(route.monitoring_ui_update)
        if route.audio_badge_update is not None:
            self.window.set_audio_status_badge(
                route.audio_badge_update.label,
                route.audio_badge_update.color_hex,
            )

    def _apply_monitoring_ui_update(self, update) -> None:
        self.window._monitoring_ui_active = update.active
        if update.status_label == "running_device_summary":
            self.window.status_label.setText(f"Running - {self.window.runtime.device_summary()}")
        else:
            self.window.status_label.setText(update.status_label)
        if update.reset_audio_watchdog_warned:
            self.window._audio_watchdog_warned = False
        if update.clear_monitoring_started_at:
            self.window._monitoring_started_at = None
        if update.restart_meter_preview:
            self.window.restart_meter_preview()
        if update.update_meter_display_mode:
            self.window.update_meter_display_mode()
        if update.update_device_controls:
            self.window.update_device_controls()

    def append_formatted_event(self, event_type: str, data: dict[str, object]) -> None:
        message = self.window.format_event(event_type, data)
        self.window.append_event(message)
        self.window.append_console(message)
