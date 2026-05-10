"""Route runtime events into structured UI actions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeEventContext:
    obs_enabled: bool
    obs_connected: bool
    selected_source: str
    saved_source: str
    runtime_running: bool
    keep_preview_while_monitoring: bool


@dataclass(frozen=True)
class AudioLevelUpdate:
    peak_dbfs: float
    rms_dbfs: float
    skip_display: bool


@dataclass(frozen=True)
class ObsRefreshRequest:
    source: str
    action: str = "chat_refresh"


@dataclass(frozen=True)
class MonitoringUiUpdate:
    active: bool
    status_label: str
    reset_audio_watchdog_warned: bool = False
    clear_monitoring_started_at: bool = False
    restart_meter_preview: bool = False
    update_meter_display_mode: bool = True
    update_device_controls: bool = True


@dataclass(frozen=True)
class AudioBadgeUpdate:
    label: str
    color_hex: str


@dataclass(frozen=True)
class RuntimeEventRoute:
    consume_event: bool = False
    append_console: list[str] = field(default_factory=list)
    append_event: list[str] = field(default_factory=list)
    mark_audio_level_received: bool = False
    clear_audio_watchdog_warned: bool = False
    audio_level: AudioLevelUpdate | None = None
    monitoring_ui_update: MonitoringUiUpdate | None = None
    audio_badge_update: AudioBadgeUpdate | None = None
    obs_refresh_request: ObsRefreshRequest | None = None
    trigger_obs_auto_refresh: bool = False
    append_formatted_event: bool = True


def route_runtime_event(event_type: str, data: dict[str, object], ctx: RuntimeEventContext) -> RuntimeEventRoute:
    if event_type == "chat_commands.fix_requested":
        user = str(data.get("user", "")).strip() or "unknown"
        if not ctx.obs_enabled:
            return RuntimeEventRoute(
                consume_event=True,
                append_console=[f"Chat fix by {user} skipped: OBS WebSocket integration disabled."],
                append_formatted_event=False,
            )
        if not ctx.obs_connected:
            return RuntimeEventRoute(
                consume_event=True,
                append_console=[f"Chat fix by {user} skipped: OBS is not connected."],
                append_formatted_event=False,
            )
        source = ctx.selected_source or ctx.saved_source
        if not source:
            return RuntimeEventRoute(
                consume_event=True,
                append_console=[f"Chat fix by {user} skipped: no OBS source selected."],
                append_formatted_event=False,
            )
        return RuntimeEventRoute(
            consume_event=True,
            append_console=[f"Chat fix requested by {user}: refreshing source '{source}'."],
            obs_refresh_request=ObsRefreshRequest(source=source, action="chat_refresh"),
            append_formatted_event=False,
        )

    if event_type == "audio.level":
        peak_dbfs = float(data.get("peak_dbfs", data.get("dbfs", -120.0)))
        rms_dbfs = float(data.get("rms_dbfs", data.get("dbfs", -120.0)))
        return RuntimeEventRoute(
            consume_event=True,
            mark_audio_level_received=True,
            clear_audio_watchdog_warned=True,
            audio_level=AudioLevelUpdate(
                peak_dbfs=peak_dbfs,
                rms_dbfs=rms_dbfs,
                skip_display=ctx.runtime_running and not ctx.keep_preview_while_monitoring,
            ),
            append_formatted_event=False,
        )

    if event_type == "glitch.detected":
        return RuntimeEventRoute(trigger_obs_auto_refresh=True)

    if event_type == "runtime.console":
        line = str(data.get("line", "")).rstrip()
        if not line:
            return RuntimeEventRoute(consume_event=True, append_formatted_event=False)
        return RuntimeEventRoute(
            consume_event=True,
            append_console=[line],
            append_event=[line],
            append_formatted_event=False,
        )

    if event_type in {"monitoring.started", "audio.stream_opened"}:
        return RuntimeEventRoute(
            monitoring_ui_update=MonitoringUiUpdate(
                active=True,
                status_label="running_device_summary",
            ),
            audio_badge_update=AudioBadgeUpdate(label="Live", color_hex="#3fcf5e"),
        )

    if event_type in {"monitoring.stopped", "monitoring.stopped_by_request"}:
        return RuntimeEventRoute(
            monitoring_ui_update=MonitoringUiUpdate(
                active=False,
                status_label="Stopped",
                reset_audio_watchdog_warned=True,
                clear_monitoring_started_at=True,
                restart_meter_preview=True,
            ),
            audio_badge_update=AudioBadgeUpdate(label="Idle", color_hex="#8f8f8f"),
        )

    if event_type in {"monitoring.error", "audio.stream_error", "detector.error"}:
        return RuntimeEventRoute(
            monitoring_ui_update=MonitoringUiUpdate(
                active=False,
                status_label="Error",
            ),
            audio_badge_update=AudioBadgeUpdate(label="Error", color_hex="#ff5c5c"),
        )

    if event_type == "audio.stale":
        return RuntimeEventRoute(
            audio_badge_update=AudioBadgeUpdate(label="Stale", color_hex="#ff5c5c"),
            append_formatted_event=False,
        )

    if event_type == "audio.recovered":
        return RuntimeEventRoute(
            audio_badge_update=AudioBadgeUpdate(label="Live", color_hex="#3fcf5e"),
            append_formatted_event=False,
        )

    if event_type == "audio.stream_restarting":
        return RuntimeEventRoute(
            audio_badge_update=AudioBadgeUpdate(label="Restarting", color_hex="#ff9c4a"),
            append_formatted_event=False,
        )

    return RuntimeEventRoute()
