"""Workflow service for OBS connection and refresh orchestration."""

from __future__ import annotations

import time

from choppy_detector_gui.obs_connection_controller import build_connection_config, test_connection_once
from choppy_detector_gui.websocket_settings_controller import collect_obs_from_controls


def start_obs_auto_connect(window, *, max_attempts: int) -> None:
    window._obs_auto_connect_attempt = 0
    attempt_obs_auto_connect(window, max_attempts=max_attempts)


def attempt_obs_auto_connect(window, *, max_attempts: int) -> None:
    if window.obs_service.is_connected:
        return
    if window._obs_auto_connect_attempt >= max_attempts:
        window.append_console(f"OBS auto-connect failed after {max_attempts} attempts.")
        window.update_obs_controls_enabled()
        return
    window._obs_auto_connect_attempt += 1
    attempt = window._obs_auto_connect_attempt
    window.append_console(f"OBS auto-connect attempt {attempt}/{max_attempts}...")
    cfg = build_connection_config(window.settings)
    window.set_obs_status("Connecting", "#4aa3ff")
    window.set_obs_busy(True)
    window._run_obs_task(
        "connect_auto",
        lambda: window.obs_service.connect(cfg),
        {"attempt": attempt, "max_attempts": max_attempts},
    )


def connect_obs(window) -> None:
    window.cancel_obs_auto_connect_retry()
    if not window.obs_enabled.isChecked():
        window._show_obs_disabled_message()
        return
    collect_obs_from_controls(window)
    cfg = build_connection_config(window.settings)
    window.set_obs_status("Connecting", "#4aa3ff")
    window.set_obs_busy(True)
    window._run_obs_task("connect", lambda: window.obs_service.connect(cfg))


def test_obs_connection(window) -> None:
    if not window.obs_enabled.isChecked():
        window._show_obs_disabled_message()
        return
    collect_obs_from_controls(window)
    window.set_obs_status("Testing", "#4aa3ff")
    window.set_obs_busy(True)
    window._run_obs_task("test", lambda: test_connection_once(window.settings))


def disconnect_obs(window) -> None:
    window.cancel_obs_auto_connect_retry()
    window.obs_service.disconnect()
    window.set_obs_status("Disconnected", "#ff9c4a")
    window.append_console("Disconnected from OBS WebSocket.")
    window.update_obs_controls_enabled()


def refresh_obs_source_now(window) -> None:
    if not window.obs_enabled.isChecked():
        window._show_obs_disabled_message()
        return
    source = window.obs_target_source.currentText().strip()
    if not source:
        window._show_obs_source_required_message()
        return
    queue_obs_refresh_request(window, source=source, action="refresh")


def queue_obs_refresh_request(window, *, source: str, action: str = "refresh") -> None:
    window.set_obs_status("Refreshing", "#4aa3ff")
    window.set_obs_busy(True)
    window._run_obs_task(
        action,
        lambda: window.obs_service.refresh_source_in_scene(
            source_name=source,
            scene_name=""
            if window.obs_target_scene.currentText().strip() == "All Scenes"
            else window.obs_target_scene.currentText().strip(),
            off_on_delay_ms=window.obs_refresh_off_on_delay_ms.value(),
        ),
    )


def maybe_trigger_obs_auto_refresh(window, glitch_data: dict[str, object]) -> None:
    obs_settings = window.settings.obs_websocket
    if not obs_settings.enabled or not obs_settings.auto_refresh_enabled:
        return
    if not window.obs_service.is_connected:
        window.append_console("OBS auto-refresh skipped: OBS is not connected.")
        return

    source = window.obs_target_source.currentText().strip() or obs_settings.target_source.strip()
    scene_choice = window.obs_target_scene.currentText().strip()
    scene = "" if scene_choice == "All Scenes" else (scene_choice or obs_settings.target_scene.strip())
    if not source:
        window.append_console("OBS auto-refresh skipped: no target source selected.")
        return

    event_severity = window.derive_glitch_severity(glitch_data)
    required_severity = (obs_settings.auto_refresh_min_severity or "severe").strip().lower()
    if not window.severity_meets_threshold(event_severity, required_severity):
        window.append_console(
            f"OBS auto-refresh skipped: event severity {event_severity} below threshold {required_severity}."
        )
        return

    now = time.monotonic()
    cooldown_sec = max(0, int(obs_settings.auto_refresh_cooldown_sec))
    elapsed = now - window._obs_last_auto_refresh_at
    if window._obs_last_auto_refresh_at > 0 and elapsed < cooldown_sec:
        remaining = max(0.0, cooldown_sec - elapsed)
        window.append_console(f"OBS auto-refresh skipped: cooldown active ({remaining:.1f}s remaining).")
        return

    window._obs_last_auto_refresh_at = now
    window.append_console(
        f"OBS auto-refresh triggered: severity {event_severity} met threshold {required_severity}."
    )
    window.set_obs_status("Auto Refreshing", "#4aa3ff")
    window.set_obs_busy(True)
    window._run_obs_task(
        "auto_refresh",
        lambda: window.obs_service.refresh_source_in_scene(
            source_name=source,
            scene_name=scene,
            off_on_delay_ms=obs_settings.refresh_off_on_delay_ms,
        ),
    )
