"""OBS WebSocket settings mapping helpers for GUI controls."""

from __future__ import annotations

from dataclasses import dataclass

from .settings import ObsWebSocketSettings


@dataclass(frozen=True)
class ObsFormValues:
    enabled: bool
    host: str
    port: int
    password: str
    auto_refresh_enabled: bool
    auto_refresh_min_severity: str
    auto_refresh_cooldown_sec: int
    refresh_off_on_delay_ms: int
    baseline_rebuild_on_scene_exit_enabled: bool
    baseline_rebuild_scene: str
    baseline_rebuild_min_dwell_sec: int
    baseline_rebuild_delay_sec: int
    baseline_rebuild_cooldown_sec: int
    target_scene: str
    target_source: str


def normalize_scene(scene_value: str) -> str:
    stripped = str(scene_value or "").strip()
    return "" if stripped == "All Scenes" else stripped


def normalize_watched_scene(scene_value: str) -> str:
    stripped = str(scene_value or "").strip()
    return "" if stripped == "Any Scene" else stripped


def read_form_values(
    *,
    enabled: bool,
    host: str,
    port: int,
    password: str,
    auto_refresh_enabled: bool,
    auto_refresh_min_severity: str,
    auto_refresh_cooldown_sec: int,
    refresh_off_on_delay_ms: int,
    baseline_rebuild_on_scene_exit_enabled: bool,
    baseline_rebuild_scene_value: str,
    baseline_rebuild_min_dwell_sec: int,
    baseline_rebuild_delay_sec: int,
    baseline_rebuild_cooldown_sec: int,
    scene_value: str,
    source_value: str,
) -> ObsFormValues:
    return ObsFormValues(
        enabled=bool(enabled),
        host=str(host or "").strip(),
        port=int(port),
        password=str(password or ""),
        auto_refresh_enabled=bool(auto_refresh_enabled),
        auto_refresh_min_severity=str(auto_refresh_min_severity or "").strip().lower(),
        auto_refresh_cooldown_sec=int(auto_refresh_cooldown_sec),
        refresh_off_on_delay_ms=int(refresh_off_on_delay_ms),
        baseline_rebuild_on_scene_exit_enabled=bool(baseline_rebuild_on_scene_exit_enabled),
        baseline_rebuild_scene=normalize_watched_scene(baseline_rebuild_scene_value),
        baseline_rebuild_min_dwell_sec=int(baseline_rebuild_min_dwell_sec),
        baseline_rebuild_delay_sec=int(baseline_rebuild_delay_sec),
        baseline_rebuild_cooldown_sec=int(baseline_rebuild_cooldown_sec),
        target_scene=normalize_scene(scene_value),
        target_source=str(source_value or "").strip(),
    )


def apply_form_values_to_settings(values: ObsFormValues, settings: ObsWebSocketSettings) -> None:
    settings.enabled = values.enabled
    settings.host = values.host
    settings.port = values.port
    settings.password = values.password
    settings.auto_refresh_enabled = values.auto_refresh_enabled
    settings.auto_refresh_min_severity = values.auto_refresh_min_severity
    settings.auto_refresh_cooldown_sec = values.auto_refresh_cooldown_sec
    settings.refresh_off_on_delay_ms = values.refresh_off_on_delay_ms
    settings.baseline_rebuild_on_scene_exit_enabled = values.baseline_rebuild_on_scene_exit_enabled
    settings.baseline_rebuild_scene = values.baseline_rebuild_scene
    settings.baseline_rebuild_min_dwell_sec = values.baseline_rebuild_min_dwell_sec
    settings.baseline_rebuild_delay_sec = values.baseline_rebuild_delay_sec
    settings.baseline_rebuild_cooldown_sec = values.baseline_rebuild_cooldown_sec
    settings.target_scene = values.target_scene
    if values.target_source:
        settings.target_source = values.target_source


def is_form_dirty(saved: ObsWebSocketSettings, current: ObsFormValues) -> bool:
    return any(
        (
            current.enabled != saved.enabled,
            current.host != saved.host,
            current.port != saved.port,
            current.password != saved.password,
            current.auto_refresh_enabled != saved.auto_refresh_enabled,
            current.auto_refresh_min_severity != saved.auto_refresh_min_severity,
            current.auto_refresh_cooldown_sec != saved.auto_refresh_cooldown_sec,
            current.refresh_off_on_delay_ms != saved.refresh_off_on_delay_ms,
            current.baseline_rebuild_on_scene_exit_enabled != saved.baseline_rebuild_on_scene_exit_enabled,
            current.baseline_rebuild_scene != saved.baseline_rebuild_scene,
            current.baseline_rebuild_min_dwell_sec != saved.baseline_rebuild_min_dwell_sec,
            current.baseline_rebuild_delay_sec != saved.baseline_rebuild_delay_sec,
            current.baseline_rebuild_cooldown_sec != saved.baseline_rebuild_cooldown_sec,
            current.target_scene != saved.target_scene,
            current.target_source != saved.target_source,
        )
    )
