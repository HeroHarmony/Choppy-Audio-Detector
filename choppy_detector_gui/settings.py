"""GUI/runtime settings for Choppy Audio Detector."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .alert_templates import AlertTemplates


APP_NAME = "ChoppyAudioDetector"

DEFAULT_APPROACHES = {
    "silence_gaps": True,
    "amplitude_jumps": False,
    "envelope_discontinuity": True,
    "amplitude_modulation": True,
    "temporal_consistency": False,
    "energy_variance": False,
    "zero_crossings": False,
    "spectral_rolloff": False,
    "spectral_centroid": False,
}

DEFAULT_ALERT_CONFIG = {
    "production_window_ms": 1000,
    "production_step_ms": 50,
    "detections_for_alert": 4,
    "alert_cooldown_ms": 60000,
    "detection_window_seconds": 180,
    "confidence_threshold": 66,
    "stream_start_warmup_ignore_seconds": 1.5,
    "baseline_clean_lock_seconds": 10.0,
    "baseline_min_rms_samples": 80,
    "baseline_rms_cv_max": 0.85,
    "clean_audio_reset_seconds": 180,
    "event_dedup_seconds": 0.6,
    "fast_alert_burst_detections": 2,
    "fast_alert_window_seconds": 20,
    "fast_alert_min_confidence": 68,
    "log_possible_glitches": True,
    "possible_log_min_confidence": 0.70,
    "possible_log_interval_seconds": 10.0,
    "enable_subtle_modulation_promotion": False,
    "enable_burst_episode_promotion": True,
    "max_alert_age_seconds": 15.0,
    "max_alert_send_window_seconds": 8.0,
    "twitch_send_failures_for_pause": 3,
    "twitch_send_pause_seconds": 60.0,
}

DEFAULT_THRESHOLDS = {
    "silence_ratio": 0.70,
    "amplitude_jump": 2.5,
    "envelope_discontinuity": 2.0,
    "modulation_freq_min_hz": 15.0,
    "modulation_freq_max_hz": 36.0,
    "modulation_strength": 6.3,
    "modulation_depth": 0.42,
    "modulation_peak_concentration": 0.20,
    "gap_duration_ms": 180,
    "min_audio_level": 0.005,
    "max_normal_gaps": 2,
    "suspicious_gap_count": 7,
    "silence_guardrail_cap": 0.76,
    "silence_extreme_ratio": 0.92,
    "silence_extreme_gap_ms": 800,
    "silence_extreme_gap_count_offset": 2,
    "silence_require_modulation_hit": True,
    "silence_persistence_require_modulation_hit": True,
    "burst_promotion_require_modulation_hit": True,
    "long_window_sparse_promotion_require_modulation_hit": True,
    "burst_promotion_uncorroborated_cap": 0.78,
    "long_window_sparse_uncorroborated_cap": 0.78,
    "burst_episode_window_seconds": 8.0,
    "burst_episode_hits_required": 5,
    "burst_episode_min_conf": 0.68,
    "burst_episode_max_conf": 0.75,
    "burst_episode_min_gap_ms": 500,
    "burst_episode_max_density_per_second": 2,
    "burst_episode_promotion_conf": 0.76,
    "burst_episode_max_span_seconds": 3.0,
    "burst_episode_guard_window_seconds": 20.0,
    "burst_episode_guard_max_candidates": 10,
    "compact_silence_cluster_min_conf": 0.68,
    "compact_silence_cluster_max_conf": 0.75,
    "compact_silence_cluster_min_silence_ratio": 0.70,
    "compact_silence_cluster_max_silence_ratio": 0.82,
    "compact_silence_cluster_min_gap_ms": 500,
    "compact_silence_cluster_max_gap_count": 1,
    "compact_silence_cluster_min_mod_strength": 3.8,
    "compact_silence_cluster_min_mod_depth": 0.90,
    "compact_silence_cluster_min_mod_concentration": 0.10,
    "compact_silence_cluster_mod_halo_seconds": 1.0,
    "compact_silence_cluster_recent_mod_window_seconds": 12.0,
    "compact_silence_cluster_recent_mod_min_hits": 2,
    "compact_silence_cluster_recent_mod_max_hits": 12,
    "compact_silence_cluster_recent_silence_window_seconds": 20.0,
    "compact_silence_cluster_recent_silence_max_hits": 20,
    "compact_silence_cluster_hits_required": 2,
    "compact_silence_cluster_max_span_seconds": 2.0,
    "compact_silence_cluster_guard_window_seconds": 20.0,
    "compact_silence_cluster_guard_max_candidates": 24,
    "compact_silence_cluster_promotion_conf": 0.76,
    "subtle_sparse_min_conf": 0.68,
    "subtle_sparse_max_conf": 0.75,
    "subtle_sparse_min_silence_ratio": 0.65,
    "subtle_sparse_max_silence_ratio": 0.82,
    "subtle_sparse_min_gap_ms": 500,
    "subtle_sparse_recent_mod_window_seconds": 1.5,
    "subtle_sparse_hits_required": 2,
    "subtle_sparse_max_span_seconds": 0.8,
    "subtle_sparse_guard_window_seconds": 45.0,
    "subtle_sparse_guard_max_candidates": 4,
    "subtle_sparse_promotion_conf": 0.76,
    "modulation_mid_silence_min_strength": 6.0,
    "modulation_mid_silence_min_depth": 0.44,
    "modulation_mid_silence_min_concentration": 0.20,
    "modulation_mid_silence_min_silence_ratio": 0.22,
    "modulation_mid_silence_max_silence_ratio": 0.55,
    "modulation_mid_silence_max_gap_count": 0,
    "modulation_mid_silence_promotion_conf": 0.76,
    "modulation_burst_min_strength": 8.5,
    "modulation_burst_min_depth": 0.44,
    "modulation_burst_min_concentration": 0.20,
    "modulation_burst_max_silence_ratio": 0.35,
    "modulation_burst_max_gap_count": 1,
    "modulation_burst_window_seconds": 2.8,
    "modulation_burst_hits_required": 3,
    "modulation_burst_spacing_seconds": 0.18,
    "modulation_burst_guard_window_seconds": 20.0,
    "modulation_burst_guard_max_candidates": 10,
    "modulation_burst_promotion_conf": 0.76,
}


def normalize_twitch_username(value: str) -> str:
    """Normalize Twitch usernames for case-insensitive checks."""
    return str(value or "").strip().lstrip("@").casefold()


@dataclass
class ChatCommandSettings:
    chat_commands_enabled: bool = False
    start_command: str = "!choppy start"
    stop_command: str = "!choppy stop"
    restart_command: str = "!choppy restart"
    status_command: str = "!choppy status"
    list_devices_command: str = "!choppy devices"
    fix_command: str = "!choppy fix"
    rebuild_command: str = "!choppy rebuild"
    clip_command: str = "!choppy clip"
    switch_device_command_prefix: str = "!choppy device"
    rebuild_response_template: str = "Rebuilding baseline profile."
    rebuild_completed_response_template: str = "Baseline profile created."
    allowed_chat_users: list[str] = field(default_factory=list)
    allow_broadcaster: bool = True
    allow_moderators: bool = True
    send_command_responses: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChatCommandSettings":
        if not isinstance(data, dict):
            return cls()
        users = data.get("allowed_chat_users")
        if not isinstance(users, list):
            users = []
        return cls(
            chat_commands_enabled=bool(data.get("chat_commands_enabled", False)),
            start_command=str(data.get("start_command") or cls.start_command),
            stop_command=str(data.get("stop_command") or cls.stop_command),
            restart_command=str(data.get("restart_command") or cls.restart_command),
            status_command=str(data.get("status_command") or cls.status_command),
            list_devices_command=str(data.get("list_devices_command") or cls.list_devices_command),
            fix_command=str(data.get("fix_command") or cls.fix_command),
            rebuild_command=str(data.get("rebuild_command") or cls.rebuild_command),
            clip_command=str(data.get("clip_command") or cls.clip_command),
            switch_device_command_prefix=str(
                data.get("switch_device_command_prefix") or cls.switch_device_command_prefix
            ),
            rebuild_response_template=str(
                data.get("rebuild_response_template") or cls.rebuild_response_template
            ),
            rebuild_completed_response_template=str(
                data.get("rebuild_completed_response_template") or cls.rebuild_completed_response_template
            ),
            allowed_chat_users=[
                normalized
                for normalized in (normalize_twitch_username(str(user)) for user in users)
                if normalized
            ],
            allow_broadcaster=bool(data.get("allow_broadcaster", True)),
            allow_moderators=bool(data.get("allow_moderators", True)),
            send_command_responses=bool(data.get("send_command_responses", True)),
        )


@dataclass
class LogSettings:
    logs_enabled: bool = True
    log_directory: str = ""
    log_retention_days: int = 30
    log_window_retention_minutes: int = 60

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LogSettings":
        if not isinstance(data, dict):
            return cls()
        retention = int(data.get("log_retention_days") or 30)
        window_retention = int(data.get("log_window_retention_minutes") or 60)
        return cls(
            logs_enabled=bool(data.get("logs_enabled", True)),
            log_directory=str(data.get("log_directory") or ""),
            log_retention_days=max(1, retention),
            log_window_retention_minutes=max(1, window_retention),
        )


@dataclass
class ObsWebSocketSettings:
    enabled: bool = False
    auto_connect_on_launch: bool = False
    auto_connect_retry_enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 4455
    password: str = ""
    target_source: str = ""
    target_sources: list[str] = field(default_factory=list)
    target_scene: str = ""
    auto_refresh_enabled: bool = False
    auto_refresh_min_severity: str = "severe"
    auto_refresh_cooldown_sec: int = 300
    refresh_off_on_delay_ms: int = 500
    baseline_rebuild_on_scene_exit_enabled: bool = False
    baseline_rebuild_scene: str = ""
    baseline_rebuild_min_dwell_sec: int = 5
    baseline_rebuild_delay_sec: int = 2
    baseline_rebuild_cooldown_sec: int = 120

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ObsWebSocketSettings":
        if not isinstance(data, dict):
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            auto_connect_on_launch=bool(data.get("auto_connect_on_launch", False)),
            auto_connect_retry_enabled=bool(data.get("auto_connect_retry_enabled", True)),
            host=str(data.get("host") or "127.0.0.1").strip() or "127.0.0.1",
            port=int(data.get("port") or 4455),
            password=str(data.get("password") or ""),
            target_source=str(data.get("target_source") or ""),
            target_sources=[
                str(value).strip()
                for value in (data.get("target_sources") or [])
                if str(value).strip()
            ],
            target_scene=str(data.get("target_scene") or ""),
            auto_refresh_enabled=bool(data.get("auto_refresh_enabled", False)),
            auto_refresh_min_severity=str(data.get("auto_refresh_min_severity") or "severe").strip().lower(),
            auto_refresh_cooldown_sec=int(data.get("auto_refresh_cooldown_sec") or 300),
            refresh_off_on_delay_ms=int(data.get("refresh_off_on_delay_ms") or 500),
            baseline_rebuild_on_scene_exit_enabled=bool(data.get("baseline_rebuild_on_scene_exit_enabled", False)),
            baseline_rebuild_scene=str(data.get("baseline_rebuild_scene") or "").strip(),
            baseline_rebuild_min_dwell_sec=int(data.get("baseline_rebuild_min_dwell_sec") or 5),
            baseline_rebuild_delay_sec=int(data.get("baseline_rebuild_delay_sec") or 2),
            baseline_rebuild_cooldown_sec=int(data.get("baseline_rebuild_cooldown_sec") or 120),
        )


@dataclass
class AppSettings:
    selected_device_id: int | None = None
    selected_channel_index: int = 0
    twitch_enabled: bool = True
    twitch_channel: str = ""
    twitch_bot_username: str = ""
    twitch_oauth_token: str = ""
    auto_restart_minutes: int = 60
    auto_start_monitoring: bool = False
    alert_cooldown_ms: int = 60000
    keep_preview_while_monitoring: bool = False
    smooth_preview_meter: bool = True
    preview_meter_fps: int = 20
    dark_mode_enabled: bool = True
    enable_clip_capture_buffer: bool = False
    advanced_alert_config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ALERT_CONFIG))
    advanced_thresholds: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    detection_methods: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_APPROACHES))
    chat_commands: ChatCommandSettings = field(default_factory=ChatCommandSettings)
    log_settings: LogSettings = field(default_factory=LogSettings)
    alert_templates: AlertTemplates = field(default_factory=AlertTemplates)
    obs_websocket: ObsWebSocketSettings = field(default_factory=ObsWebSocketSettings)

    def __post_init__(self) -> None:
        if not self.twitch_channel:
            self.twitch_channel = default_twitch_channel()
        if not self.twitch_bot_username:
            self.twitch_bot_username = default_twitch_bot_username()
        if not self.twitch_oauth_token:
            self.twitch_oauth_token = default_twitch_oauth_token()

    def normalize(self) -> "AppSettings":
        self.selected_channel_index = max(0, int(self.selected_channel_index or 0))
        self.auto_restart_minutes = min(1440, max(5, int(self.auto_restart_minutes or 60)))
        self.auto_start_monitoring = bool(self.auto_start_monitoring)
        self.alert_cooldown_ms = max(1000, int(self.alert_cooldown_ms or 60000))
        self.smooth_preview_meter = bool(self.smooth_preview_meter)
        self.preview_meter_fps = min(60, max(5, int(self.preview_meter_fps or 20)))
        self.dark_mode_enabled = bool(self.dark_mode_enabled)
        self.enable_clip_capture_buffer = bool(self.enable_clip_capture_buffer)
        self.advanced_alert_config = _merge_numeric_dict(DEFAULT_ALERT_CONFIG, self.advanced_alert_config)
        self.advanced_thresholds = _merge_numeric_dict(DEFAULT_THRESHOLDS, self.advanced_thresholds)
        self.detection_methods = _merge_bool_dict(DEFAULT_APPROACHES, self.detection_methods)
        self.alert_cooldown_ms = int(self.advanced_alert_config.get("alert_cooldown_ms", self.alert_cooldown_ms))
        self.obs_websocket.host = self.obs_websocket.host.strip() or "127.0.0.1"
        self.obs_websocket.port = min(65535, max(1, int(self.obs_websocket.port or 4455)))
        normalized_targets: list[str] = []
        for value in list(self.obs_websocket.target_sources or []):
            source = str(value or "").strip()
            if source and source not in normalized_targets:
                normalized_targets.append(source)
        fallback_target = str(self.obs_websocket.target_source or "").strip()
        if fallback_target and fallback_target not in normalized_targets:
            normalized_targets.insert(0, fallback_target)
        self.obs_websocket.target_sources = normalized_targets
        self.obs_websocket.target_source = normalized_targets[0] if normalized_targets else fallback_target
        self.obs_websocket.auto_connect_retry_enabled = bool(self.obs_websocket.auto_connect_retry_enabled)
        self.obs_websocket.auto_refresh_min_severity = (
            self.obs_websocket.auto_refresh_min_severity
            if self.obs_websocket.auto_refresh_min_severity in {"minor", "moderate", "severe"}
            else "severe"
        )
        self.obs_websocket.auto_refresh_cooldown_sec = max(0, int(self.obs_websocket.auto_refresh_cooldown_sec or 300))
        self.obs_websocket.refresh_off_on_delay_ms = min(
            10000,
            max(0, int(self.obs_websocket.refresh_off_on_delay_ms or 500)),
        )
        self.obs_websocket.baseline_rebuild_min_dwell_sec = max(
            1,
            int(self.obs_websocket.baseline_rebuild_min_dwell_sec or 5),
        )
        self.obs_websocket.baseline_rebuild_delay_sec = max(
            0,
            int(self.obs_websocket.baseline_rebuild_delay_sec or 2),
        )
        self.obs_websocket.baseline_rebuild_cooldown_sec = max(
            0,
            int(self.obs_websocket.baseline_rebuild_cooldown_sec or 120),
        )
        self.obs_websocket.baseline_rebuild_scene = str(self.obs_websocket.baseline_rebuild_scene or "").strip()
        self.chat_commands.rebuild_command = (
            str(self.chat_commands.rebuild_command or ChatCommandSettings.rebuild_command).strip()
            or ChatCommandSettings.rebuild_command
        )
        self.chat_commands.clip_command = (
            str(self.chat_commands.clip_command or ChatCommandSettings.clip_command).strip()
            or ChatCommandSettings.clip_command
        )
        self.chat_commands.rebuild_response_template = (
            str(self.chat_commands.rebuild_response_template or ChatCommandSettings.rebuild_response_template).strip()
            or ChatCommandSettings.rebuild_response_template
        )
        self.chat_commands.rebuild_completed_response_template = (
            str(
                self.chat_commands.rebuild_completed_response_template
                or ChatCommandSettings.rebuild_completed_response_template
            ).strip()
            or ChatCommandSettings.rebuild_completed_response_template
        )
        self.log_settings.log_retention_days = max(1, int(self.log_settings.log_retention_days or 30))
        self.log_settings.log_window_retention_minutes = max(
            1,
            int(self.log_settings.log_window_retention_minutes or 60),
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["alert_templates"] = self.alert_templates.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppSettings":
        if not isinstance(data, dict):
            return cls().normalize()
        selected = data.get("selected_device_id")
        try:
            selected_device_id = None if selected is None else int(selected)
        except (TypeError, ValueError):
            selected_device_id = None
        return cls(
            selected_device_id=selected_device_id,
            selected_channel_index=int(data.get("selected_channel_index") or 0),
            twitch_enabled=bool(data.get("twitch_enabled", True)),
            twitch_channel=str(data.get("twitch_channel") or default_twitch_channel()),
            twitch_bot_username=str(data.get("twitch_bot_username") or default_twitch_bot_username()),
            twitch_oauth_token=str(data.get("twitch_oauth_token") or default_twitch_oauth_token()),
            auto_restart_minutes=int(data.get("auto_restart_minutes") or 60),
            auto_start_monitoring=bool(data.get("auto_start_monitoring", False)),
            alert_cooldown_ms=int(data.get("alert_cooldown_ms") or 60000),
            keep_preview_while_monitoring=bool(data.get("keep_preview_while_monitoring", False)),
            smooth_preview_meter=bool(data.get("smooth_preview_meter", True)),
            preview_meter_fps=int(data.get("preview_meter_fps") or 20),
            dark_mode_enabled=bool(data.get("dark_mode_enabled", True)),
            enable_clip_capture_buffer=bool(data.get("enable_clip_capture_buffer", False)),
            advanced_alert_config=dict(data.get("advanced_alert_config") or DEFAULT_ALERT_CONFIG),
            advanced_thresholds=dict(data.get("advanced_thresholds") or DEFAULT_THRESHOLDS),
            detection_methods=dict(data.get("detection_methods") or DEFAULT_APPROACHES),
            chat_commands=ChatCommandSettings.from_dict(data.get("chat_commands")),
            log_settings=LogSettings.from_dict(data.get("log_settings")),
            alert_templates=AlertTemplates.from_dict(data.get("alert_templates")),
            obs_websocket=ObsWebSocketSettings.from_dict(data.get("obs_websocket")),
        ).normalize()


def _merge_numeric_dict(defaults: dict[str, Any], provided: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, default_value in defaults.items():
        value = provided.get(key, default_value) if isinstance(provided, dict) else default_value
        try:
            if isinstance(default_value, bool):
                merged[key] = bool(value)
            elif isinstance(default_value, int):
                merged[key] = int(value)
            else:
                merged[key] = float(value)
        except (TypeError, ValueError):
            merged[key] = default_value
    return merged


def _merge_bool_dict(defaults: dict[str, bool], provided: dict[str, Any]) -> dict[str, bool]:
    merged = dict(defaults)
    if not isinstance(provided, dict):
        return merged
    for key, default_value in defaults.items():
        merged[key] = bool(provided.get(key, default_value))
    return merged


def default_settings_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / APP_NAME / "settings.json"
    if sys_platform_is_macos():
        return Path.home() / "Library" / "Application Support" / APP_NAME / "settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / APP_NAME / "settings.json"


def default_log_directory() -> Path:
    return default_settings_path().parent / "Log"


def load_settings(path: Path | None = None) -> AppSettings:
    settings_path = path or default_settings_path()
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AppSettings()
    except Exception:
        return AppSettings()
    return AppSettings.from_dict(data)


def save_settings(settings: AppSettings, path: Path | None = None) -> Path:
    settings_path = path or default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings.normalize().to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return settings_path


def sys_platform_is_macos() -> bool:
    import sys

    return sys.platform == "darwin"


def default_twitch_channel() -> str:
    try:
        from config import TWITCH_CHANNEL
    except Exception:
        return ""
    return str(TWITCH_CHANNEL or "")


def default_twitch_bot_username() -> str:
    try:
        from config import TWITCH_BOT_USERNAME
    except Exception:
        return ""
    return str(TWITCH_BOT_USERNAME or "")


def default_twitch_oauth_token() -> str:
    try:
        from config import TWITCH_OAUTH_TOKEN
    except Exception:
        return ""
    return str(TWITCH_OAUTH_TOKEN or "")
