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
    "detections_for_alert": 6,
    "alert_cooldown_ms": 60000,
    "detection_window_seconds": 90,
    "confidence_threshold": 70,
    "clean_audio_reset_seconds": 60,
    "event_dedup_seconds": 0.9,
    "fast_alert_burst_detections": 3,
    "fast_alert_window_seconds": 15,
    "fast_alert_min_confidence": 75,
    "log_possible_glitches": True,
    "possible_log_min_confidence": 0.70,
    "possible_log_interval_seconds": 10.0,
    "max_alert_age_seconds": 15.0,
    "max_alert_send_window_seconds": 8.0,
}

DEFAULT_THRESHOLDS = {
    "silence_ratio": 0.60,
    "amplitude_jump": 2.5,
    "envelope_discontinuity": 2.0,
    "modulation_freq_min_hz": 15.0,
    "modulation_freq_max_hz": 36.0,
    "modulation_strength": 8.5,
    "modulation_depth": 0.42,
    "modulation_peak_concentration": 0.20,
    "gap_duration_ms": 100,
    "min_audio_level": 0.005,
    "max_normal_gaps": 2,
    "suspicious_gap_count": 4,
}


@dataclass
class ChatCommandSettings:
    chat_commands_enabled: bool = False
    start_command: str = "!choppy start"
    stop_command: str = "!choppy stop"
    restart_command: str = "!choppy restart"
    status_command: str = "!choppy status"
    list_devices_command: str = "!choppy devices"
    switch_device_command_prefix: str = "!choppy device"
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
            switch_device_command_prefix=str(
                data.get("switch_device_command_prefix") or cls.switch_device_command_prefix
            ),
            allowed_chat_users=[str(user).strip().lower() for user in users if str(user).strip()],
            allow_broadcaster=bool(data.get("allow_broadcaster", True)),
            allow_moderators=bool(data.get("allow_moderators", True)),
            send_command_responses=bool(data.get("send_command_responses", True)),
        )


@dataclass
class LogSettings:
    logs_enabled: bool = True
    log_directory: str = ""
    log_retention_days: int = 30

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LogSettings":
        if not isinstance(data, dict):
            return cls()
        retention = int(data.get("log_retention_days") or 30)
        return cls(
            logs_enabled=bool(data.get("logs_enabled", True)),
            log_directory=str(data.get("log_directory") or ""),
            log_retention_days=max(1, retention),
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
    advanced_alert_config: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ALERT_CONFIG))
    advanced_thresholds: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    detection_methods: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_APPROACHES))
    chat_commands: ChatCommandSettings = field(default_factory=ChatCommandSettings)
    log_settings: LogSettings = field(default_factory=LogSettings)
    alert_templates: AlertTemplates = field(default_factory=AlertTemplates)

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
        self.advanced_alert_config = _merge_numeric_dict(DEFAULT_ALERT_CONFIG, self.advanced_alert_config)
        self.advanced_thresholds = _merge_numeric_dict(DEFAULT_THRESHOLDS, self.advanced_thresholds)
        self.detection_methods = _merge_bool_dict(DEFAULT_APPROACHES, self.detection_methods)
        self.alert_cooldown_ms = int(self.advanced_alert_config.get("alert_cooldown_ms", self.alert_cooldown_ms))
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
            advanced_alert_config=dict(data.get("advanced_alert_config") or DEFAULT_ALERT_CONFIG),
            advanced_thresholds=dict(data.get("advanced_thresholds") or DEFAULT_THRESHOLDS),
            detection_methods=dict(data.get("detection_methods") or DEFAULT_APPROACHES),
            chat_commands=ChatCommandSettings.from_dict(data.get("chat_commands")),
            log_settings=LogSettings.from_dict(data.get("log_settings")),
            alert_templates=AlertTemplates.from_dict(data.get("alert_templates")),
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
    return Path.cwd() / "Log"


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
