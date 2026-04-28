"""GUI/runtime settings for Choppy Audio Detector."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .alert_templates import AlertTemplates


APP_NAME = "ChoppyAudioDetector"


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
    alert_cooldown_ms: int = 60000
    keep_preview_while_monitoring: bool = False
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
        self.alert_cooldown_ms = max(1000, int(self.alert_cooldown_ms or 60000))
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
            alert_cooldown_ms=int(data.get("alert_cooldown_ms") or 60000),
            keep_preview_while_monitoring=bool(data.get("keep_preview_while_monitoring", False)),
            chat_commands=ChatCommandSettings.from_dict(data.get("chat_commands")),
            log_settings=LogSettings.from_dict(data.get("log_settings")),
            alert_templates=AlertTemplates.from_dict(data.get("alert_templates")),
        ).normalize()


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
