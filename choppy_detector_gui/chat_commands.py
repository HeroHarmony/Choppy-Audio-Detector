"""Twitch chat command matching and authorization."""

from __future__ import annotations

from dataclasses import dataclass

from .settings import ChatCommandSettings, normalize_twitch_username


@dataclass(frozen=True)
class ChatUser:
    username: str
    is_broadcaster: bool = False
    is_moderator: bool = False


@dataclass(frozen=True)
class ParsedCommand:
    action: str
    device_number: int | None = None
    status_full: bool = False


def normalize_command(value: str) -> str:
    return " ".join(value.strip().lower().split())


def is_authorized(user: ChatUser, settings: ChatCommandSettings) -> bool:
    username = normalize_twitch_username(user.username)
    allowed = {normalize_twitch_username(name) for name in settings.allowed_chat_users if name.strip()}
    if username in allowed:
        return True
    if settings.allow_broadcaster and user.is_broadcaster:
        return True
    if settings.allow_moderators and user.is_moderator:
        return True
    return False


def parse_chat_command(message: str, settings: ChatCommandSettings) -> ParsedCommand | None:
    text = normalize_command(message)
    if not text:
        return None

    exact_matches = {
        normalize_command(settings.start_command): "start",
        normalize_command(settings.stop_command): "stop",
        normalize_command(settings.restart_command): "restart",
        normalize_command(settings.status_command): "status",
        normalize_command(settings.list_devices_command): "list_devices",
        normalize_command(settings.fix_command): "fix",
        normalize_command(settings.rebuild_command): "rebuild_baseline",
        normalize_command(settings.clip_command): "capture_clip",
    }
    action = exact_matches.get(text)
    if action:
        return ParsedCommand(action=action)

    status_prefix = normalize_command(settings.status_command)
    if text == f"{status_prefix} full":
        return ParsedCommand(action="status", status_full=True)

    prefix = normalize_command(settings.switch_device_command_prefix)
    if text.startswith(prefix + " "):
        raw_number = text[len(prefix) :].strip()
        try:
            return ParsedCommand(action="switch_device", device_number=int(raw_number))
        except ValueError:
            return ParsedCommand(action="switch_device")

    return None
