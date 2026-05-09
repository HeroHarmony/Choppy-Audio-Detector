"""Helpers for chat-command service orchestration."""

from __future__ import annotations

from typing import Any

from .settings import AppSettings


def desired_chat_credentials(settings: AppSettings) -> tuple[str, str, str]:
    channel = f"#{str(settings.twitch_channel or '').strip().lstrip('#')}"
    username = str(settings.twitch_bot_username or "").strip()
    token = str(settings.twitch_oauth_token or "").strip()
    return channel, username, token


def should_restart_listener_for_credentials(bot: Any, settings: AppSettings) -> bool:
    desired_channel, desired_username, desired_token = desired_chat_credentials(settings)
    current_channel = str(getattr(bot, "channel", "") or "").strip()
    current_username = str(getattr(bot, "username", "") or "").strip()
    current_token = str(getattr(bot, "token", "") or "").strip()
    return any(
        (
            current_channel != desired_channel,
            current_username != desired_username,
            current_token != desired_token,
        )
    )
