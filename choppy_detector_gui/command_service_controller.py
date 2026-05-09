"""Orchestration helpers for Twitch command service lifecycle."""

from __future__ import annotations

from collections.abc import Callable

from choppy_detector_gui.chat_command_controller import should_restart_listener_for_credentials
from choppy_detector_gui.settings import AppSettings
from choppy_detector_gui.twitch_command_service import TwitchCommandService
from choppy_detector_gui.twitch_status_coordinator import TwitchStatusCoordinator


def sync_command_service(
    *,
    settings: AppSettings,
    command_service: TwitchCommandService,
    twitch_status: TwitchStatusCoordinator,
    set_badge: Callable[[str, str], None],
    append_console: Callable[[str], None],
) -> None:
    command_service.settings = settings
    if not settings.chat_commands.chat_commands_enabled:
        command_service.stop()
        label, color = twitch_status.mark_chat_disabled(
            alerts_enabled=bool(settings.twitch_enabled),
            chat_enabled=bool(settings.chat_commands.chat_commands_enabled),
        )
        set_badge(label, color)
        return

    if not command_service.running:
        command_service.start()
        return

    bot = getattr(command_service, "bot", None)
    if bot is None:
        # Defensive: running state should normally have an active bot.
        command_service.stop()
        command_service.start()
        return

    if should_restart_listener_for_credentials(bot, settings):
        append_console("Twitch chat settings changed; reconnecting chat command listener.")
        command_service.stop()
        command_service.start()
