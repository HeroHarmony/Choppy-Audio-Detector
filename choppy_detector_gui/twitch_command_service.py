"""Background Twitch chat command listener."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .chat_commands import ChatUser, is_authorized, parse_chat_command
from .file_logging import AppFileLogger
from .settings import AppSettings

if TYPE_CHECKING:
    from .runtime import DetectorRuntime, RuntimeEventHandler
else:
    RuntimeEventHandler = Any


@dataclass(frozen=True)
class TwitchChatMessage:
    username: str
    message: str
    is_broadcaster: bool = False
    is_moderator: bool = False


class TwitchCommandService:
    def __init__(
        self,
        settings: AppSettings,
        runtime: "DetectorRuntime",
        *,
        event_handler: RuntimeEventHandler | None = None,
        file_logger: AppFileLogger | None = None,
    ):
        self.settings = settings
        self.runtime = runtime
        self.event_handler = event_handler
        self.file_logger = file_logger or AppFileLogger(settings.log_settings)
        self.bot = None
        self.thread: threading.Thread | None = None
        self.running = False

    def start(self) -> None:
        if self.running or not self.settings.chat_commands.chat_commands_enabled:
            return
        try:
            from twitch_chat import TwitchBot
        except Exception as exc:
            self.emit("chat_commands.unavailable", error=str(exc))
            self.file_logger.log("error", "chat_commands.unavailable", error=str(exc))
            return

        self.bot = TwitchBot(
            channel=self.settings.twitch_channel,
            username=self.settings.twitch_bot_username,
            token=self.settings.twitch_oauth_token,
        )
        self.running = True
        self.thread = threading.Thread(target=self._run, name="twitch-command-listener", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.bot:
            self.bot.disconnect()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None
        self.bot = None

    def emit(self, event_type: str, **payload: Any) -> None:
        if self.event_handler:
            self.event_handler(event_type, payload)

    def _run(self) -> None:
        assert self.bot is not None
        self.emit(
            "chat_commands.connecting",
            channel=getattr(self.bot, "channel", ""),
            username=getattr(self.bot, "username", ""),
        )
        self.file_logger.log(
            "info",
            "chat_commands.connecting",
            channel=getattr(self.bot, "channel", ""),
            username=getattr(self.bot, "username", ""),
        )
        if not self.bot.connect():
            error = getattr(self.bot, "last_error", "") or "Unknown Twitch connection failure"
            response = getattr(self.bot, "last_response", "")
            self.emit("chat_commands.connection_failed", error=error, response=response)
            self.file_logger.log(
                "error",
                "chat_commands.connection_failed",
                error=error,
                response=response,
            )
            self.running = False
            return

        self.emit("chat_commands.connected")
        self.file_logger.log("info", "chat_commands.connected")
        try:
            self.bot.listen(callback=self._handle_raw_message, should_continue=lambda: self.running)
        finally:
            self.running = False
            self.emit("chat_commands.disconnected")
            self.file_logger.log("info", "chat_commands.disconnected")

    def _handle_raw_message(self, raw_message: str) -> None:
        for message in parse_twitch_messages(raw_message):
            command = parse_chat_command(message.message, self.settings.chat_commands)
            if command is None:
                continue

            user = ChatUser(
                username=message.username,
                is_broadcaster=message.is_broadcaster,
                is_moderator=message.is_moderator,
            )
            if not is_authorized(user, self.settings.chat_commands):
                self.emit("chat_commands.rejected", user=message.username, reason="unauthorized")
                self.file_logger.log(
                    "warn",
                    "chat_commands.rejected",
                    user=message.username,
                    reason="unauthorized",
                )
                return

            self.emit("chat_commands.accepted", user=message.username, action=command.action)
            self.file_logger.log(
                "info",
                "chat_commands.accepted",
                user=message.username,
                action=command.action,
            )
            response = self.execute_command(command.action, command.device_number, message.username)
            if response and self.settings.chat_commands.send_command_responses:
                self.send_response(response)

    def execute_command(self, action: str, device_number: int | None, username: str) -> str:
        if action == "start":
            self.runtime.start(source="twitch")
            return "Monitoring started."
        if action == "stop":
            self.runtime.stop(source="twitch")
            return "Monitoring stopped."
        if action == "restart":
            self.runtime.restart(source="twitch")
            return "Monitoring restarted."
        if action == "status":
            state = "running" if self.runtime.is_running else "stopped"
            return f"Monitoring is {state}. Current device: {self.runtime.device_summary()}."
        if action == "list_devices":
            labels = [
                f"{device.selection_index}: {device.name}"
                + ("" if device.is_monitorable else " (output-only)")
                for device in self.runtime.list_devices()
            ]
            return "Available devices: " + "; ".join(labels)
        if action == "switch_device":
            if device_number is None:
                return "Could not switch device: invalid device number."
            ok, message = self.runtime.switch_device(device_number, source="twitch", user=username)
            if ok:
                return f"Switching monitor device to {message.replace('Selected device ', '')}."
            return f"Could not switch device: {message}."
        return ""

    def send_response(self, message: str) -> None:
        if not self.bot:
            return
        try:
            self.bot.send_message(message[:500])
        except Exception as exc:
            self.emit("chat_commands.response_failed", error=str(exc))
            self.file_logger.log("error", "chat_commands.response_failed", error=str(exc))


def parse_twitch_messages(raw_message: str) -> list[TwitchChatMessage]:
    messages: list[TwitchChatMessage] = []
    for line in raw_message.splitlines():
        parsed = parse_twitch_privmsg(line.strip())
        if parsed:
            messages.append(parsed)
    return messages


def parse_twitch_privmsg(line: str) -> TwitchChatMessage | None:
    if " PRIVMSG " not in line:
        return None

    tags: dict[str, str] = {}
    if line.startswith("@"):
        tag_text, _, line = line.partition(" ")
        for part in tag_text[1:].split(";"):
            key, _, value = part.partition("=")
            tags[key] = value

    match = re.match(r":([^!]+)![^ ]+ PRIVMSG #[^ ]+ :(.+)$", line)
    if not match:
        return None

    username = match.group(1)
    message = match.group(2)
    badges = tags.get("badges", "")
    return TwitchChatMessage(
        username=username,
        message=message,
        is_broadcaster="broadcaster/" in badges,
        is_moderator="moderator/" in badges,
    )
