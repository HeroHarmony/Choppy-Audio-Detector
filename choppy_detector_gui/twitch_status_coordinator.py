"""Aggregates Twitch alert/chat states into one badge state."""

from __future__ import annotations

from typing import Any

from . import runtime_events as events


class TwitchStatusCoordinator:
    def __init__(self):
        self._alert_state = "idle"
        self._chat_state = "idle"

    def sync_from_settings(
        self,
        *,
        alerts_enabled: bool,
        chat_enabled: bool,
        chat_connected: bool,
        chat_running: bool,
        alert_connected: bool,
    ) -> tuple[str, str]:
        self._chat_state = "disabled" if not chat_enabled else "idle"
        self._alert_state = "disabled" if not alerts_enabled else "idle"

        if chat_enabled:
            if chat_connected:
                self._chat_state = "connected"
            elif chat_running:
                self._chat_state = "connecting"

        if alerts_enabled and alert_connected:
            self._alert_state = "connected"

        return self.current_badge(alerts_enabled=alerts_enabled, chat_enabled=chat_enabled)

    def on_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        alerts_enabled: bool,
        chat_enabled: bool,
    ) -> tuple[bool, tuple[str, str]]:
        changed = False

        if event_type == events.TWITCH_CONNECTING:
            self._alert_state = "connecting"
            changed = True
        elif event_type == events.TWITCH_CONNECTED:
            self._alert_state = "connected"
            changed = True
        elif event_type in {events.TWITCH_CONNECTION_FAILED, events.TWITCH_CONNECTION_ERROR}:
            self._alert_state = self._state_from_error_text(str(payload.get("error", "")))
            changed = True
        elif event_type == events.TWITCH_SEND_CIRCUIT_OPEN:
            self._alert_state = "paused"
            changed = True
        elif event_type == events.TWITCH_SEND_RESUMED:
            self._alert_state = "connected"
            changed = True
        elif event_type in events.MONITORING_STOP_EVENTS:
            self._alert_state = "idle" if alerts_enabled else "disabled"
            changed = True

        if event_type == events.CHAT_COMMANDS_CONNECTING:
            self._chat_state = "connecting"
            changed = True
        elif event_type in events.CHAT_RECONNECT_EVENTS:
            self._chat_state = "reconnecting"
            changed = True
        elif event_type == events.CHAT_COMMANDS_CONNECTED:
            self._chat_state = "connected"
            changed = True
        elif event_type == events.CHAT_COMMANDS_CONNECTION_FAILED:
            self._chat_state = self._state_from_error_text(str(payload.get("error", "")))
            changed = True
        elif event_type == events.CHAT_COMMANDS_DISCONNECTED:
            self._chat_state = "disconnected" if chat_enabled else "disabled"
            changed = True

        return changed, self.current_badge(alerts_enabled=alerts_enabled, chat_enabled=chat_enabled)

    def mark_chat_disabled(self, *, alerts_enabled: bool, chat_enabled: bool) -> tuple[str, str]:
        self._chat_state = "disabled"
        return self.current_badge(alerts_enabled=alerts_enabled, chat_enabled=chat_enabled)

    def current_badge(self, *, alerts_enabled: bool, chat_enabled: bool) -> tuple[str, str]:
        if not alerts_enabled and not chat_enabled:
            return "Disabled", "#8f8f8f"

        states: list[str] = []
        if alerts_enabled:
            states.append(self._alert_state)
        if chat_enabled:
            states.append(self._chat_state)

        if any(state == "connected" for state in states):
            return "Connected", "#3fcf5e"
        if any(state == "paused" for state in states):
            return "Paused", "#ff9c4a"
        if any(state == "auth_failed" for state in states):
            return "Auth failed", "#ff6a6a"
        if any(state == "reconnecting" for state in states):
            return "Reconnecting", "#f0c04a"
        if any(state == "connecting" for state in states):
            return "Connecting", "#4aa3ff"
        if any(state == "disconnected" for state in states):
            return "Disconnected", "#ff9c4a"
        return "Idle", "#bdbdbd"

    @staticmethod
    def _state_from_error_text(error_text: str) -> str:
        lowered = str(error_text or "").lower()
        if "auth" in lowered or "login" in lowered or "token" in lowered:
            return "auth_failed"
        return "disconnected"
