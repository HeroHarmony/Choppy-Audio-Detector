"""Runtime event name constants used by GUI coordinators."""

from __future__ import annotations

TWITCH_CONNECTING = "twitch.connecting"
TWITCH_CONNECTED = "twitch.connected"
TWITCH_CONNECTION_FAILED = "twitch.connection_failed"
TWITCH_CONNECTION_ERROR = "twitch.connection_error"
TWITCH_SEND_CIRCUIT_OPEN = "twitch.send_circuit_open"
TWITCH_SEND_RESUMED = "twitch.send_resumed"

CHAT_COMMANDS_CONNECTING = "chat_commands.connecting"
CHAT_COMMANDS_RECONNECTING = "chat_commands.reconnecting"
CHAT_COMMANDS_RECONNECT_SCHEDULED = "chat_commands.reconnect_scheduled"
CHAT_COMMANDS_CONNECTED = "chat_commands.connected"
CHAT_COMMANDS_CONNECTION_FAILED = "chat_commands.connection_failed"
CHAT_COMMANDS_DISCONNECTED = "chat_commands.disconnected"

MONITORING_STOPPED = "monitoring.stopped"
MONITORING_STOPPED_BY_REQUEST = "monitoring.stopped_by_request"
MONITORING_THREAD_EXITED = "monitoring.thread_exited"

CHAT_RECONNECT_EVENTS = {CHAT_COMMANDS_RECONNECTING, CHAT_COMMANDS_RECONNECT_SCHEDULED}
MONITORING_STOP_EVENTS = {MONITORING_STOPPED, MONITORING_STOPPED_BY_REQUEST, MONITORING_THREAD_EXITED}
