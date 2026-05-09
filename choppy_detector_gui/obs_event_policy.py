"""OBS event outcome policy for GUI rendering."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObsEventDialog:
    level: str
    title: str
    message: str


@dataclass(frozen=True)
class ObsEventDecision:
    status_label: str
    status_color: str
    console_messages: list[str] = field(default_factory=list)
    event_messages: list[str] = field(default_factory=list)
    dialog: ObsEventDialog | None = None
    refresh_sources: bool = False
    update_controls: bool = False
    cancel_auto_connect_retry: bool = False
    schedule_auto_connect_retry_ms: int | None = None


def decide_obs_event(
    *,
    action: str,
    ok: bool,
    message: str,
    attempt: int,
    max_attempts: int,
    retry_enabled: bool,
    retry_delay_ms: int,
) -> ObsEventDecision:
    if action == "connect":
        if ok:
            return ObsEventDecision(
                status_label="Connected",
                status_color="#3fcf5e",
                console_messages=[message],
                refresh_sources=True,
                cancel_auto_connect_retry=True,
            )
        return ObsEventDecision(
            status_label="Error",
            status_color="#ff6a6a",
            console_messages=[message],
            dialog=ObsEventDialog(level="warning", title="OBS Connection Failed", message=message),
        )

    if action == "connect_auto":
        if ok:
            return ObsEventDecision(
                status_label="Connected",
                status_color="#3fcf5e",
                console_messages=[
                    f"OBS auto-connect succeeded on attempt {attempt}/{max_attempts}.",
                    message,
                ],
                refresh_sources=True,
                cancel_auto_connect_retry=True,
            )

        console_messages = [f"OBS auto-connect attempt {attempt}/{max_attempts} failed: {message}"]
        if retry_enabled and attempt < max_attempts:
            retry_seconds = retry_delay_ms // 1000
            console_messages.append(f"Retrying OBS auto-connect in {retry_seconds} seconds.")
            return ObsEventDecision(
                status_label="Error",
                status_color="#ff6a6a",
                console_messages=console_messages,
                schedule_auto_connect_retry_ms=retry_delay_ms,
                update_controls=True,
            )

        if not retry_enabled and attempt == 1:
            console_messages.append("OBS auto-connect retry is disabled; no further attempts will be made.")
        else:
            console_messages.append(f"OBS auto-connect stopped after {attempt}/{max_attempts} attempts.")
        return ObsEventDecision(
            status_label="Error",
            status_color="#ff6a6a",
            console_messages=console_messages,
            cancel_auto_connect_retry=True,
            update_controls=True,
        )

    if action == "test":
        if ok:
            return ObsEventDecision(
                status_label="Test OK",
                status_color="#3fcf5e",
                console_messages=[message],
                dialog=ObsEventDialog(level="info", title="OBS Connection Test", message=message),
                update_controls=True,
            )
        return ObsEventDecision(
            status_label="Test Failed",
            status_color="#ff6a6a",
            console_messages=[message],
            dialog=ObsEventDialog(level="warning", title="OBS Connection Test Failed", message=message),
            update_controls=True,
        )

    if action == "refresh":
        if ok:
            return ObsEventDecision(
                status_label="Connected",
                status_color="#3fcf5e",
                console_messages=[message],
                event_messages=[message],
                update_controls=True,
            )
        return ObsEventDecision(
            status_label="Error",
            status_color="#ff6a6a",
            console_messages=[message],
            dialog=ObsEventDialog(level="warning", title="OBS Refresh Failed", message=message),
            update_controls=True,
        )

    if action == "chat_refresh":
        if ok:
            return ObsEventDecision(
                status_label="Connected",
                status_color="#3fcf5e",
                console_messages=[f"OBS chat refresh succeeded: {message}"],
                event_messages=[f"OBS chat refresh: {message}"],
                update_controls=True,
            )
        return ObsEventDecision(
            status_label="Error",
            status_color="#ff6a6a",
            console_messages=[f"OBS chat refresh failed: {message}"],
            event_messages=[f"OBS chat refresh failed: {message}"],
            update_controls=True,
        )

    if action == "auto_refresh":
        if ok:
            return ObsEventDecision(
                status_label="Connected",
                status_color="#3fcf5e",
                console_messages=[f"OBS auto-refresh succeeded: {message}"],
                event_messages=[f"OBS auto-refresh: {message}"],
                update_controls=True,
            )
        return ObsEventDecision(
            status_label="Error",
            status_color="#ff6a6a",
            console_messages=[f"OBS auto-refresh failed: {message}"],
            event_messages=[f"OBS auto-refresh failed: {message}"],
            update_controls=True,
        )

    return ObsEventDecision(status_label="Disconnected", status_color="#ff9c4a")
