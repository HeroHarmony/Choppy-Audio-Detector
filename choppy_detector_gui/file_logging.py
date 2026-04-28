"""Persistent local log file support."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .settings import LogSettings, default_log_directory


class AppFileLogger:
    def __init__(self, settings: LogSettings | None = None):
        self.settings = settings or LogSettings()
        self.log_dir = Path(self.settings.log_directory) if self.settings.log_directory else default_log_directory()
        self.current_date = ""
        self.current_path: Path | None = None

    def log(self, level: str, event: str, message: str = "", **fields: object) -> None:
        if not self.settings.logs_enabled:
            return
        configured_dir = Path(self.settings.log_directory) if self.settings.log_directory else default_log_directory()
        if configured_dir != self.log_dir:
            self.log_dir = configured_dir
            self.current_date = ""
        now = datetime.now().astimezone()
        stamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + now.strftime(" %z")
        date_key = now.strftime("%Y-%m-%d")
        if self.current_date != date_key:
            self.current_date = date_key
            self.current_path = self.log_dir / f"choppy-audio-detector-{date_key}.log"

        field_text = " ".join(f'{key}="{_escape(value)}"' for key, value in fields.items() if value is not None)
        parts = [stamp, level.upper(), event]
        if message:
            parts.append(message)
        if field_text:
            parts.append(field_text)
        line = " ".join(parts).rstrip() + "\n"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        assert self.current_path is not None
        with self.current_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')
