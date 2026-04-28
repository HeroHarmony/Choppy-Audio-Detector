"""Threaded detector runtime used by the GUI."""

from __future__ import annotations

import threading
import contextlib
import io
from collections.abc import Callable
from typing import Any

from .audio_devices import AudioDeviceInfo, device_label_for_portaudio_index, list_audio_devices
from .file_logging import AppFileLogger
from .settings import AppSettings, save_settings


RuntimeEventHandler = Callable[[str, dict[str, Any]], None]


class DetectorRuntime:
    def __init__(
        self,
        settings: AppSettings,
        *,
        event_handler: RuntimeEventHandler | None = None,
        file_logger: AppFileLogger | None = None,
    ):
        self.settings = settings
        self.event_handler = event_handler
        self.file_logger = file_logger or AppFileLogger(settings.log_settings)
        self.detector = None
        self.thread: threading.Thread | None = None
        self.lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        with self.lock:
            return bool(self.detector and getattr(self.detector, "running", False))

    def emit(self, event_type: str, **payload: Any) -> None:
        if self.event_handler:
            self.event_handler(event_type, payload)

    def start(self, source: str = "gui") -> bool:
        with self.lock:
            if self.is_running:
                self.emit("monitoring.already_running", source=source)
                return True

            import live_analysis

            live_analysis.APPROACHES.update(self.settings.detection_methods)
            live_analysis.THRESHOLDS.update(self.settings.advanced_thresholds)
            live_analysis.ALERT_CONFIG.update(self.settings.advanced_alert_config)
            live_analysis.ALERT_CONFIG["alert_cooldown_ms"] = self.settings.alert_cooldown_ms

            device = self.selected_device()
            if device and not device.is_monitorable:
                message = f"Device is output-only and cannot be monitored directly: {device.full_label}"
                self.emit("monitoring.start_failed", source=source, error=message)
                self.file_logger.log("error", "monitoring.start_failed", source=source, error=message)
                return False
            portaudio_index = device.portaudio_index if device else None
            self.detector = live_analysis.BalancedChoppyDetector(
                enable_twitch=self.settings.twitch_enabled,
                audio_device=portaudio_index,
                input_channel_index=self.settings.selected_channel_index,
                twitch_channel=self.settings.twitch_channel,
                twitch_bot_username=self.settings.twitch_bot_username,
                twitch_oauth_token=self.settings.twitch_oauth_token,
                event_callback=self.emit,
                alert_templates=self.settings.alert_templates,
                file_logger=self.file_logger,
            )
            self.thread = threading.Thread(
                target=self._run_detector,
                name="choppy-detector-runtime",
                daemon=True,
                kwargs={"source": source},
            )
            self.thread.start()
            self.emit(
                "monitoring.started",
                source=source,
                device=device.full_label if device else "Default input device",
            )
            self.file_logger.log(
                "info",
                "monitoring.started",
                source=source,
                device=device.full_label if device else "Default input device",
            )
            return True

    def stop(self, source: str = "gui") -> None:
        with self.lock:
            detector = self.detector
            thread = self.thread
            if detector is None:
                self.emit("monitoring.already_stopped", source=source)
                return
            detector.running = False

        if thread and thread.is_alive():
            thread.join(timeout=5.0)

        with self.lock:
            self.detector = None
            self.thread = None

        self.emit("monitoring.stopped_by_request", source=source)
        self.file_logger.log("info", "monitoring.stopped", source=source)

    def restart(self, source: str = "gui") -> bool:
        self.emit("monitoring.restart_requested", source=source)
        self.file_logger.log("info", "monitoring.restart_requested", source=source)
        self.stop(source=source)
        return self.start(source=source)

    def switch_device(self, selection_index: int, source: str = "gui", user: str | None = None) -> tuple[bool, str]:
        devices = self.list_devices()
        if not 0 <= selection_index < len(devices):
            message = f"Invalid device number: {selection_index}"
            self.emit("device.switch_failed", source=source, user=user, error=message)
            self.file_logger.log("warn", "device.switch_failed", source=source, user=user, error=message)
            return False, message

        old_device = self.selected_device()
        new_device = devices[selection_index]
        if not new_device.is_monitorable:
            message = (
                f"{new_device.full_label} is output-only. Route it to a virtual cable or "
                "select its matching input/capture endpoint."
            )
            self.emit("device.switch_failed", source=source, user=user, error=message)
            self.file_logger.log("warn", "device.switch_failed", source=source, user=user, error=message)
            return False, message
        self.settings.selected_device_id = selection_index
        save_settings(self.settings)

        old_label = old_device.full_label if old_device else "Default input device"
        self.emit(
            "device.changed",
            source=source,
            user=user,
            old_device=old_label,
            new_device=new_device.full_label,
        )
        self.file_logger.log(
            "info",
            "device.changed",
            source=source,
            user=user,
            old=old_label,
            new=new_device.full_label,
        )

        if self.is_running:
            self.restart(source=source)
        return True, f"Selected device {new_device.full_label}"

    def list_devices(self) -> list[AudioDeviceInfo]:
        return list_audio_devices(include_output_only=True)

    def selected_device(self) -> AudioDeviceInfo | None:
        devices = self.list_devices()
        selected = self.settings.selected_device_id
        if selected is not None and 0 <= selected < len(devices):
            return devices[selected]
        for device in devices:
            if device.is_monitorable:
                return device
        return devices[0] if devices else None

    def device_summary(self) -> str:
        device = self.selected_device()
        if device:
            return device.full_label
        with self.lock:
            detector = self.detector
            if detector is not None:
                return device_label_for_portaudio_index(detector.audio_device)
        return "No input device"

    def _run_detector(self, source: str) -> None:
        try:
            assert self.detector is not None
            tee_stream = _RuntimeConsoleTee(self.emit)
            with contextlib.redirect_stdout(tee_stream), contextlib.redirect_stderr(tee_stream):
                self.detector.start_detection()
        except Exception as exc:
            self.emit("monitoring.error", source=source, error=str(exc))
            self.file_logger.log("error", "monitoring.error", source=source, error=str(exc))
        finally:
            self.emit("monitoring.thread_exited", source=source)


class _RuntimeConsoleTee(io.TextIOBase):
    """Capture detector prints and forward them as runtime events."""

    def __init__(self, emit_fn: Callable[..., None]):
        super().__init__()
        self._emit = emit_fn
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():
                    self._emit("runtime.console", line=line.rstrip())
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if self._buffer.strip():
                self._emit("runtime.console", line=self._buffer.rstrip())
            self._buffer = ""
