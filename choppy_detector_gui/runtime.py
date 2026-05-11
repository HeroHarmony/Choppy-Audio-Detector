"""Threaded detector runtime used by the GUI."""

from __future__ import annotations

import threading
import contextlib
import io
import json
import re
import time
import wave
from datetime import datetime, timezone
from collections.abc import Callable
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

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
        self._last_clip_capture_at = 0.0
        self.clip_min_interval_seconds = 3.0
        self._monitoring_started_at = 0.0
        self._last_clip_result = ""
        self._last_clip_result_at = 0.0

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

            configured_window_ms = int(
                self.settings.advanced_alert_config.get("production_window_ms", live_analysis.production_window_ms())
            )
            configured_step_ms = int(
                self.settings.advanced_alert_config.get("production_step_ms", live_analysis.production_step_ms())
            )
            live_analysis.configure_production_timing(configured_window_ms, configured_step_ms)
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
                clip_capture_enabled=self.settings.enable_clip_capture_buffer,
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
            self._monitoring_started_at = time.time()
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

    def set_clip_capture_enabled(self, enabled: bool) -> None:
        with self.lock:
            detector = self.detector
            self.settings.enable_clip_capture_buffer = bool(enabled)
        if detector is not None:
            setter = getattr(detector, "set_clip_capture_enabled", None)
            if callable(setter):
                setter(bool(enabled))

    def capture_clip(self, source: str = "twitch", user: str | None = None) -> tuple[bool, str]:
        with self.lock:
            detector = self.detector
            running = bool(detector and getattr(detector, "running", False))
            enabled = bool(self.settings.enable_clip_capture_buffer)
            now = time.time()
            elapsed = now - self._last_clip_capture_at
            min_interval = max(0.0, float(self.clip_min_interval_seconds))

        if not enabled:
            message = "Clip capture is switched off in Settings."
            self.emit("clip.capture_failed", source=source, user=user, reason="capture_disabled", error=message)
            self.file_logger.log("warn", "clip.capture_failed", source=source, user=user, reason="capture_disabled")
            self._last_clip_result = "failed:capture_disabled"
            self._last_clip_result_at = time.time()
            return False, message
        if detector is None or not running:
            message = "Monitoring is not running."
            self.emit("clip.capture_failed", source=source, user=user, reason="monitoring_stopped", error=message)
            self.file_logger.log("warn", "clip.capture_failed", source=source, user=user, reason="monitoring_stopped")
            self._last_clip_result = "failed:monitoring_stopped"
            self._last_clip_result_at = time.time()
            return False, message
        if elapsed < min_interval:
            wait_seconds = max(0.0, min_interval - elapsed)
            message = f"Clip cooldown active ({wait_seconds:.1f}s)."
            self.emit(
                "clip.capture_failed",
                source=source,
                user=user,
                reason="cooldown_active",
                seconds=wait_seconds,
                error=message,
            )
            self.file_logger.log(
                "warn",
                "clip.capture_failed",
                source=source,
                user=user,
                reason="cooldown_active",
                seconds=f"{wait_seconds:.1f}",
            )
            self._last_clip_result = "failed:cooldown_active"
            self._last_clip_result_at = time.time()
            return False, message

        capture_fn = getattr(detector, "capture_recent_audio_clip", None)
        if not callable(capture_fn):
            message = "Detector does not support clip capture."
            self.emit("clip.capture_failed", source=source, user=user, reason="unsupported", error=message)
            self.file_logger.log("error", "clip.capture_failed", source=source, user=user, reason="unsupported")
            self._last_clip_result = "failed:unsupported"
            self._last_clip_result_at = time.time()
            return False, message

        try:
            clip_payload = capture_fn(seconds=30.0)
        except Exception as exc:
            message = f"Clip capture failed while reading buffer: {exc}"
            self.emit("clip.capture_failed", source=source, user=user, reason="capture_exception", error=message)
            self.file_logger.log(
                "error",
                "clip.capture_failed",
                source=source,
                user=user,
                reason="capture_exception",
                error=str(exc),
            )
            self._last_clip_result = "failed:capture_exception"
            self._last_clip_result_at = time.time()
            return False, message
        if not isinstance(clip_payload, dict):
            state_fn = getattr(detector, "clip_capture_state", None)
            state = state_fn() if callable(state_fn) else {}
            callback_started = bool(state.get("callback_started", False))
            buffer_seconds = float(state.get("buffer_seconds", 0.0) or 0.0)
            callback_age = state.get("callback_age_seconds", None)
            stale_active = bool(state.get("audio_stale_active", False))
            if not callback_started:
                message = "Clip buffer not ready: no audio callback data yet."
                if bool(self.settings.keep_preview_while_monitoring):
                    message = (
                        f"{message} Hint: disable 'Keep Preview Running' while monitoring, "
                        "then restart monitoring."
                    )
            elif stale_active:
                message = "Clip buffer not ready: audio callback is stale."
            elif callback_age is not None and float(callback_age) > 5.0:
                message = f"Clip buffer not ready: last callback was {float(callback_age):.1f}s ago."
            else:
                message = f"Clip buffer warming up ({buffer_seconds:.1f}s collected)."
            self.emit(
                "clip.capture_failed",
                source=source,
                user=user,
                reason="buffer_not_ready",
                error=message,
                callback_started=callback_started,
                buffer_seconds=round(buffer_seconds, 3),
                callback_age_seconds=round(float(callback_age), 3) if callback_age is not None else None,
                stale_active=stale_active,
            )
            self.file_logger.log(
                "warn",
                "clip.capture_failed",
                source=source,
                user=user,
                reason="buffer_not_ready",
                callback_started=callback_started,
                buffer_seconds=round(buffer_seconds, 3),
                callback_age_seconds=round(float(callback_age), 3) if callback_age is not None else "",
                stale_active=stale_active,
            )
            self._last_clip_result = "failed:buffer_not_ready"
            self._last_clip_result_at = time.time()
            return False, message

        audio = np.asarray(clip_payload.get("audio"), dtype=np.float32)
        if audio.size <= 0:
            message = "Clip buffer not ready yet."
            self.emit("clip.capture_failed", source=source, user=user, reason="buffer_empty", error=message)
            self.file_logger.log("warn", "clip.capture_failed", source=source, user=user, reason="buffer_empty")
            self._last_clip_result = "failed:buffer_empty"
            self._last_clip_result_at = time.time()
            return False, message

        try:
            sample_rate = int(clip_payload.get("sample_rate") or 44100)
            timestamp = datetime.now(timezone.utc)
            stamp = timestamp.strftime("%Y%m%d_%H%M%S")
            slug = _sanitize_clip_slug(user or "unknown")
            clip_dir = Path.cwd() / "Clips"
            clip_dir.mkdir(parents=True, exist_ok=True)
            wav_name = f"{stamp}_clip_{slug}.wav"
            wav_path = clip_dir / wav_name
            meta_path = clip_dir / f"{stamp}_clip_{slug}.meta.json"

            pcm = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm16.tobytes())

            baseline_snapshot = clip_payload.get("baseline") if isinstance(clip_payload.get("baseline"), dict) else {}
            runtime_snapshot = clip_payload.get("runtime") if isinstance(clip_payload.get("runtime"), dict) else {}
            created_ts = time.time()
            callback_age = max(0.0, created_ts - float(runtime_snapshot.get("last_audio_callback_time", 0.0) or 0.0))
            metadata = {
                "clip_version": 1,
                "created_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "requester_twitch_name": str(user or "").strip() or "unknown",
                "request_channel": str(self.settings.twitch_channel or "").strip(),
                "command_source": source,
                "clip_file": wav_name,
                "clip_seconds": round(float(len(audio) / max(1, sample_rate)), 3),
                "sample_rate": sample_rate,
                "channels": 1,
                "baseline": baseline_snapshot,
                "settings_snapshot": {
                    "alert_config": dict(self.settings.advanced_alert_config),
                    "thresholds": dict(self.settings.advanced_thresholds),
                    "detection_methods": dict(self.settings.detection_methods),
                },
                "runtime_snapshot": {
                    "audio_stale_active": bool(runtime_snapshot.get("audio_stale_active", False)),
                    "stream_restart_requested": bool(runtime_snapshot.get("stream_restart_requested", False)),
                    "last_audio_callback_age_seconds": round(callback_age, 3),
                    "device": runtime_snapshot.get("device") or self.device_summary(),
                    "channel_index": int(runtime_snapshot.get("channel_index", self.settings.selected_channel_index)),
                },
                "reviews": [],
            }
            meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            message = f"Clip save failed: {exc}"
            self.emit("clip.capture_failed", source=source, user=user, reason="io_error", error=message)
            self.file_logger.log(
                "error",
                "clip.capture_failed",
                source=source,
                user=user,
                reason="io_error",
                error=str(exc),
            )
            self._last_clip_result = "failed:io_error"
            self._last_clip_result_at = time.time()
            return False, message

        with self.lock:
            self._last_clip_capture_at = created_ts

        self.emit(
            "clip.captured",
            source=source,
            user=user,
            wav_path=str(wav_path),
            meta_path=str(meta_path),
            seconds=metadata["clip_seconds"],
        )
        self.file_logger.log(
            "info",
            "clip.captured",
            source=source,
            user=user,
            wav=str(wav_path),
            meta=str(meta_path),
            seconds=metadata["clip_seconds"],
        )
        self._last_clip_result = f"ok:{wav_name}"
        self._last_clip_result_at = time.time()
        return True, f"Clip saved: {wav_name}"

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
            self._monitoring_started_at = 0.0

        self.emit("monitoring.stopped_by_request", source=source)
        self.file_logger.log("info", "monitoring.stopped", source=source)

    def restart(self, source: str = "gui") -> bool:
        self.emit("monitoring.restart_requested", source=source)
        self.file_logger.log("info", "monitoring.restart_requested", source=source)
        self.stop(source=source)
        return self.start(source=source)

    def rebuild_baseline(self, source: str = "gui", user: str | None = None) -> tuple[bool, str]:
        with self.lock:
            detector = self.detector
            running = bool(detector and getattr(detector, "running", False))
        if detector is None or not running:
            message = "Monitoring is not running."
            self.emit("baseline.rebuild_failed", source=source, user=user, error=message)
            self.file_logger.log("warn", "baseline.rebuild_failed", source=source, user=user, error=message)
            return False, message
        rebuild_fn = getattr(detector, "rebuild_baseline", None)
        if not callable(rebuild_fn):
            message = "Detector does not support baseline rebuild."
            self.emit("baseline.rebuild_failed", source=source, user=user, error=message)
            self.file_logger.log("error", "baseline.rebuild_failed", source=source, user=user, error=message)
            return False, message
        rebuild_fn(reason=source)
        return True, "Baseline relearn started."

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

    def latest_audio_levels(self) -> dict[str, float] | None:
        with self.lock:
            detector = self.detector
            running = bool(detector and getattr(detector, "running", False))
        if detector is None or not running:
            return None
        getter = getattr(detector, "latest_audio_levels", None)
        if not callable(getter):
            return None
        try:
            levels = getter()
        except Exception:
            return None
        if not isinstance(levels, dict):
            return None
        return {
            "peak_dbfs": float(levels.get("peak_dbfs", -120.0)),
            "rms_dbfs": float(levels.get("rms_dbfs", -120.0)),
            "timestamp": float(levels.get("timestamp", 0.0)),
        }

    def status_snapshot(self) -> dict[str, object]:
        now = time.time()
        with self.lock:
            detector = self.detector
            running = bool(detector and getattr(detector, "running", False))
            started_at = float(self._monitoring_started_at or 0.0)
            last_clip_result = str(self._last_clip_result or "")
            last_clip_result_at = float(self._last_clip_result_at or 0.0)

        snapshot: dict[str, object] = {
            "running": running,
            "device": self.device_summary(),
            "channel_index": int(self.settings.selected_channel_index) + 1,
            "monitoring_uptime_seconds": max(0.0, now - started_at) if started_at > 0 else 0.0,
            "clip_enabled": bool(self.settings.enable_clip_capture_buffer),
            "last_clip_result": last_clip_result,
            "last_clip_age_seconds": max(0.0, now - last_clip_result_at) if last_clip_result_at > 0 else None,
        }
        if detector is None or not running:
            return snapshot

        import live_analysis

        with detector.lock:
            sample_rate = int(getattr(detector, "sample_rate", 0) or 0)
            stream_channels = int(getattr(detector, "stream_channels", 0) or 0)
            last_callback_at = float(getattr(detector, "last_audio_callback_time", 0.0) or 0.0)
            audio_stale_active = bool(getattr(detector, "audio_stale_active", False))
            stream_restart_requested = bool(getattr(detector, "stream_restart_requested", False))
            baseline_stats = dict(getattr(detector, "baseline_stats", {}) or {})
            detection_history = list(getattr(detector, "detection_history", []) or [])
            last_alert_time = float(getattr(detector, "last_alert_time", 0.0) or 0.0)
            twitch_enabled = bool(getattr(detector, "twitch_enabled", False))
            twitch_bot = getattr(detector, "twitch_bot", None)

        levels = self.latest_audio_levels() or {"peak_dbfs": -120.0, "rms_dbfs": -120.0, "timestamp": 0.0}
        callback_age = max(0.0, now - last_callback_at) if last_callback_at > 0 else None
        rms_history: list[float] = []
        for value in list(baseline_stats.get("rms_history", [])):
            if isinstance(value, Real):
                as_float = float(value)
                if np.isfinite(as_float):
                    rms_history.append(as_float)
        rms_cv = None
        if rms_history:
            mean = float(np.mean(np.asarray(rms_history, dtype=np.float64)))
            std = float(np.std(np.asarray(rms_history, dtype=np.float64)))
            rms_cv = std / (mean + 1e-12) if mean > 0 else None

        alert_cfg = live_analysis.ALERT_CONFIG
        threshold = float(alert_cfg.get("confidence_threshold", 70))
        detection_window_seconds = float(alert_cfg.get("detection_window_seconds", 120))
        fast_window_seconds = float(alert_cfg.get("fast_alert_window_seconds", 15))
        fast_min_conf = float(alert_cfg.get("fast_alert_min_confidence", 75))
        detections_for_alert = int(alert_cfg.get("detections_for_alert", 6))
        fast_required = int(alert_cfg.get("fast_alert_burst_detections", 4))
        cooldown_seconds = float(alert_cfg.get("alert_cooldown_ms", self.settings.alert_cooldown_ms)) / 1000.0

        detect_cutoff = now - detection_window_seconds
        fast_cutoff = now - fast_window_seconds
        recent_high = [
            det for det in detection_history
            if float(det.get("timestamp", 0.0) or 0.0) > detect_cutoff
            and float(det.get("confidence", 0.0) or 0.0) >= threshold
        ]
        fast_hits = [
            det for det in detection_history
            if float(det.get("timestamp", 0.0) or 0.0) > fast_cutoff
            and float(det.get("confidence", 0.0) or 0.0) >= fast_min_conf
            and bool(det.get("primary_hit", False))
        ]
        last_detection_ts = max((float(det.get("timestamp", 0.0) or 0.0) for det in detection_history), default=0.0)
        cooldown_remaining = max(0.0, (last_alert_time + cooldown_seconds) - now) if last_alert_time > 0 else 0.0

        clip_state_fn = getattr(detector, "clip_capture_state", None)
        clip_state = clip_state_fn() if callable(clip_state_fn) else {}
        clip_buffer_seconds = float(clip_state.get("buffer_seconds", 0.0) or 0.0) if isinstance(clip_state, dict) else 0.0

        snapshot.update(
            {
                "sample_rate": sample_rate,
                "stream_channels": stream_channels,
                "callback_age_seconds": callback_age,
                "audio_stale_active": audio_stale_active,
                "stream_restart_requested": stream_restart_requested,
                "peak_dbfs": float(levels.get("peak_dbfs", -120.0)),
                "rms_dbfs": float(levels.get("rms_dbfs", -120.0)),
                "baseline_established": bool(baseline_stats.get("established_baseline", False)),
                "baseline_samples": int(len(rms_history)),
                "baseline_cv": rms_cv,
                "baseline_clean_lock_seconds": float(alert_cfg.get("baseline_clean_lock_seconds", 20.0)),
                "baseline_min_samples": int(alert_cfg.get("baseline_min_rms_samples", 20)),
                "baseline_cv_max": float(alert_cfg.get("baseline_rms_cv_max", 1.0)),
                "recent_high_conf": int(len(recent_high)),
                "detections_for_alert": detections_for_alert,
                "detection_window_seconds": detection_window_seconds,
                "fast_hits": int(len(fast_hits)),
                "fast_required": fast_required,
                "fast_window_seconds": fast_window_seconds,
                "cooldown_remaining_seconds": cooldown_remaining,
                "last_detection_age_seconds": (max(0.0, now - last_detection_ts) if last_detection_ts > 0 else None),
                "twitch_alerts_enabled": twitch_enabled,
                "twitch_alerts_connected": bool(twitch_bot and getattr(twitch_bot, "connected", False)),
                "clip_buffer_seconds": clip_buffer_seconds,
            }
        )
        return snapshot

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


def _sanitize_clip_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip())
    normalized = normalized.strip("-_")
    if not normalized:
        return "unknown"
    return normalized[:32]
