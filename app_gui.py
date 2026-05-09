#!/usr/bin/env python3
"""PySide6 GUI for Choppy Audio Detector."""

from __future__ import annotations

import argparse
from collections import deque
import math
import sys
import threading
import time
from datetime import datetime, timedelta

try:
    from PySide6.QtCore import QObject, QTimer, Signal, Qt
    from PySide6.QtGui import QColor, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGroupBox,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSplashScreen,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 is required for the GUI. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from choppy_detector_gui.alert_templates import severity_for_detection_count
from choppy_detector_gui.advanced_controller import (
    advanced_dirty,
    apply_advanced_to_controls as apply_advanced_controls_controller,
    collect_advanced_from_controls as collect_advanced_from_controls_controller,
    reset_advanced_defaults as reset_advanced_defaults_controller,
)
from choppy_detector_gui.command_service_controller import sync_command_service
from choppy_detector_gui.file_logging import AppFileLogger
from choppy_detector_gui.gui.tabs.advanced_tab import build_advanced_tab as build_advanced_tab_ui
from choppy_detector_gui.gui.tabs.console_tab import build_console_tab as build_console_tab_ui
from choppy_detector_gui.gui.tabs.main_tab import build_main_tab as build_main_tab_ui
from choppy_detector_gui.gui.tabs.responses_tab import build_responses_tab as build_responses_tab_ui
from choppy_detector_gui.gui.tabs.settings_tab import build_settings_tab as build_settings_tab_ui
from choppy_detector_gui.gui.tabs.support_tab import build_support_tab as build_support_tab_ui
from choppy_detector_gui.gui.tabs.websocket_tab import build_websocket_tab as build_websocket_tab_ui
from choppy_detector_gui.obs_connection_controller import build_connection_config, test_connection_once
from choppy_detector_gui.obs_event_policy import decide_obs_event
from choppy_detector_gui.obs_websocket_service import ObsWebSocketService
from choppy_detector_gui.responses_controller import (
    apply_templates_to_controls,
    build_preview_text,
    collect_templates_from_controls,
    reset_template_to_default as reset_template_to_default_controller,
    templates_dirty,
)
from choppy_detector_gui.settings_controller import (
    apply_settings_to_controls as apply_settings_controls_controller,
    collect_settings_from_controls,
    settings_dirty,
)
from choppy_detector_gui.runtime_event_presenter import RuntimeEventPresenter
from choppy_detector_gui.runtime_event_pipeline import RuntimeEventPipeline
from choppy_detector_gui.runtime_event_router import RuntimeEventContext
from choppy_detector_gui.status_badge_presenter import StatusBadgePresenter
from choppy_detector_gui.twitch_status_coordinator import TwitchStatusCoordinator
from choppy_detector_gui.runtime import DetectorRuntime
from choppy_detector_gui.settings import (
    AppSettings,
    load_settings,
    save_settings,
)
from choppy_detector_gui.twitch_command_service import TwitchCommandService
from choppy_detector_gui.websocket_settings_controller import (
    apply_obs_settings_to_controls as apply_obs_settings_to_controls_controller,
    collect_obs_from_controls as collect_obs_from_controls_controller,
    update_obs_bundle_network_notice as update_obs_bundle_network_notice_controller,
    websocket_dirty,
)

try:
    import numpy as np
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

DARK_THEME_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #3f3f3f;
    color: #ececec;
}
QGroupBox {
    border: 1px solid #5a5a5a;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #161616;
    color: #ededed;
    border: 1px solid #505050;
    selection-background-color: #2a82da;
}
QPushButton, QToolButton {
    background-color: #666;
    color: #eee;
    border: 1px solid #7a7a7a;
    border-radius: 5px;
    padding: 4px 8px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #757575;
}
QPushButton:disabled, QToolButton:disabled {
    color: #a0a0a0;
    background-color: #535353;
}
QTabWidget::pane {
    border: 1px solid #565656;
}
QTabBar::tab {
    background: #4c4c4c;
    color: #ececec;
    border: 1px solid #5f5f5f;
    border-bottom: none;
    padding: 6px 14px;
    min-width: 72px;
}
QTabBar::tab:selected {
    background: #2a82da;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background: #5b5b5b;
}
QCheckBox {
    spacing: 6px;
}
QScrollBar:vertical {
    background: #2d2d2d;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #6a6a6a;
    min-height: 20px;
}
"""

OBS_AUTO_CONNECT_RETRY_DELAY_MS = 60_000
OBS_AUTO_CONNECT_MAX_ATTEMPTS = 5


class RuntimeSignals(QObject):
    event = Signal(str, object)
    obs_event = Signal(str, object)


class MainWindow(QMainWindow):
    def __init__(self, launch_options: argparse.Namespace | None = None):
        super().__init__()
        self.setWindowTitle("Choppy Audio Detector")
        self.resize(860, 620)
        self.setMinimumSize(740, 540)

        self.settings = load_settings()
        self.auto_start_requested = False
        self.apply_launch_options(launch_options)
        self.file_logger = AppFileLogger(self.settings.log_settings)
        self.signals = RuntimeSignals()
        self.signals.event.connect(self.handle_runtime_event)
        self.signals.obs_event.connect(self.handle_obs_event)
        self.runtime = DetectorRuntime(
            self.settings,
            event_handler=lambda event_type, payload: self.signals.event.emit(event_type, payload),
            file_logger=self.file_logger,
        )
        self.command_service = TwitchCommandService(
            self.settings,
            self.runtime,
            event_handler=lambda event_type, payload: self.signals.event.emit(event_type, payload),
            file_logger=self.file_logger,
        )
        self.runtime_event_presenter = RuntimeEventPresenter(self)
        self.twitch_status = TwitchStatusCoordinator()
        self.runtime_event_pipeline = RuntimeEventPipeline(
            twitch_status=self.twitch_status,
            set_twitch_status_badge=self.set_twitch_status_badge,
            is_alerts_enabled=lambda: bool(self.settings.twitch_enabled),
            is_chat_enabled=lambda: bool(self.settings.chat_commands.chat_commands_enabled),
            context_provider=self._runtime_event_context,
            presenter=self.runtime_event_presenter,
            queue_obs_refresh_request=self.queue_obs_refresh_request,
        )
        self.obs_service = ObsWebSocketService()
        self._loading_devices = False
        self._last_audio_level_seen_at: datetime | None = None
        self._monitoring_started_at: datetime | None = None
        self._audio_watchdog_warned = False
        self._monitoring_ui_active = False
        self._meter_preview_stream = None
        self._meter_preview_lock = threading.Lock()
        self._meter_preview_peak_dbfs = -120.0
        self._meter_preview_rms_dbfs = -120.0
        self._meter_preview_last_update = datetime.min
        self._display_peak_dbfs = -120.0
        self._display_rms_dbfs = -120.0
        self._display_level_last_update_at = time.monotonic()
        self._obs_last_auto_refresh_at = 0.0
        self._obs_auto_connect_attempt = 0
        self._obs_auto_connect_retry_timer = QTimer(self)
        self._obs_auto_connect_retry_timer.setSingleShot(True)
        self._obs_auto_connect_retry_timer.timeout.connect(self.attempt_obs_auto_connect)
        self._recent_event_entries: deque[tuple[datetime, str]] = deque()
        self._console_entries: deque[tuple[datetime, str]] = deque()
        self._log_window_cleanup_timer = QTimer(self)
        self._log_window_cleanup_timer.timeout.connect(self.prune_log_windows)
        self._log_window_cleanup_timer.start(60_000)

        self.auto_restart_timer = QTimer(self)
        self.auto_restart_timer.timeout.connect(self.auto_restart)
        self.audio_watchdog_timer = QTimer(self)
        self.audio_watchdog_timer.timeout.connect(self.check_audio_watchdog)
        self.audio_watchdog_timer.start(1000)
        self.meter_preview_timer = QTimer(self)
        self.meter_preview_timer.timeout.connect(self.refresh_meter_preview_ui)
        self.update_meter_refresh_timer()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.global_twitch_status_badge = QLabel("Twitch: Idle")
        self.global_obs_status_badge = QLabel("OBS: Disconnected")
        self.statusBar().setStyleSheet("QStatusBar::item { border: none; }")
        self.statusBar().addPermanentWidget(self.global_twitch_status_badge)
        self.statusBar().addPermanentWidget(self.global_obs_status_badge)
        self.twitch_badge_presenter = StatusBadgePresenter(self.global_twitch_status_badge, prefix="Twitch")
        self.obs_badge_presenter = StatusBadgePresenter(self.global_obs_status_badge, prefix="OBS")
        self.build_main_tab()
        self.build_templates_tab()
        self.build_settings_tab()
        self.build_advanced_tab()
        self.build_websocket_tab()
        self.build_console_tab()
        self.build_support_tab()
        self._last_tab_index = self.tabs.currentIndex()
        self.tabs.currentChanged.connect(self.handle_tab_changed)
        self.refresh_devices()
        self.apply_settings_to_controls()
        self.apply_theme()
        self.update_auto_restart_timer()
        self.update_command_service()
        self.restart_meter_preview()
        self.append_console("GUI started.")
        if self.settings.obs_websocket.enabled and self.settings.obs_websocket.auto_connect_on_launch:
            QTimer.singleShot(250, self.start_obs_auto_connect)
        if self.auto_start_requested or self.settings.auto_start_monitoring:
            QTimer.singleShot(120, self.start_monitoring)

    def apply_launch_options(self, options: argparse.Namespace | None) -> None:
        if options is None:
            return
        if getattr(options, "audio_device", None) is not None:
            self.settings.selected_device_id = int(options.audio_device)
        if getattr(options, "audio_channel", None) is not None:
            self.settings.selected_channel_index = max(0, int(options.audio_channel))
            self.auto_start_requested = True
        if getattr(options, "twitch", False):
            self.settings.twitch_enabled = True
        if getattr(options, "no_twitch", False):
            self.settings.twitch_enabled = False
        if getattr(options, "twitch_channel", None):
            self.settings.twitch_channel = str(options.twitch_channel).strip().lstrip("#")
        if getattr(options, "twitch_bot_username", None):
            self.settings.twitch_bot_username = str(options.twitch_bot_username).strip()
        if getattr(options, "twitch_oauth_token", None):
            self.settings.twitch_oauth_token = str(options.twitch_oauth_token).strip()
        save_settings(self.settings)

    def build_main_tab(self) -> None:
        build_main_tab_ui(self)

    def build_templates_tab(self) -> None:
        build_responses_tab_ui(self)

    def build_settings_tab(self) -> None:
        build_settings_tab_ui(self)

    def build_support_tab(self) -> None:
        build_support_tab_ui(self)

    def build_websocket_tab(self) -> None:
        build_websocket_tab_ui(self)

    def build_console_tab(self) -> None:
        build_console_tab_ui(self)

    def build_advanced_tab(self) -> None:
        build_advanced_tab_ui(self)

    def _advanced_row(
        self,
        key: str,
        desc: str,
        value_type: str,
        min_v: float,
        max_v: float,
        step: float,
        default_value,
    ) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        left = QWidget()
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        title = QLabel(key)
        title.setStyleSheet("font-weight: 600;")
        left_layout.addWidget(title)

        if value_type == "int":
            input_widget = QSpinBox()
            input_widget.setRange(int(min_v), int(max_v))
            input_widget.setSingleStep(int(step))
        elif value_type == "float":
            input_widget = QDoubleSpinBox()
            input_widget.setRange(float(min_v), float(max_v))
            input_widget.setSingleStep(float(step))
            input_widget.setDecimals(4 if step < 0.01 else 2)
        else:
            input_widget = QCheckBox()
        input_widget.setMinimumWidth(180)
        self.advanced_widgets[f"value:{key}"] = input_widget
        left_layout.addWidget(input_widget)
        left_layout.addStretch(1)

        desc_label = QLabel(f"{desc} (Default: {self._format_default_value(default_value)})")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #bdbdbd;")
        row_layout.addWidget(left, 3)
        row_layout.addWidget(desc_label, 5)
        return row_widget

    def _format_default_value(self, value) -> str:
        if isinstance(value, bool):
            return "On" if value else "Off"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def alert_config_schema(self):
        return (
            ("detections_for_alert", "Detections required before an alert can fire.", "int", 1, 100, 1),
            ("alert_cooldown_ms", "Minimum time between alerts in milliseconds.", "int", 1000, 3600000, 1000),
            ("detection_window_seconds", "Window for counting detections toward an alert.", "int", 5, 600, 1),
            ("confidence_threshold", "Minimum confidence percent to count a detection.", "int", 1, 100, 1),
            ("clean_audio_reset_seconds", "Seconds of clean audio before episode resets.", "int", 5, 600, 1),
            ("event_dedup_seconds", "Suppress duplicate events from one burst.", "float", 0.0, 10.0, 0.1),
            ("fast_alert_burst_detections", "Fast alert count in short window.", "int", 1, 50, 1),
            ("fast_alert_window_seconds", "Short window for fast alert path.", "int", 1, 300, 1),
            ("fast_alert_min_confidence", "Minimum confidence for fast alert path.", "int", 1, 100, 1),
            ("log_possible_glitches", "Log lower-confidence possible glitches.", "bool", 0, 1, 1),
            ("possible_log_min_confidence", "Minimum confidence for possible glitch logging.", "float", 0.0, 1.0, 0.01),
            ("possible_log_interval_seconds", "Minimum seconds between possible glitch logs.", "float", 0.0, 60.0, 0.5),
            ("max_alert_age_seconds", "Drop queued alerts older than this age.", "float", 1.0, 120.0, 0.5),
            ("max_alert_send_window_seconds", "Max send+retry window before giving up.", "float", 0.5, 60.0, 0.5),
            ("twitch_send_failures_for_pause", "Consecutive send failures before pausing alerts.", "int", 1, 20, 1),
            ("twitch_send_pause_seconds", "Pause duration when circuit breaker opens.", "float", 1.0, 600.0, 1.0),
        )

    def threshold_schema(self):
        return (
            ("silence_ratio", "Silence ratio needed to consider silence-gaps suspicious.", "float", 0.0, 1.0, 0.01),
            ("amplitude_jump", "Relative jump threshold for amplitude jump detector.", "float", 0.0, 20.0, 0.1),
            ("envelope_discontinuity", "Envelope discontinuity threshold.", "float", 0.0, 20.0, 0.1),
            ("modulation_freq_min_hz", "Lower envelope modulation frequency bound.", "float", 0.0, 200.0, 0.5),
            ("modulation_freq_max_hz", "Upper envelope modulation frequency bound.", "float", 0.0, 200.0, 0.5),
            ("modulation_strength", "Peak-versus-floor modulation strength threshold.", "float", 0.0, 50.0, 0.1),
            ("modulation_depth", "Required envelope modulation depth.", "float", 0.0, 1.0, 0.01),
            ("modulation_peak_concentration", "Required concentration of modulation peak energy.", "float", 0.0, 1.0, 0.01),
            ("gap_duration_ms", "Minimum silence gap duration in ms.", "int", 1, 2000, 1),
            ("min_audio_level", "Minimum RMS needed before analysis runs.", "float", 0.0, 1.0, 0.0005),
            ("max_normal_gaps", "Normal gaps allowed before suspicious.", "int", 0, 20, 1),
            ("suspicious_gap_count", "Significant gap count considered suspicious.", "int", 1, 30, 1),
        )

    def methods_schema(self):
        return (
            ("silence_gaps", "Detects dropouts and long silence breaks."),
            ("amplitude_jumps", "Detects abrupt amplitude swings."),
            ("envelope_discontinuity", "Detects sudden envelope breaks."),
            ("amplitude_modulation", "Detects rapid robotic/flutter modulation texture."),
            ("temporal_consistency", "Detects unstable temporal loudness consistency."),
            ("energy_variance", "Detects unusual energy variance."),
            ("zero_crossings", "Detects zero-crossing pattern anomalies."),
            ("spectral_rolloff", "Detects spectral rolloff instability."),
            ("spectral_centroid", "Detects spectral centroid instability."),
        )

    def apply_settings_to_controls(self) -> None:
        apply_templates_to_controls(self)
        apply_settings_controls_controller(self)
        self.apply_obs_settings_to_controls()
        self.apply_advanced_to_controls()
        self.refresh_channel_options()
        self.update_meter_refresh_timer()
        self.update_twitch_status_from_settings()
        self.update_obs_controls_enabled()

    def apply_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setStyleSheet(DARK_THEME_STYLESHEET if self.settings.dark_mode_enabled else "")

    def update_meter_refresh_timer(self) -> None:
        fps = max(5, min(60, int(self.settings.preview_meter_fps)))
        interval_ms = max(10, int(round(1000.0 / fps)))
        self.meter_preview_timer.start(interval_ms)

    def set_twitch_status_badge(self, label: str, color_hex: str) -> None:
        self.twitch_badge_presenter.apply(label, color_hex)

    def _is_chat_commands_connected(self) -> bool:
        bot = getattr(self.command_service, "bot", None)
        return bool(
            self.command_service.running
            and bot is not None
            and bool(getattr(bot, "connected", False))
        )

    def _is_alert_twitch_connected(self) -> bool:
        detector = self.runtime.detector
        if detector is None:
            return False
        twitch_bot = getattr(detector, "twitch_bot", None)
        return bool(getattr(detector, "twitch_enabled", False) and twitch_bot and getattr(twitch_bot, "connected", False))

    def update_twitch_status_from_settings(self) -> None:
        label, color = self.twitch_status.sync_from_settings(
            alerts_enabled=bool(self.settings.twitch_enabled),
            chat_enabled=bool(self.settings.chat_commands.chat_commands_enabled),
            chat_connected=self._is_chat_commands_connected(),
            chat_running=bool(self.command_service.running),
            alert_connected=self._is_alert_twitch_connected(),
        )
        self.set_twitch_status_badge(label, color)

    def refresh_devices(self) -> None:
        self._loading_devices = True
        self.device_combo.clear()
        devices = self.runtime.list_devices()
        for device in devices:
            self.device_combo.addItem(device.display_name, device.selection_index)
        if devices:
            selected = self.settings.selected_device_id
            if (
                selected is None
                or not 0 <= selected < len(devices)
                or not devices[selected].is_monitorable
            ):
                selected = next(
                    (device.selection_index for device in devices if device.is_monitorable),
                    0,
                )
            self.device_combo.setCurrentIndex(selected)
            self.settings.selected_device_id = selected
        self._loading_devices = False
        self.refresh_channel_options()
        self.update_device_controls()
        self.restart_meter_preview()

    def device_selection_changed(self) -> None:
        if self._loading_devices:
            return
        selection = self.device_combo.currentData()
        if selection is None:
            return
        selected_device = self.selected_combo_device()
        self.update_device_controls()
        if selected_device and not selected_device.is_monitorable:
            self.append_event(
                f"{selected_device.full_label} is output-only. Select an input/capture or loopback device to monitor audio."
            )
            return
        self.refresh_channel_options()
        self.restart_meter_preview()
        if self.runtime.is_running:
            ok, message = self.runtime.switch_device(int(selection), source="gui")
            if not ok:
                QMessageBox.warning(self, "Device switch failed", message)
        else:
            self.settings.selected_device_id = int(selection)
            save_settings(self.settings)
            self.append_event(f"Selected device {self.device_combo.currentText()}")

    def channel_selection_changed(self) -> None:
        if self._loading_devices:
            return
        channel_index = self.channel_combo.currentData()
        if channel_index is None:
            return
        self.settings.selected_channel_index = int(channel_index)
        save_settings(self.settings)
        self.restart_meter_preview()
        if self.runtime.is_running:
            self.restart_monitoring()

    def start_monitoring(self) -> None:
        selected_device = self.selected_combo_device()
        if selected_device and not selected_device.is_monitorable:
            QMessageBox.warning(
                self,
                "Output-only device",
                (
                    f"{selected_device.full_label} is an output/render device and cannot be monitored directly.\n\n"
                    "Select an input/capture device, or route this output through a loopback/virtual cable device "
                    "and select that capture endpoint."
                ),
            )
            self.update_device_controls()
            return
        self.save_main_settings()
        if not self.settings.keep_preview_while_monitoring:
            self.stop_meter_preview()
        self._last_audio_level_seen_at = None
        self._monitoring_started_at = datetime.now()
        self._audio_watchdog_warned = False
        self._monitoring_ui_active = True
        self.update_device_controls()
        self.status_label.setText("Starting")
        started = self.runtime.start(source="gui")
        if not started:
            self._monitoring_ui_active = False
            self.update_device_controls()
            self.status_label.setText("Stopped")
            return
        if self.settings.keep_preview_while_monitoring:
            QTimer.singleShot(700, self.ensure_meter_preview_stream)

    def stop_monitoring(self) -> None:
        self._monitoring_ui_active = False
        self.update_device_controls()
        self.runtime.stop(source="gui")
        self.status_label.setText("Stopped")
        self._monitoring_started_at = None
        self._audio_watchdog_warned = False
        self.restart_meter_preview()

    def restart_monitoring(self) -> None:
        selected_device = self.selected_combo_device()
        if selected_device and not selected_device.is_monitorable:
            QMessageBox.warning(
                self,
                "Output-only device",
                "Select a monitorable input/capture device before restarting detection.",
            )
            self.update_device_controls()
            return
        self.save_main_settings()
        if not self.settings.keep_preview_while_monitoring:
            self.stop_meter_preview()
        self._last_audio_level_seen_at = None
        self._monitoring_started_at = datetime.now()
        self._audio_watchdog_warned = False
        self._monitoring_ui_active = True
        self.update_device_controls()
        self.status_label.setText("Restarting")
        restarted = self.runtime.restart(source="gui")
        if not restarted:
            self._monitoring_ui_active = False
            self.update_device_controls()
            self.status_label.setText("Stopped")
            return
        if self.settings.keep_preview_while_monitoring:
            QTimer.singleShot(700, self.ensure_meter_preview_stream)

    def selected_combo_device(self):
        selection = self.device_combo.currentData()
        if selection is None:
            return None
        for device in self.runtime.list_devices():
            if device.selection_index == int(selection):
                return device
        return None

    def update_device_controls(self) -> None:
        device = self.selected_combo_device()
        is_monitorable = bool(device and device.is_monitorable)
        monitoring_active = self._monitoring_ui_active or self.runtime.is_running
        self.start_button.setEnabled(is_monitorable and not monitoring_active)
        self.restart_button.setEnabled(is_monitorable)
        self.stop_button.setEnabled(monitoring_active)
        if device and not device.is_monitorable:
            self.device_hint_label.setText(
                "⚠ Selected device is output-only. On macOS, route output through BlackHole/Loopback and select the matching input/capture endpoint."
            )
            self.device_hint_label.setStyleSheet("color: #f0c04a; font-weight: 600;")
            self.device_hint_label.show()
        else:
            self.device_hint_label.clear()
            self.device_hint_label.setStyleSheet("color: #bdbdbd;")
            self.device_hint_label.hide()

    def auto_restart(self) -> None:
        if self.runtime.is_running:
            self.append_console("Scheduled restart triggered.")
            self.runtime.restart(source="scheduled")

    def save_main_settings(self) -> None:
        self.settings.twitch_enabled = self.twitch_enabled.isChecked()
        self.settings.chat_commands.chat_commands_enabled = self.chat_commands_enabled.isChecked()
        selection = self.device_combo.currentData()
        if selection is not None:
            self.settings.selected_device_id = int(selection)
        channel_index = self.channel_combo.currentData()
        if channel_index is not None:
            self.settings.selected_channel_index = int(channel_index)
        save_settings(self.settings)
        self.update_command_service()
        self.update_twitch_status_from_settings()

    def save_templates(self) -> None:
        templates = self.collect_templates()
        errors = templates.validate_all()
        if errors:
            QMessageBox.warning(self, "Template validation failed", "\n".join(errors))
            return
        self.settings.alert_templates = templates
        save_settings(self.settings)
        self.append_console("Response templates saved.")

    def preview_templates(self) -> None:
        preview_text, _ = build_preview_text(self)
        self.template_preview.setPlainText(preview_text)

    def collect_templates(self):
        return collect_templates_from_controls(self)

    def is_templates_dirty(self) -> bool:
        return templates_dirty(self)

    def reset_template_to_default(self, template_key: str) -> None:
        reset_template_to_default_controller(self, template_key)

    def copy_template_token(self, token: str) -> None:
        QApplication.clipboard().setText(token)
        self.append_console(f"Copied token {token}")

    def save_all_settings(self) -> None:
        collect_settings_from_controls(self)
        self.collect_obs_from_controls()
        self.collect_advanced_from_controls()
        self.file_logger.settings = self.settings.log_settings
        save_settings(self.settings)
        self.apply_theme()
        self.update_meter_refresh_timer()
        self.update_auto_restart_timer()
        self.update_command_service()
        self.restart_meter_preview()
        self.update_meter_display_mode()
        self.update_twitch_status_from_settings()
        self.prune_log_windows()
        self.append_console("Settings saved.")

    def is_settings_dirty(self) -> bool:
        return settings_dirty(self)

    def apply_obs_settings_to_controls(self) -> None:
        apply_obs_settings_to_controls_controller(self)

    def collect_obs_from_controls(self) -> None:
        collect_obs_from_controls_controller(self)

    def update_obs_bundle_network_notice(self) -> None:
        update_obs_bundle_network_notice_controller(self)

    def save_obs_settings(self) -> None:
        self.collect_obs_from_controls()
        save_settings(self.settings)
        self.append_console("WebSocket settings saved.")
        self.update_obs_controls_enabled()

    def is_websocket_dirty(self) -> bool:
        return websocket_dirty(self)

    def is_advanced_dirty(self) -> bool:
        return advanced_dirty(self)

    def handle_tab_changed(self, new_index: int) -> None:
        previous_index = self._last_tab_index
        self._last_tab_index = new_index
        previous_widget = self.tabs.widget(previous_index)
        if previous_widget is self.settings_tab and self.is_settings_dirty():
            answer = QMessageBox.question(
                self,
                "Unsaved Settings",
                "You have unsaved changes in Settings. Save now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.save_all_settings()
        elif previous_widget is self.templates_tab and self.is_templates_dirty():
            answer = QMessageBox.question(
                self,
                "Unsaved Templates",
                "You have unsaved changes in Responses. Save now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.save_templates()
        elif previous_widget is self.advanced_tab and self.is_advanced_dirty():
            answer = QMessageBox.question(
                self,
                "Unsaved Advanced Changes",
                "You have unsaved changes in Advanced. Save now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.save_advanced_settings()
        elif previous_widget is getattr(self, "websocket_tab", None) and self.is_websocket_dirty():
            answer = QMessageBox.question(
                self,
                "Unsaved WebSocket Changes",
                "You have unsaved changes in WebSocket. Save now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.save_obs_settings()

    def set_obs_status(self, label: str, color_hex: str) -> None:
        self.obs_status.setText(label)
        self.obs_status.setStyleSheet(f"color: {color_hex}; font-weight: 700;")
        self.obs_badge_presenter.apply(label, color_hex)

    def start_obs_auto_connect(self) -> None:
        self._obs_auto_connect_attempt = 0
        self.attempt_obs_auto_connect()

    def attempt_obs_auto_connect(self) -> None:
        if self.obs_service.is_connected:
            return
        max_attempts = OBS_AUTO_CONNECT_MAX_ATTEMPTS
        if self._obs_auto_connect_attempt >= max_attempts:
            self.append_console(f"OBS auto-connect failed after {max_attempts} attempts.")
            self.update_obs_controls_enabled()
            return
        self._obs_auto_connect_attempt += 1
        attempt = self._obs_auto_connect_attempt
        self.append_console(f"OBS auto-connect attempt {attempt}/{max_attempts}...")
        cfg = build_connection_config(self.settings)
        self.set_obs_status("Connecting", "#4aa3ff")
        self.set_obs_busy(True)
        self._run_obs_task(
            "connect_auto",
            lambda: self.obs_service.connect(cfg),
            {"attempt": attempt, "max_attempts": max_attempts},
        )

    def cancel_obs_auto_connect_retry(self) -> None:
        self._obs_auto_connect_retry_timer.stop()
        self._obs_auto_connect_attempt = 0

    def connect_obs(self) -> None:
        self.cancel_obs_auto_connect_retry()
        if not self.obs_enabled.isChecked():
            QMessageBox.information(self, "OBS WebSocket Disabled", "Enable OBS WebSocket integration first.")
            return
        self.collect_obs_from_controls()
        cfg = build_connection_config(self.settings)
        self.set_obs_status("Connecting", "#4aa3ff")
        self.set_obs_busy(True)
        self._run_obs_task("connect", lambda: self.obs_service.connect(cfg))

    def test_obs_connection(self) -> None:
        if not self.obs_enabled.isChecked():
            QMessageBox.information(self, "OBS WebSocket Disabled", "Enable OBS WebSocket integration first.")
            return
        self.collect_obs_from_controls()
        self.set_obs_status("Testing", "#4aa3ff")
        self.set_obs_busy(True)
        self._run_obs_task("test", lambda: test_connection_once(self.settings))

    def disconnect_obs(self) -> None:
        self.cancel_obs_auto_connect_retry()
        self.obs_service.disconnect()
        self.set_obs_status("Disconnected", "#ff9c4a")
        self.append_console("Disconnected from OBS WebSocket.")
        self.update_obs_controls_enabled()

    def refresh_obs_scenes(self) -> None:
        selected_before = self.obs_target_scene.currentText().strip() or self.settings.obs_websocket.target_scene
        self.obs_target_scene.blockSignals(True)
        self.obs_target_scene.clear()
        self.obs_target_scene.addItem("All Scenes")
        scenes = self.obs_service.list_scenes()
        if scenes:
            self.obs_target_scene.addItems(scenes)
            idx = self.obs_target_scene.findText(selected_before)
            self.obs_target_scene.setCurrentIndex(0 if idx < 0 else idx)
        self.obs_target_scene.blockSignals(False)

    def refresh_obs_sources(self) -> None:
        selected_before = self.obs_target_source.currentText().strip() or self.settings.obs_websocket.target_source
        selected_scene = self.obs_target_scene.currentText().strip()
        self.refresh_obs_scenes()
        self.obs_target_source.blockSignals(True)
        self.obs_target_source.clear()
        sources = self.obs_service.list_sources()
        if sources:
            self.obs_target_source.addItems(sources)
            idx = self.obs_target_source.findText(selected_before)
            self.obs_target_source.setCurrentIndex(max(0, idx))
            self.obs_refresh_now_button.setEnabled(True)
            self.set_obs_status("Connected", "#3fcf5e")
            if selected_before and idx < 0:
                self.append_console(f"Saved OBS source '{selected_before}' no longer exists. Select a new source.")
        else:
            if selected_before:
                self.obs_target_source.addItem(selected_before)
                self.obs_target_source.setCurrentIndex(0)
            self.obs_refresh_now_button.setEnabled(bool(selected_before))
        if selected_scene and selected_scene != "All Scenes" and self.obs_target_scene.findText(selected_scene) < 0:
            self.append_console(f"Saved OBS scene '{selected_scene}' no longer exists. Using scene-agnostic mode.")
        self.obs_target_source.blockSignals(False)
        self.update_obs_controls_enabled()

    def refresh_obs_source_now(self) -> None:
        if not self.obs_enabled.isChecked():
            QMessageBox.information(self, "OBS WebSocket Disabled", "Enable OBS WebSocket integration first.")
            return
        source = self.obs_target_source.currentText().strip()
        if not source:
            QMessageBox.warning(self, "No Source Selected", "Choose an OBS source first.")
            return
        self.queue_obs_refresh_request(source=source, action="refresh")

    def queue_obs_refresh_request(self, source: str, action: str = "refresh") -> None:
        self.set_obs_status("Refreshing", "#4aa3ff")
        self.set_obs_busy(True)
        self._run_obs_task(
            action,
            lambda: self.obs_service.refresh_source_in_scene(
                source_name=source,
                scene_name="" if self.obs_target_scene.currentText().strip() == "All Scenes" else self.obs_target_scene.currentText().strip(),
                off_on_delay_ms=self.obs_refresh_off_on_delay_ms.value(),
            ),
        )

    def set_obs_busy(self, busy: bool) -> None:
        self.obs_connect_button.setEnabled(not busy)
        self.obs_disconnect_button.setEnabled(not busy)
        self.obs_test_button.setEnabled(not busy)
        self.obs_refresh_now_button.setEnabled(not busy and bool(self.obs_target_source.currentText().strip()))
        self.obs_refresh_sources_button.setEnabled(not busy)
        self.obs_refresh_scenes_button.setEnabled(not busy)

    def update_obs_controls_enabled(self) -> None:
        enabled = self.obs_enabled.isChecked()
        if not enabled:
            self.cancel_obs_auto_connect_retry()
        if not enabled and self.obs_service.is_connected:
            self.obs_service.disconnect()
            self.append_console("OBS WebSocket integration disabled: disconnected existing OBS session.")
        widgets = (
            self.obs_host,
            self.obs_port,
            self.obs_password,
            self.obs_connect_button,
            self.obs_disconnect_button,
            self.obs_test_button,
            self.obs_target_source,
            self.obs_target_scene,
            self.obs_refresh_sources_button,
            self.obs_refresh_scenes_button,
            self.obs_refresh_now_button,
            self.obs_auto_refresh_enabled,
            self.obs_auto_refresh_min_severity,
            self.obs_auto_refresh_cooldown_sec,
            self.obs_refresh_off_on_delay_ms,
            self.obs_save_button,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        if not enabled:
            self.set_obs_status("Disabled", "#8f8f8f")
            return
        if self.obs_service.is_connected:
            self.set_obs_status("Connected", "#3fcf5e")
        else:
            self.set_obs_status("Disconnected", "#ff9c4a")
        # Refresh button should require a selected source.
        if enabled:
            self.obs_refresh_now_button.setEnabled(bool(self.obs_target_source.currentText().strip()))

    def _run_obs_task(self, action: str, fn, context: dict[str, object] | None = None) -> None:
        def worker() -> None:
            try:
                ok, message = fn()
            except Exception as exc:
                ok, message = False, str(exc)
            payload: dict[str, object] = {"ok": ok, "message": message}
            if context:
                payload.update(context)
            self.signals.obs_event.emit(action, payload)

        threading.Thread(target=worker, name=f"obs-{action}-worker", daemon=True).start()

    def handle_obs_event(self, action: str, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        ok = bool(data.get("ok", False))
        message = str(data.get("message", ""))
        self.set_obs_busy(False)
        decision = decide_obs_event(
            action=action,
            ok=ok,
            message=message,
            attempt=int(data.get("attempt") or self._obs_auto_connect_attempt or 1),
            max_attempts=int(data.get("max_attempts") or OBS_AUTO_CONNECT_MAX_ATTEMPTS),
            retry_enabled=bool(self.settings.obs_websocket.auto_connect_retry_enabled),
            retry_delay_ms=OBS_AUTO_CONNECT_RETRY_DELAY_MS,
        )

        if decision.cancel_auto_connect_retry:
            self.cancel_obs_auto_connect_retry()
        if decision.schedule_auto_connect_retry_ms:
            self._obs_auto_connect_retry_timer.start(decision.schedule_auto_connect_retry_ms)

        self.set_obs_status(decision.status_label, decision.status_color)
        for line in decision.console_messages:
            self.append_console(line)
        for line in decision.event_messages:
            self.append_event(line)

        if decision.dialog is not None:
            if decision.dialog.level == "warning":
                QMessageBox.warning(self, decision.dialog.title, decision.dialog.message)
            else:
                QMessageBox.information(self, decision.dialog.title, decision.dialog.message)
        if decision.refresh_sources:
            self.refresh_obs_sources()
        if decision.update_controls:
            self.update_obs_controls_enabled()

    def apply_advanced_to_controls(self) -> None:
        apply_advanced_controls_controller(self)

    def collect_advanced_from_controls(self) -> None:
        collect_advanced_from_controls_controller(self)

    def save_advanced_settings(self) -> None:
        self.collect_advanced_from_controls()
        save_settings(self.settings)
        self.append_console("Advanced settings saved.")

    def reset_advanced_defaults(self) -> None:
        reset_advanced_defaults_controller(self)
        self.apply_advanced_to_controls()
        save_settings(self.settings)
        self.append_console("Advanced settings reset to defaults.")

    def _set_advanced_widget_value(self, widget: QWidget, value) -> None:
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))

    def _get_advanced_widget_value(self, widget: QWidget, default_value):
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        return default_value

    def update_auto_restart_timer(self) -> None:
        self.auto_restart_timer.stop()
        interval_ms = self.settings.auto_restart_minutes * 60 * 1000
        self.auto_restart_timer.start(interval_ms)

    def update_command_service(self) -> None:
        sync_command_service(
            settings=self.settings,
            command_service=self.command_service,
            twitch_status=self.twitch_status,
            set_badge=self.set_twitch_status_badge,
            append_console=self.append_console,
        )

    def handle_runtime_event(self, event_type: str, payload: object) -> None:
        self.runtime_event_pipeline.handle(event_type, payload)

    def _runtime_event_context(self) -> RuntimeEventContext:
        return RuntimeEventContext(
            obs_enabled=self.obs_enabled.isChecked(),
            obs_connected=self.obs_service.is_connected,
            selected_source=self.obs_target_source.currentText().strip(),
            saved_source=self.settings.obs_websocket.target_source.strip(),
            runtime_running=self.runtime.is_running,
            keep_preview_while_monitoring=self.settings.keep_preview_while_monitoring,
        )

    def maybe_trigger_obs_auto_refresh(self, glitch_data: dict[str, object]) -> None:
        obs_settings = self.settings.obs_websocket
        if not obs_settings.enabled or not obs_settings.auto_refresh_enabled:
            return
        if not self.obs_service.is_connected:
            self.append_console("OBS auto-refresh skipped: OBS is not connected.")
            return

        source = self.obs_target_source.currentText().strip() or obs_settings.target_source.strip()
        scene_choice = self.obs_target_scene.currentText().strip()
        scene = "" if scene_choice == "All Scenes" else (scene_choice or obs_settings.target_scene.strip())
        if not source:
            self.append_console("OBS auto-refresh skipped: no target source selected.")
            return

        event_severity = self.derive_glitch_severity(glitch_data)
        required_severity = (obs_settings.auto_refresh_min_severity or "severe").strip().lower()
        if not self.severity_meets_threshold(event_severity, required_severity):
            self.append_console(
                f"OBS auto-refresh skipped: event severity {event_severity} below threshold {required_severity}."
            )
            return

        now = time.monotonic()
        cooldown_sec = max(0, int(obs_settings.auto_refresh_cooldown_sec))
        elapsed = now - self._obs_last_auto_refresh_at
        if self._obs_last_auto_refresh_at > 0 and elapsed < cooldown_sec:
            remaining = max(0.0, cooldown_sec - elapsed)
            self.append_console(f"OBS auto-refresh skipped: cooldown active ({remaining:.1f}s remaining).")
            return

        self._obs_last_auto_refresh_at = now
        self.append_console(
            f"OBS auto-refresh triggered: severity {event_severity} met threshold {required_severity}."
        )
        self.set_obs_status("Auto Refreshing", "#4aa3ff")
        self.set_obs_busy(True)
        self._run_obs_task(
            "auto_refresh",
            lambda: self.obs_service.refresh_source_in_scene(
                source_name=source,
                scene_name=scene,
                off_on_delay_ms=obs_settings.refresh_off_on_delay_ms,
            ),
        )

    def derive_glitch_severity(self, glitch_data: dict[str, object]) -> str:
        raw = str(glitch_data.get("severity", "")).strip().lower()
        if raw in {"minor", "moderate", "severe"}:
            return raw
        detection_count = glitch_data.get("detection_count")
        try:
            count = int(detection_count)
            if count >= 0:
                template_severity = severity_for_detection_count(count).strip("[]").lower()
                if template_severity in {"minor", "moderate", "severe"}:
                    return template_severity
        except Exception:
            pass
        confidence = glitch_data.get("confidence")
        try:
            score = float(confidence)
            if not math.isfinite(score):
                raise ValueError("non-finite confidence")
        except Exception:
            return "minor"

        if score >= 90.0:
            return "severe"
        if score >= 75.0:
            return "moderate"
        return "minor"

    def severity_meets_threshold(self, event_severity: str, threshold: str) -> bool:
        rank = {"minor": 1, "moderate": 2, "severe": 3}
        return rank.get(event_severity, 1) >= rank.get(threshold, 3)

    def format_event(self, event_type: str, data: dict[str, object]) -> str:
        if event_type == "glitch.detected":
            return f"Glitch detected: {data.get('confidence')}% - {data.get('reasons')}"
        if event_type == "device.changed":
            return f"Device changed: {data.get('old_device')} -> {data.get('new_device')}"
        if event_type == "monitoring.started":
            return f"Monitoring runtime started: {data.get('device')}"
        if event_type == "chat_commands.accepted":
            return f"Chat command accepted: {data.get('user')} -> {data.get('action')}"
        if event_type == "chat_commands.rejected":
            return f"Chat command rejected: {data.get('user')} ({data.get('reason')})"
        if event_type in {"twitch.connecting", "chat_commands.connecting"}:
            return f"{event_type}: connecting as {data.get('username')} to {data.get('channel')}"
        if event_type == "chat_commands.reconnecting":
            attempt = data.get("attempt")
            if attempt is not None:
                return f"chat_commands.reconnecting: attempt {attempt} as {data.get('username')} to {data.get('channel')}"
            return f"chat_commands.reconnecting: {data.get('username')} to {data.get('channel')}"
        if event_type == "chat_commands.reconnect_scheduled":
            return (
                f"chat_commands.reconnect_scheduled: retrying in {data.get('delay_seconds')}s "
                f"(attempt {data.get('attempt')})"
            )
        if event_type == "audio.stream_opened":
            return (
                f"Audio stream opened: {data.get('device')} "
                f"({data.get('sample_rate')} Hz, {data.get('channels')} channel, using ch {int(data.get('channel_index', 0)) + 1})"
            )
        if event_type == "audio.callback_started":
            return "Audio callbacks started."
        if "error" in data:
            return f"{event_type}: {data.get('error')}"
        return event_type

    def check_audio_watchdog(self) -> None:
        if not self.runtime.is_running or self._audio_watchdog_warned:
            return
        if self._monitoring_started_at is None:
            return
        if (datetime.now() - self._monitoring_started_at).total_seconds() < 4:
            return
        if self._last_audio_level_seen_at is None:
            self.append_console(
                "No audio level callbacks received yet. On macOS, check Microphone permission for the app that launched Python."
            )
            self._audio_watchdog_warned = True
            return
        seconds_since_level = (datetime.now() - self._last_audio_level_seen_at).total_seconds()
        if seconds_since_level > 5:
            self.append_console(
                "Audio stream is running, but no recent meter updates were received. Check the selected input and macOS microphone permission."
            )
            self._audio_watchdog_warned = True

    def append_event(self, message: str) -> None:
        self._append_log_message(self.recent_events, self._recent_event_entries, message)

    def append_console(self, message: str) -> None:
        self._append_log_message(self.console_output, self._console_entries, message)

    def clear_console_messages(self) -> None:
        self._console_entries.clear()
        self.console_output.clear()

    def _append_log_message(
        self,
        widget: QPlainTextEdit,
        entries: deque[tuple[datetime, str]],
        message: str,
    ) -> None:
        now = datetime.now()
        entries.append((now, message))
        if self._prune_log_entries(entries, now=now):
            self._render_log_entries(widget, entries)
            return
        widget.appendPlainText(self.timestamped(message, now=now))

    def prune_log_windows(self) -> None:
        now = datetime.now()
        if self._prune_log_entries(self._recent_event_entries, now=now):
            self._render_log_entries(self.recent_events, self._recent_event_entries)
        if self._prune_log_entries(self._console_entries, now=now):
            self._render_log_entries(self.console_output, self._console_entries)

    def _prune_log_entries(self, entries: deque[tuple[datetime, str]], now: datetime) -> bool:
        if not entries:
            return False
        retention_minutes = max(1, int(self.settings.log_settings.log_window_retention_minutes))
        cutoff = now - timedelta(minutes=retention_minutes)
        removed = False
        while entries and entries[0][0] < cutoff:
            entries.popleft()
            removed = True
        return removed

    def _render_log_entries(self, widget: QPlainTextEdit, entries: deque[tuple[datetime, str]]) -> None:
        if not entries:
            widget.clear()
            return
        widget.setPlainText("\n".join(self.timestamped(message, now=entry_time) for entry_time, message in entries))
        scrollbar = widget.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def timestamped(self, message: str, now: datetime | None = None) -> str:
        at = now or datetime.now()
        return f"[{at.strftime('%H:%M:%S')}] {message}"

    def closeEvent(self, event) -> None:
        self.stop_meter_preview()
        self.command_service.stop()
        self.cancel_obs_auto_connect_retry()
        self.obs_service.disconnect()
        self.runtime.stop(source="gui-close")
        super().closeEvent(event)

    def refresh_channel_options(self) -> None:
        device = self.selected_combo_device()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        if device and device.is_monitorable and device.max_input_channels > 0:
            for index in range(int(device.max_input_channels)):
                self.channel_combo.addItem(f"Channel {index + 1}", index)
            selected = min(
                max(0, int(self.settings.selected_channel_index)),
                int(device.max_input_channels) - 1,
            )
            self.channel_combo.setCurrentIndex(selected)
            self.settings.selected_channel_index = selected
        self.channel_combo.blockSignals(False)

    def restart_meter_preview(self) -> None:
        self.stop_meter_preview()
        if (self.runtime.is_running and not self.settings.keep_preview_while_monitoring) or not SOUNDDEVICE_AVAILABLE:
            self.update_meter_display_mode()
            return
        device = self.selected_combo_device()
        if not device or not device.is_monitorable:
            self.update_meter_display_mode()
            return
        channels = max(1, int(self.settings.selected_channel_index) + 1)
        try:
            samplerate = int(round(float(device.default_samplerate or 44100)))
            self._meter_preview_stream = sd.InputStream(
                device=device.portaudio_index,
                samplerate=samplerate,
                channels=channels,
                blocksize=1024,
                callback=self._meter_preview_callback,
            )
            self._meter_preview_stream.start()
            self.update_meter_display_mode()
        except Exception as exc:
            self.append_console(f"Meter preview unavailable: {exc}")
            self._meter_preview_stream = None
            self.update_meter_display_mode()

    def stop_meter_preview(self) -> None:
        stream = self._meter_preview_stream
        self._meter_preview_stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _meter_preview_callback(self, indata, frames, time_info, status) -> None:
        if status:
            return
        try:
            if indata.ndim > 1:
                idx = min(self.settings.selected_channel_index, indata.shape[1] - 1)
                audio_data = indata[:, idx]
            else:
                audio_data = indata
            rms = float(np.sqrt(np.mean(audio_data**2)))
            peak = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
            rms_dbfs = 20.0 * np.log10(rms + 1e-12)
            peak_dbfs = 20.0 * np.log10(peak + 1e-12)
            with self._meter_preview_lock:
                self._meter_preview_peak_dbfs = max(-120.0, min(0.0, peak_dbfs))
                self._meter_preview_rms_dbfs = max(-120.0, min(0.0, rms_dbfs))
                self._meter_preview_last_update = datetime.now()
        except Exception:
            return

    def _smooth_display_levels(self, peak_dbfs: float, rms_dbfs: float) -> tuple[float, float]:
        if not self.settings.smooth_preview_meter:
            self._display_peak_dbfs = max(-120.0, min(0.0, float(peak_dbfs)))
            self._display_rms_dbfs = max(-120.0, min(0.0, float(rms_dbfs)))
            self._display_level_last_update_at = time.monotonic()
            return self._display_peak_dbfs, self._display_rms_dbfs

        now = time.monotonic()
        dt = max(0.001, now - self._display_level_last_update_at)
        self._display_level_last_update_at = now

        # Ballistics tuned for smoother motion: quick attack, slower release.
        attack_tau = 0.07
        release_tau = 0.24

        def step(current: float, target: float) -> float:
            tau = attack_tau if target >= current else release_tau
            alpha = 1.0 - np.exp(-dt / tau)
            next_value = current + (target - current) * float(alpha)
            return max(-120.0, min(0.0, float(next_value)))

        self._display_peak_dbfs = step(self._display_peak_dbfs, peak_dbfs)
        self._display_rms_dbfs = step(self._display_rms_dbfs, rms_dbfs)
        return self._display_peak_dbfs, self._display_rms_dbfs

    def refresh_meter_preview_ui(self) -> None:
        if self.runtime.is_running and not self.settings.keep_preview_while_monitoring:
            self.update_meter_display_mode()
            return
        if self.runtime.is_running and self.settings.keep_preview_while_monitoring:
            self.ensure_meter_preview_stream()
        if self._meter_preview_stream is None:
            self.update_meter_display_mode()
            return
        with self._meter_preview_lock:
            peak_dbfs = self._meter_preview_peak_dbfs
            rms_dbfs = self._meter_preview_rms_dbfs
            seen_at = self._meter_preview_last_update
        if seen_at == datetime.min:
            return
        self.update_meter_display_mode()
        smooth_peak_dbfs, smooth_rms_dbfs = self._smooth_display_levels(peak_dbfs, rms_dbfs)
        self.peak_meter.set_level_dbfs(smooth_peak_dbfs, peak_source=True)
        self.rms_meter.set_level_dbfs(smooth_rms_dbfs, peak_source=False)
        self.level_text.setText(f"Peak {peak_dbfs:.1f} dBFS | RMS {rms_dbfs:.1f} dBFS")

    def update_meter_display_mode(self) -> None:
        preview_disabled = self.runtime.is_running and not self.settings.keep_preview_while_monitoring
        self.peak_meter.setEnabled(not preview_disabled)
        self.rms_meter.setEnabled(not preview_disabled)
        if preview_disabled:
            self.level_text.setText("Preview disabled while monitoring")

    def ensure_meter_preview_stream(self) -> None:
        """Keep preview stream alive when experimental mode is enabled."""
        stream = self._meter_preview_stream
        if stream is None:
            self.restart_meter_preview()
            return
        try:
            if not stream.active:
                self.restart_meter_preview()
        except Exception:
            self.restart_meter_preview()


def main() -> int:
    args = parse_gui_arguments()
    if args.list_devices:
        import live_analysis

        live_analysis.list_audio_devices()
        return 0
    if args.twitch and args.no_twitch:
        print("[ERROR] Cannot specify both --twitch and --no-twitch")
        return 1
    app = QApplication(sys.argv)
    splash_pixmap = QPixmap(420, 180)
    splash_pixmap.fill(QColor("#1f1f1f"))
    splash = QSplashScreen(splash_pixmap)
    splash.showMessage(
        "Choppy loading...\nDeveloped by HeroHarmony",
        Qt.AlignCenter,
        QColor("#f2f2f2"),
    )
    splash.show()
    app.processEvents()
    window = MainWindow(launch_options=args)
    window.show()
    splash.finish(window)
    return app.exec()


def parse_gui_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choppy Audio Detector GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_gui.py --audio-device 2
  python app_gui.py --audio-device 2 --audio-channel 1
  python app_gui.py --no-twitch
  python app_gui.py --list-devices
        """,
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--audio-device", type=int, metavar="N", help="Select audio input device by GUI index")
    parser.add_argument(
        "--audio-channel",
        "--channel",
        dest="audio_channel",
        type=int,
        metavar="N",
        help="Select input audio channel index and auto-start",
    )
    parser.add_argument("--twitch", action="store_true", help="Enable Twitch chat alerts")
    parser.add_argument("--no-twitch", action="store_true", help="Disable Twitch chat alerts")
    parser.add_argument("--twitch-channel", type=str, help="Twitch channel name (without #)")
    parser.add_argument("--twitch-bot-username", type=str, help="Twitch bot username")
    parser.add_argument("--twitch-oauth-token", type=str, help="Twitch OAuth token")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
