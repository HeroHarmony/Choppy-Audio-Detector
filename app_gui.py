#!/usr/bin/env python3
"""PySide6 GUI for Choppy Audio Detector."""

from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import re
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
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSplashScreen,
        QSpinBox,
        QTableWidgetItem,
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
    apply_advanced_to_controls as apply_advanced_controls_controller,
    collect_advanced_from_controls as collect_advanced_from_controls_controller,
    reset_advanced_defaults as reset_advanced_defaults_controller,
)
from choppy_detector_gui.command_service_controller import sync_command_service
from choppy_detector_gui.file_logging import AppFileLogger
from choppy_detector_gui.gui.tabs.advanced_tab import build_advanced_tab as build_advanced_tab_ui
from choppy_detector_gui.gui.tabs.console_tab import build_console_tab as build_console_tab_ui
from choppy_detector_gui.gui.tabs.main_tab import build_main_tab as build_main_tab_ui
from choppy_detector_gui.gui.tabs.playground_tab import build_playground_tab as build_playground_tab_ui
from choppy_detector_gui.gui.tabs.responses_tab import build_responses_tab as build_responses_tab_ui
from choppy_detector_gui.gui.tabs.settings_tab import build_settings_tab as build_settings_tab_ui
from choppy_detector_gui.gui.tabs.support_tab import build_support_tab as build_support_tab_ui
from choppy_detector_gui.gui.tabs.websocket_tab import build_websocket_tab as build_websocket_tab_ui
from choppy_detector_gui.obs_workflow_service import (
    attempt_obs_auto_connect as attempt_obs_auto_connect_service,
    connect_obs as connect_obs_service,
    disconnect_obs as disconnect_obs_service,
    maybe_trigger_obs_auto_refresh as maybe_trigger_obs_auto_refresh_service,
    queue_obs_refresh_request as queue_obs_refresh_request_service,
    refresh_obs_source_now as refresh_obs_source_now_service,
    start_obs_auto_connect as start_obs_auto_connect_service,
    test_obs_connection as test_obs_connection_service,
)
from choppy_detector_gui.obs_event_policy import decide_obs_event
from choppy_detector_gui.obs_websocket_service import ObsWebSocketService
from choppy_detector_gui.responses_controller import (
    apply_templates_to_controls,
    build_preview_text,
    collect_templates_from_controls,
    reset_template_to_default as reset_template_to_default_controller,
)
from choppy_detector_gui.settings_controller import (
    apply_settings_to_controls as apply_settings_controls_controller,
)
from choppy_detector_gui.settings_save_service import save_all_settings as save_all_settings_service
from choppy_detector_gui.tab_dirty_service import handle_tab_changed as handle_tab_changed_service
from choppy_detector_gui.runtime_event_presenter import RuntimeEventPresenter
from choppy_detector_gui.runtime_event_pipeline import RuntimeEventPipeline
from choppy_detector_gui.runtime_event_router import RuntimeEventContext
from choppy_detector_gui.status_badge_presenter import StatusBadgePresenter
from choppy_detector_gui.twitch_status_coordinator import TwitchStatusCoordinator
from choppy_detector_gui.runtime import DetectorRuntime
from choppy_detector_gui.playground_analysis import (
    LoadedWavFile,
    analyze_wav_file,
    load_baseline_sidecar,
    load_marker_sidecar,
    load_wav_file,
    save_baseline_sidecar,
    save_marker_sidecar,
    write_mono_wav_file,
    write_compact_report,
)
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
QProgressBar {
    background-color: #2a2a2a;
    color: #ececec;
    border: 1px solid #5a5a5a;
    border-radius: 3px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2a82da;
}
QTableView, QTableWidget {
    background-color: #3f3f3f;
    alternate-background-color: #4a4a4a;
    color: #ececec;
    gridline-color: #5a5a5a;
    selection-background-color: #2a82da;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #4d4d4d;
    color: #ececec;
    border: 1px solid #5a5a5a;
    padding: 3px 6px;
}
QTableCornerButton::section {
    background-color: #4d4d4d;
    border: 1px solid #5a5a5a;
}
"""

OBS_AUTO_CONNECT_RETRY_DELAY_MS = 60_000
OBS_AUTO_CONNECT_MAX_ATTEMPTS = 5


class RuntimeSignals(QObject):
    event = Signal(str, object)
    obs_event = Signal(str, object)
    playground_analysis_done = Signal(object)
    playground_analysis_failed = Signal(str)


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
        self.signals.playground_analysis_done.connect(self._on_playground_analysis_done)
        self.signals.playground_analysis_failed.connect(self._on_playground_analysis_failed)
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
        self._obs_scene_watch_last_scene = ""
        self._obs_scene_watch_entered_at = 0.0
        self._obs_scene_watch_pending_rebuild_at = 0.0
        self._obs_scene_watch_pending_from_scene = ""
        self._obs_scene_watch_pending_to_scene = ""
        self._obs_scene_watch_last_rebuild_at = 0.0
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
        self.obs_scene_watch_timer = QTimer(self)
        self.obs_scene_watch_timer.timeout.connect(self.check_obs_scene_rebuild_automation)
        self.obs_scene_watch_timer.start(1000)
        self.meter_preview_timer = QTimer(self)
        self.meter_preview_timer.timeout.connect(self.refresh_meter_preview_ui)
        self.update_meter_refresh_timer()
        self.playground_playback_timer = QTimer(self)
        self.playground_playback_timer.timeout.connect(self.refresh_playground_playback_status)
        self.playground_live_timer = QTimer(self)
        self.playground_live_timer.timeout.connect(self.refresh_live_playground_status)
        self._playground_audio_file = None
        self._playground_loaded_files: list[LoadedWavFile] = []
        self._playground_is_playing = False
        self._playground_started_at_monotonic = 0.0
        self._playground_play_duration_seconds = 0.0
        self._playground_live_stream = None
        self._playground_live_running = False
        self._playground_live_sample_rate = 0
        self._playground_live_channel_index = 0
        self._playground_live_started_at_monotonic = 0.0
        self._playground_live_chunks: list[np.ndarray] = []
        self._playground_live_lock = threading.Lock()
        self._playground_analysis_running = False
        self._playground_analysis_thread: threading.Thread | None = None
        self._playground_preview_active_path: str | None = None
        self._playground_preview_active_channel_index: int = 0
        self._playground_markers_by_path: dict[str, list[int]] = {}
        self._playground_baseline_profiles_by_path: dict[str, dict] = {}
        self._playground_marker_flash_phase = False
        self._playground_display_peak_dbfs = -120.0
        self._playground_display_rms_dbfs = -120.0
        self._playground_display_last_update_at = time.monotonic()
        self._playground_live_paused_monitoring = False
        self._playground_live_baseline_profile: dict | None = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.global_audio_status_badge = QLabel("Audio: Idle")
        self.global_twitch_status_badge = QLabel("Twitch: Idle")
        self.global_obs_status_badge = QLabel("OBS: Disconnected")
        self.statusBar().setStyleSheet("QStatusBar::item { border: none; }")
        self.statusBar().addPermanentWidget(self.global_audio_status_badge)
        self.statusBar().addPermanentWidget(self.global_twitch_status_badge)
        self.statusBar().addPermanentWidget(self.global_obs_status_badge)
        self.audio_badge_presenter = StatusBadgePresenter(self.global_audio_status_badge, prefix="Audio")
        self.twitch_badge_presenter = StatusBadgePresenter(self.global_twitch_status_badge, prefix="Twitch")
        self.obs_badge_presenter = StatusBadgePresenter(self.global_obs_status_badge, prefix="OBS")
        self.set_audio_status_badge("Idle", "#8f8f8f")
        build_main_tab_ui(self)
        build_responses_tab_ui(self)
        build_settings_tab_ui(self)
        build_advanced_tab_ui(self)
        build_websocket_tab_ui(self)
        build_playground_tab_ui(self)
        build_console_tab_ui(self)
        build_support_tab_ui(self)
        self._last_tab_index = self.tabs.currentIndex()
        self.tabs.currentChanged.connect(self.handle_tab_changed)
        self.refresh_devices()
        apply_templates_to_controls(self)
        apply_settings_controls_controller(self)
        apply_obs_settings_to_controls_controller(self)
        apply_advanced_controls_controller(self)
        self.refresh_channel_options()
        self.update_meter_refresh_timer()
        self.update_twitch_status_from_settings()
        self.update_obs_controls_enabled()
        self.apply_theme()
        self.update_auto_restart_timer()
        self.update_command_service()
        self.restart_meter_preview()
        self.update_playground_controls()
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
            ("production_window_ms", "Live detection analysis window size in milliseconds.", "int", 100, 4000, 50),
            ("production_step_ms", "Live detection step/cadence in milliseconds.", "int", 10, 1000, 10),
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

    def apply_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setStyleSheet(DARK_THEME_STYLESHEET if self.settings.dark_mode_enabled else "")

    def update_meter_refresh_timer(self) -> None:
        interval_ms = self._preview_timer_interval_ms()
        self.meter_preview_timer.start(interval_ms)
        playback_timer = getattr(self, "playground_playback_timer", None)
        if (
            getattr(self, "_playground_is_playing", False)
            and playback_timer is not None
            and playback_timer.isActive()
        ):
            playback_timer.start(interval_ms)

    def set_twitch_status_badge(self, label: str, color_hex: str) -> None:
        self.twitch_badge_presenter.apply(label, color_hex)

    def set_audio_status_badge(self, label: str, color_hex: str) -> None:
        self.audio_badge_presenter.apply(label, color_hex)

    def _preview_timer_interval_ms(self) -> int:
        fps = max(5, min(60, int(self.settings.preview_meter_fps)))
        return max(10, int(round(1000.0 / fps)))

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
        self.update_playground_controls()

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
        self.update_playground_controls()
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
        self.update_playground_controls()
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

    def relearn_baseline(self) -> None:
        ok, message = self.runtime.rebuild_baseline(source="gui")
        if ok:
            self.append_console("Baseline relearn requested from Main tab.")
        else:
            QMessageBox.information(self, "Relearn Baseline", message)
        self.update_device_controls()

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
        self.rebuild_baseline_button.setEnabled(monitoring_active)
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

    def browse_playground_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WAV file",
            "",
            "WAV files (*.wav);;All files (*)",
        )
        if not path:
            return
        self.load_playground_files([path])

    def browse_playground_files_batch(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select WAV files",
            "",
            "WAV files (*.wav);;All files (*)",
        )
        if not paths:
            return
        self.load_playground_files(paths)

    def load_playground_file_clicked(self) -> None:
        path = self.playground_file_path.text().strip()
        if not path:
            QMessageBox.information(self, "Playground", "Choose a WAV file first.")
            return
        parts = [piece.strip() for piece in re.split(r"[|,]", path) if piece.strip()]
        if not parts:
            QMessageBox.information(self, "Playground", "Choose a WAV file first.")
            return
        self.load_playground_files(parts)

    def load_playground_file(self, path: str) -> None:
        self.load_playground_files([path])

    def load_playground_files(self, paths: list[str]) -> None:
        self.stop_playground_audio()
        unique_paths: list[str] = []
        for raw in paths:
            normalized = str(raw).strip()
            if not normalized:
                continue
            if normalized not in unique_paths:
                unique_paths.append(normalized)
        if not unique_paths:
            return

        loaded_files: list[LoadedWavFile] = []
        failures: list[tuple[str, str]] = []
        for path in unique_paths:
            try:
                loaded_files.append(load_wav_file(path))
            except Exception as exc:
                failures.append((path, str(exc)))

        if not loaded_files:
            self._playground_audio_file = None
            self._playground_loaded_files = []
            self._playground_markers_by_path = {}
            self._playground_baseline_profiles_by_path = {}
            self._sync_playground_progress_for_loaded(None)
            self.update_playground_controls()
            error_lines = "\n".join(f"{p}: {err}" for p, err in failures[:5])
            QMessageBox.warning(self, "WAV load failed", error_lines or "No WAV files were loaded.")
            return

        self._playground_loaded_files = loaded_files
        self._playground_audio_file = loaded_files[0]
        self._playground_markers_by_path = {}
        self._playground_baseline_profiles_by_path = {}
        for loaded in loaded_files:
            self._set_playground_markers(loaded.path, load_marker_sidecar(loaded.path))
            baseline_payload = load_baseline_sidecar(loaded.path)
            if baseline_payload is not None:
                self._playground_baseline_profiles_by_path[loaded.path] = baseline_payload
        self._sync_playground_progress_for_loaded(self._playground_audio_file)
        self.playground_file_path.setText(" | ".join(file.path for file in loaded_files))

        max_channel_count = max(file.channel_count for file in loaded_files)
        self.playground_channel_spin.setRange(1, max(1, max_channel_count))
        self.playground_channel_spin.setValue(min(self.playground_channel_spin.value(), max_channel_count))

        if len(loaded_files) == 1:
            loaded = loaded_files[0]
            self.playground_file_info.setPlainText(
                f"{loaded.path} | {loaded.sample_rate} Hz | {loaded.channel_count} channel(s) | "
                f"{loaded.duration_ms / 1000.0:.2f}s"
            )
        else:
            total_sec = sum(file.duration_ms for file in loaded_files) / 1000.0
            sample_rates = sorted({file.sample_rate for file in loaded_files})
            channel_counts = sorted({file.channel_count for file in loaded_files})
            preview_names = ", ".join(Path(file.path).name for file in loaded_files[:3])
            if len(loaded_files) > 3:
                preview_names = f"{preview_names}, ..."
            self.playground_file_info.setPlainText(
                f"{len(loaded_files)} files loaded | Total duration: {total_sec:.2f}s | "
                f"Sample rates: {sample_rates} | Channels: {channel_counts}\n"
                f"{preview_names}"
            )

        tagged_count = sum(
            1 for file in loaded_files if self.infer_expected_outcome_from_filename(file.path) is not None
        )
        if tagged_count > 0 and self.playground_preview_on_done.isChecked():
            self.playground_preview_on_done.setChecked(False)
            self.append_console(
                "Playground preview-on-done disabled because expected-outcome tag(s) were found in filename(s)."
            )

        if len(loaded_files) == 1:
            self.playground_analysis_summary.setPlainText("File loaded. Run analysis to inspect telemetry.")
        else:
            self.playground_analysis_summary.setPlainText(
                f"{len(loaded_files)} files loaded. Run batch analysis to inspect telemetry and save reports."
            )
        self.playground_table.setRowCount(0)
        self.update_playground_table_height()
        self.update_playground_marker_status()
        self.append_console(
            f"Playground loaded {len(loaded_files)} WAV file(s): "
            + ", ".join(Path(file.path).name for file in loaded_files)
        )
        baseline_loaded_count = len(self._playground_baseline_profiles_by_path)
        if baseline_loaded_count:
            self.append_console(
                f"Playground loaded baseline profile sidecar(s): {baseline_loaded_count}"
            )
        if failures:
            error_lines = "\n".join(f"{Path(p).name}: {err}" for p, err in failures[:5])
            QMessageBox.warning(
                self,
                "Some files failed to load",
                f"{len(failures)} file(s) failed and were skipped:\n{error_lines}",
            )
        self.update_playground_controls()

    def _playground_prod_timing(self) -> tuple[int, int]:
        configured = self.settings.advanced_alert_config if isinstance(self.settings.advanced_alert_config, dict) else {}
        window_ms = int(configured.get("production_window_ms", 1000))
        step_ms = int(configured.get("production_step_ms", 50))
        return max(100, window_ms), max(10, step_ms)

    def _playground_comparison_timing(self) -> tuple[int, int]:
        """Legacy baseline timing kept for A/B comparison in Playground."""
        return 2000, 200

    def _snapshot_runtime_baseline_profile(self) -> dict | None:
        with self.runtime.lock:
            detector = self.runtime.detector
            running = bool(detector and getattr(detector, "running", False))
            if detector is None or not running:
                return None
            with detector.lock:
                rms_history = [
                    float(v)
                    for v in list((detector.baseline_stats or {}).get("rms_history", []))
                    if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0
                ]
                established = bool((detector.baseline_stats or {}).get("established_baseline", False))
                baseline_rms = float(detector.get_baseline_rms())
                audio_device = detector.audio_device
                input_channel_index = int(detector.input_channel_index)
                sample_rate = int(detector.sample_rate)

        return {
            "version": 1,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "source": "main_runtime",
            "device": self.runtime.device_summary(),
            "audio_device_index": audio_device,
            "input_channel_index": input_channel_index,
            "sample_rate": sample_rate,
            "production_window_ms": int(self.settings.advanced_alert_config.get("production_window_ms", 1000)),
            "production_step_ms": int(self.settings.advanced_alert_config.get("production_step_ms", 50)),
            "baseline": {
                "rms": baseline_rms,
                "sample_count": int(len(rms_history)),
                "established": 1 if established else 0,
                "rms_history": rms_history[-50:],
            },
        }

    def _resume_main_monitoring_after_live_capture(self) -> None:
        if not self._playground_live_paused_monitoring:
            return
        self._playground_live_paused_monitoring = False
        if self.runtime.is_running:
            return
        self.start_monitoring()

    def _playground_current_markers(self) -> list[int]:
        marker_path = self._playground_marker_context_path()
        if marker_path is None:
            return []
        return list(self._playground_markers_by_path.get(marker_path, []))

    def _playground_marker_context_path(self) -> str | None:
        if self._playground_is_playing and self._playground_preview_active_path:
            return self._playground_preview_active_path
        loaded = self._playground_audio_file
        if loaded is None:
            return None
        return loaded.path

    def _playground_find_loaded_by_path(self, path: str) -> LoadedWavFile | None:
        for loaded in self._playground_loaded_files:
            if loaded.path == path:
                return loaded
        return None

    def _set_playground_markers(self, path: str, markers_ms: list[int]) -> None:
        cleaned = sorted({max(0, int(round(v))) for v in markers_ms})
        self._playground_markers_by_path[path] = cleaned
        marker_path = self._playground_marker_context_path()
        if marker_path is not None and marker_path == path:
            self.playground_progress.set_markers_ms(cleaned)
        self.update_playground_marker_status()

    def _sync_playground_progress_for_loaded(self, loaded: LoadedWavFile | None) -> None:
        if loaded is None:
            self.playground_progress.set_duration_ms(1)
            self.playground_progress.set_position_ms(0)
            self.playground_progress.set_markers_ms([])
            self.playground_progress.set_marker_alert(active_marker_ms=None, flash_on=False)
            self._reset_playground_preview_meters()
            return
        self.playground_progress.set_duration_ms(int(max(1, loaded.duration_ms)))
        self.playground_progress.set_markers_ms(list(self._playground_markers_by_path.get(loaded.path, [])))
        if not self._playground_is_playing:
            self._reset_playground_preview_meters()

    def remove_playground_marker_at(self, marker_ms: int) -> None:
        marker_path = self._playground_marker_context_path()
        if marker_path is None:
            return
        loaded = self._playground_find_loaded_by_path(marker_path)
        if loaded is None:
            return
        markers = list(self._playground_markers_by_path.get(marker_path, []))
        try:
            markers.remove(int(marker_ms))
        except ValueError:
            return
        self._set_playground_markers(marker_path, markers)
        self._save_playground_markers_for_loaded()
        self.append_console(f"Playground marker removed at {int(marker_ms)}ms ({Path(loaded.path).name})")

    def update_playground_marker_status(self) -> None:
        loaded = self._playground_audio_file
        if loaded is None:
            self.playground_marker_status.setText("Markers: 0")
            return
        markers = self._playground_current_markers()
        self.playground_marker_status.setText(f"Markers: {len(markers)}")

    def update_playground_table_height(self) -> None:
        row_count = int(self.playground_table.rowCount())
        header_height = int(self.playground_table.horizontalHeader().height())
        frame_height = int(self.playground_table.frameWidth()) * 2
        rows_height = 0
        for row in range(row_count):
            rows_height += int(self.playground_table.rowHeight(row))
        min_height = 180
        target = max(min_height, header_height + frame_height + rows_height + 8)
        self.playground_table.setMinimumHeight(target)
        self.playground_table.setMaximumHeight(target)

    def _save_playground_markers_for_loaded(self) -> None:
        marker_path = self._playground_marker_context_path()
        if marker_path is None:
            return
        loaded = self._playground_find_loaded_by_path(marker_path)
        if loaded is None:
            return
        markers = list(self._playground_markers_by_path.get(marker_path, []))
        save_marker_sidecar(loaded, markers)

    def add_playground_marker(self) -> None:
        marker_path = self._playground_marker_context_path()
        if marker_path is None:
            QMessageBox.information(self, "Playground marker", "Load a WAV file first.")
            return
        if not self._playground_is_playing:
            QMessageBox.information(self, "Playground marker", "Start preview playback before adding markers.")
            return
        loaded = self._playground_find_loaded_by_path(marker_path)
        if loaded is None:
            QMessageBox.information(self, "Playground marker", "No active loaded WAV context for marker save.")
            return
        elapsed_ms = int(round(max(0.0, time.monotonic() - self._playground_started_at_monotonic) * 1000.0))
        latency_ms = int(self.playground_marker_latency_ms_spin.value())
        marker_ms = max(0, min(int(loaded.duration_ms), elapsed_ms - latency_ms))
        markers = list(self._playground_markers_by_path.get(marker_path, []))
        markers.append(marker_ms)
        self._set_playground_markers(marker_path, markers)
        self._save_playground_markers_for_loaded()
        self.append_console(f"Playground marker added at {marker_ms}ms ({Path(loaded.path).name})")

    def clear_playground_markers(self) -> None:
        marker_path = self._playground_marker_context_path()
        if marker_path is None:
            return
        loaded = self._playground_find_loaded_by_path(marker_path)
        if loaded is None:
            return
        self._set_playground_markers(marker_path, [])
        self._save_playground_markers_for_loaded()
        self.append_console(f"Playground markers cleared for {Path(loaded.path).name}")

    def update_playground_controls(self) -> None:
        file_count = len(self._playground_loaded_files)
        has_file = file_count > 0
        batch_mode = file_count > 1
        live_running = self._playground_live_running
        analysis_running = self._playground_analysis_running
        compare_window_ms, compare_step_ms = self._playground_comparison_timing()
        controls_locked = live_running or analysis_running
        self.playground_browse_button.setEnabled(not controls_locked)
        self.playground_load_button.setEnabled(not controls_locked)
        self.playground_file_path.setEnabled(not controls_locked)
        self.playground_preview_on_done.setEnabled(has_file and not controls_locked)
        self.playground_also_prod_timing.setEnabled(has_file and not controls_locked)
        self.playground_extended_report.setEnabled(has_file and not controls_locked)
        self.playground_also_prod_timing.setToolTip(
            "When enabled and current timing is not "
            f"{compare_window_ms}/{compare_step_ms},\n"
            "generate an additional report using legacy comparison timing."
        )
        self.playground_analyze_button.setText("Analyze Batch" if batch_mode else "Analyze File")
        self.playground_analyze_button.setToolTip(
            "Run offline detector analysis on loaded WAV files and save compact report(s)."
            if batch_mode
            else "Run offline detector analysis on the loaded WAV file and save a compact report."
        )
        self.playground_analyze_button.setEnabled(has_file and not controls_locked)
        self.playground_preview_button.setEnabled(
            has_file and (not batch_mode) and SOUNDDEVICE_AVAILABLE and not controls_locked
        )
        self.playground_add_marker_button.setEnabled(
            has_file and (not batch_mode) and not controls_locked and self._playground_is_playing
        )
        self.playground_clear_markers_button.setEnabled(
            has_file and (not batch_mode) and not controls_locked and len(self._playground_current_markers()) > 0
        )
        self.playground_preview_button.setText("Stop Preview" if self._playground_is_playing else "Preview Sound")
        selected_device = self.selected_combo_device()
        can_start_live = bool(
            SOUNDDEVICE_AVAILABLE
            and not controls_locked
            and selected_device
            and selected_device.is_monitorable
        )
        self.playground_live_start_button.setEnabled(can_start_live)
        self.playground_live_stop_button.setEnabled(SOUNDDEVICE_AVAILABLE and live_running)
        if not SOUNDDEVICE_AVAILABLE:
            self.playground_playback_status.setText("Playback unavailable (sounddevice missing)")
            self.playground_live_status.setText("Live report unavailable (sounddevice missing)")
            self.playground_preview_button.setText("Preview Sound")
            self._reset_playground_preview_meters()
        self.playground_peak_meter.setEnabled(has_file)
        self.update_playground_marker_status()

    def _smooth_playground_levels(self, peak_dbfs: float, rms_dbfs: float) -> tuple[float, float]:
        if not self.settings.smooth_preview_meter:
            self._playground_display_peak_dbfs = max(-120.0, min(0.0, float(peak_dbfs)))
            self._playground_display_rms_dbfs = max(-120.0, min(0.0, float(rms_dbfs)))
            self._playground_display_last_update_at = time.monotonic()
            return self._playground_display_peak_dbfs, self._playground_display_rms_dbfs

        now = time.monotonic()
        dt = max(0.001, now - self._playground_display_last_update_at)
        self._playground_display_last_update_at = now

        attack_tau = 0.07
        release_tau = 0.24

        def step(current: float, target: float) -> float:
            tau = attack_tau if target > current else release_tau
            alpha = 1.0 - math.exp(-dt / max(0.001, tau))
            value = current + (target - current) * alpha
            return max(-120.0, min(0.0, value))

        self._playground_display_peak_dbfs = step(self._playground_display_peak_dbfs, peak_dbfs)
        self._playground_display_rms_dbfs = step(self._playground_display_rms_dbfs, rms_dbfs)
        return self._playground_display_peak_dbfs, self._playground_display_rms_dbfs

    def _set_playground_preview_meters(self, peak_dbfs: float, rms_dbfs: float, *, force: bool = False) -> None:
        peak_clamped = max(-120.0, min(0.0, float(peak_dbfs)))
        rms_clamped = max(-120.0, min(0.0, float(rms_dbfs)))
        if force:
            self._playground_display_peak_dbfs = peak_clamped
            self._playground_display_rms_dbfs = rms_clamped
            self._playground_display_last_update_at = time.monotonic()
            display_peak = peak_clamped
        else:
            display_peak, _ = self._smooth_playground_levels(peak_clamped, rms_clamped)
        self.playground_peak_meter.set_level_dbfs(display_peak, peak_source=True)

    def _reset_playground_preview_meters(self) -> None:
        self._set_playground_preview_meters(-120.0, -120.0, force=True)

    def _update_playground_preview_meters_from_position(self, current_ms: int) -> None:
        preview_path = self._playground_preview_active_path
        if not preview_path:
            self._reset_playground_preview_meters()
            return
        loaded = self._playground_find_loaded_by_path(preview_path)
        if loaded is None or loaded.sample_rate <= 0:
            self._reset_playground_preview_meters()
            return
        channel_idx = max(0, min(int(self._playground_preview_active_channel_index), loaded.channel_count - 1))
        audio = loaded.samples[:, channel_idx]
        if audio.size <= 0:
            self._reset_playground_preview_meters()
            return
        sample_idx = int(round((max(0, current_ms) / 1000.0) * float(loaded.sample_rate)))
        half_window = max(64, int(round(float(loaded.sample_rate) * 0.06)))
        start = max(0, sample_idx - half_window)
        end = min(int(audio.size), sample_idx + half_window)
        if end <= start:
            self._reset_playground_preview_meters()
            return
        segment = audio[start:end]
        rms = float(np.sqrt(np.mean(segment**2)))
        peak = float(np.max(np.abs(segment))) if int(segment.size) > 0 else 0.0
        rms_dbfs = 20.0 * math.log10(rms + 1e-12)
        peak_dbfs = 20.0 * math.log10(peak + 1e-12)
        self._set_playground_preview_meters(peak_dbfs, rms_dbfs)

    def toggle_playground_preview(self) -> None:
        if self._playground_is_playing:
            self.stop_playground_audio()
            return
        self.play_playground_audio()

    def play_playground_audio(self) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            QMessageBox.warning(self, "Playback unavailable", "sounddevice is not available.")
            return
        loaded = self._playground_audio_file
        if loaded is None:
            QMessageBox.information(self, "Playground", "Load a WAV file first.")
            return
        channel_idx = max(0, min(self.playground_channel_spin.value() - 1, loaded.channel_count - 1))
        audio = loaded.samples[:, channel_idx]
        try:
            sd.play(audio, samplerate=loaded.sample_rate, blocking=False)
        except Exception as exc:
            QMessageBox.warning(self, "Playback failed", str(exc))
            return
        self._playground_is_playing = True
        self._playground_preview_active_path = loaded.path
        self._playground_preview_active_channel_index = channel_idx
        self._playground_started_at_monotonic = time.monotonic()
        self._playground_play_duration_seconds = max(0.001, len(audio) / float(loaded.sample_rate))
        self._playground_marker_flash_phase = False
        self._sync_playground_progress_for_loaded(loaded)
        self.playground_progress.set_position_ms(0)
        self.playground_progress.set_marker_alert(active_marker_ms=None, flash_on=False)
        self._update_playground_preview_meters_from_position(0)
        self.playground_playback_timer.start(self._preview_timer_interval_ms())
        self.playground_playback_status.setText("Playing")
        self.update_playground_controls()

    def stop_playground_audio(self) -> None:
        if SOUNDDEVICE_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass
        self._playground_is_playing = False
        self._playground_preview_active_path = None
        self._playground_preview_active_channel_index = 0
        self.playground_playback_timer.stop()
        self.playground_playback_status.setText("Not playing")
        self.playground_progress.set_position_ms(0)
        self.playground_progress.set_marker_alert(active_marker_ms=None, flash_on=False)
        self._reset_playground_preview_meters()
        self.update_playground_controls()

    def refresh_playground_playback_status(self) -> None:
        if not self._playground_is_playing:
            return
        elapsed = max(0.0, time.monotonic() - self._playground_started_at_monotonic)
        current_ms = int(round(elapsed * 1000.0))
        self.playground_progress.set_position_ms(current_ms)
        self._update_playground_preview_meters_from_position(current_ms)
        nearest_marker = None
        nearest_distance = 10_000_000
        for marker_ms in self._playground_current_markers():
            distance = abs(current_ms - marker_ms)
            if distance <= 150 and distance < nearest_distance:
                nearest_distance = distance
                nearest_marker = marker_ms
        flash_on = False
        if nearest_marker is not None:
            self._playground_marker_flash_phase = not self._playground_marker_flash_phase
            flash_on = self._playground_marker_flash_phase
        self.playground_progress.set_marker_alert(active_marker_ms=nearest_marker, flash_on=flash_on)
        if elapsed >= self._playground_play_duration_seconds:
            self.stop_playground_audio()

    def run_playground_analysis(self) -> None:
        if self._playground_live_running:
            QMessageBox.information(self, "Playground", "Stop the live report before running file analysis.")
            return
        if self._playground_analysis_running:
            return
        loaded_files = list(self._playground_loaded_files)
        if not loaded_files:
            QMessageBox.information(self, "Playground", "Load a WAV file first.")
            return

        window_ms = int(self.playground_window_ms_spin.value())
        step_ms = int(self.playground_step_ms_spin.value())
        warmup_ms = int(self.playground_warmup_ms_spin.value())
        channel_idx = max(0, self.playground_channel_spin.value() - 1)
        compare_window_ms, compare_step_ms = self._playground_comparison_timing()
        also_prod_timing = bool(self.playground_also_prod_timing.isChecked())
        run_prod_timing = also_prod_timing and (
            window_ms != compare_window_ms or step_ms != compare_step_ms
        )

        batch_mode = len(loaded_files) > 1
        self.playground_analysis_summary.setPlainText(
            "Running batch analysis..." if batch_mode else "Running analysis..."
        )
        self._playground_analysis_running = True
        self.playground_table.setSortingEnabled(False)
        self.update_playground_controls()
        worker_payload = {
            "loaded_files": loaded_files,
            "window_ms": window_ms,
            "step_ms": step_ms,
            "warmup_ms": warmup_ms,
            "channel_idx": channel_idx,
            "run_prod_timing": bool(run_prod_timing),
            "compare_window_ms": compare_window_ms,
            "compare_step_ms": compare_step_ms,
            "baseline_profiles": dict(self._playground_baseline_profiles_by_path),
            "markers_by_path": {k: list(v) for k, v in self._playground_markers_by_path.items()},
        }
        self._playground_analysis_thread = threading.Thread(
            target=self._run_playground_analysis_worker,
            args=(worker_payload,),
            name="playground-analysis-worker",
            daemon=True,
        )
        self._playground_analysis_thread.start()

    def _run_playground_analysis_worker(self, payload: dict[str, object]) -> None:
        try:
            loaded_files = list(payload.get("loaded_files", []))
            window_ms = int(payload.get("window_ms", 1000))
            step_ms = int(payload.get("step_ms", 50))
            warmup_ms = int(payload.get("warmup_ms", 700))
            channel_idx = int(payload.get("channel_idx", 0))
            run_prod_timing = bool(payload.get("run_prod_timing", False))
            compare_window_ms = int(payload.get("compare_window_ms", 2000))
            compare_step_ms = int(payload.get("compare_step_ms", 200))
            baseline_profiles = dict(payload.get("baseline_profiles", {}))
            markers_by_path = dict(payload.get("markers_by_path", {}))

            analysis_rows: list[dict[str, object]] = []
            for loaded in loaded_files:
                if not isinstance(loaded, LoadedWavFile):
                    continue
                effective_channel_idx = max(0, min(channel_idx, loaded.channel_count - 1))
                result = analyze_wav_file(
                    loaded,
                    self.settings,
                    channel_index=effective_channel_idx,
                    window_ms=window_ms,
                    step_ms=step_ms,
                    warmup_ms=warmup_ms,
                    baseline_profile=baseline_profiles.get(loaded.path),
                )
                prod_result = None
                if run_prod_timing:
                    prod_result = analyze_wav_file(
                        loaded,
                        self.settings,
                        channel_index=effective_channel_idx,
                        window_ms=compare_window_ms,
                        step_ms=compare_step_ms,
                        warmup_ms=warmup_ms,
                        baseline_profile=baseline_profiles.get(loaded.path),
                    )
                inferred = self.infer_expected_outcome_from_filename(loaded.path)
                analysis_rows.append(
                    {
                        "loaded": loaded,
                        "result": result,
                        "prod_result": prod_result,
                        "effective_channel_idx": effective_channel_idx,
                        "inferred_expected": inferred[0] if inferred is not None else None,
                        "inferred_source": inferred[1] if inferred is not None else None,
                        "markers_ms": list(markers_by_path.get(loaded.path, [])),
                    }
                )
            self.signals.playground_analysis_done.emit(analysis_rows)
        except Exception as exc:
            self.signals.playground_analysis_failed.emit(str(exc))

    def _on_playground_analysis_failed(self, message: str) -> None:
        self._playground_analysis_running = False
        self._playground_analysis_thread = None
        self.update_playground_controls()
        QMessageBox.warning(self, "Analysis failed", message)

    def _on_playground_analysis_done(self, analysis_rows_obj: object) -> None:
        analysis_rows = list(analysis_rows_obj) if isinstance(analysis_rows_obj, list) else []
        self._playground_analysis_running = False
        self._playground_analysis_thread = None
        self.update_playground_controls()
        if not analysis_rows:
            QMessageBox.warning(self, "Analysis failed", "No analysis rows were produced.")
            return
        if len(analysis_rows) > 1:
            self.finalize_batch_playground_analysis(
                analysis_rows,
                allow_preview=bool(self.playground_preview_on_done.isChecked()),
            )
            return

        row = analysis_rows[0]
        result = row["result"]
        assert result is not None
        inferred_expected = row["inferred_expected"]
        inferred_source = row["inferred_source"]
        if inferred_expected is None:
            expected_glitch = self.prompt_playground_expected_outcome(
                allow_preview=bool(self.playground_preview_on_done.isChecked())
            )
        else:
            expected_glitch = bool(inferred_expected)
            if bool(self.playground_preview_on_done.isChecked()):
                self._play_playground_loaded_file(row["loaded"], int(row["effective_channel_idx"]))
            if inferred_source is not None:
                self.append_console(
                    f"Playground expected outcome auto-set from filename tag: "
                    f"{'glitch' if expected_glitch else 'clean'} ({inferred_source})"
                )

        report_stem = Path(result.file.path).stem
        self.render_playground_result(
            result,
            expected_glitch=expected_glitch,
            report_stem=report_stem,
            source_label="file",
            markers_ms=list(row.get("markers_ms", [])),
            update_ui=True,
        )
        prod_result = row["prod_result"]
        if prod_result is not None:
            prod_report_path, _, _ = self.render_playground_result(
                prod_result,
                expected_glitch=expected_glitch,
                report_stem=report_stem,
                source_label="file-legacy",
                markers_ms=list(row.get("markers_ms", [])),
                update_ui=False,
            )
            self.playground_analysis_summary.setPlainText(
                f"{self.playground_analysis_summary.toPlainText()} | Legacy report: {prod_report_path}"
            )

    def _play_playground_loaded_file(self, loaded: LoadedWavFile, channel_idx: int) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            return
        effective_channel_idx = max(0, min(int(channel_idx), loaded.channel_count - 1))
        audio = loaded.samples[:, effective_channel_idx]
        self.stop_playground_audio()
        try:
            sd.play(audio, samplerate=loaded.sample_rate, blocking=False)
        except Exception:
            return
        self._playground_is_playing = True
        self._playground_preview_active_path = loaded.path
        self._playground_preview_active_channel_index = effective_channel_idx
        self._playground_started_at_monotonic = time.monotonic()
        self._playground_play_duration_seconds = max(0.001, len(audio) / float(loaded.sample_rate))
        self._playground_marker_flash_phase = False
        self._sync_playground_progress_for_loaded(loaded)
        self.playground_progress.set_position_ms(0)
        self.playground_progress.set_marker_alert(active_marker_ms=None, flash_on=False)
        self._update_playground_preview_meters_from_position(0)
        self.playground_playback_timer.start(self._preview_timer_interval_ms())
        self.playground_playback_status.setText("Playing")
        self.update_playground_controls()

    def toggle_playground_loaded_file_preview(self, loaded: LoadedWavFile, channel_idx: int) -> None:
        effective_channel_idx = max(0, min(int(channel_idx), loaded.channel_count - 1))
        if (
            self._playground_is_playing
            and self._playground_preview_active_path == loaded.path
            and self._playground_preview_active_channel_index == effective_channel_idx
        ):
            self.stop_playground_audio()
            return
        self._play_playground_loaded_file(loaded, effective_channel_idx)

    def finalize_batch_playground_analysis(
        self,
        analysis_rows: list[dict[str, object]],
        *,
        allow_preview: bool,
    ) -> None:
        if not analysis_rows:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Finalize Batch Outcomes")
        dialog.setModal(True)
        dialog.resize(980, 520)
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Batch analysis finished. Select expected outcomes for any unlabeled files, "
                "then click Finalize Reports."
            )
        )

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setAlignment(Qt.AlignTop)
        grid.addWidget(QLabel("File"), 0, 0)
        grid.addWidget(QLabel("Expected"), 0, 1)
        grid.addWidget(QLabel("Preview"), 0, 2)

        pending_selectors: list[tuple[str, QComboBox]] = []
        preview_rows: list[tuple[QPushButton, LoadedWavFile, int]] = []
        for idx, row in enumerate(analysis_rows, start=1):
            loaded = row["loaded"]
            assert isinstance(loaded, LoadedWavFile)
            file_name = Path(loaded.path).name
            file_label = QLabel(file_name)
            file_label.setToolTip(loaded.path)
            file_label.setWordWrap(True)
            grid.addWidget(file_label, idx, 0)

            inferred_expected = row["inferred_expected"]
            inferred_source = row["inferred_source"]
            if inferred_expected is None:
                selector = QComboBox()
                selector.addItem("Choose...", None)
                selector.addItem("No glitch detected", False)
                selector.addItem("Glitch detected", True)
                grid.addWidget(selector, idx, 1)
                pending_selectors.append((loaded.path, selector))
            else:
                expected_text = "Glitch detected" if bool(inferred_expected) else "No glitch detected"
                auto_label = QLabel(f"Auto ({inferred_source}): {expected_text}")
                grid.addWidget(auto_label, idx, 1)
                self.append_console(
                    f"Batch expected outcome auto-set from filename tag: "
                    f"{'glitch' if bool(inferred_expected) else 'clean'} "
                    f"({inferred_source}) for {file_name}"
                )

            effective_channel_idx = int(row["effective_channel_idx"])
            preview_button = QPushButton("Preview")
            preview_rows.append((preview_button, loaded, effective_channel_idx))
            grid.addWidget(preview_button, idx, 2)

        grid.setRowStretch(len(analysis_rows) + 1, 1)

        def refresh_preview_buttons() -> None:
            for button, loaded, effective_channel_idx in preview_rows:
                is_active = (
                    self._playground_is_playing
                    and self._playground_preview_active_path == loaded.path
                    and self._playground_preview_active_channel_index == effective_channel_idx
                )
                button.setText("Stop" if is_active else "Preview")

        for preview_button, loaded, effective_channel_idx in preview_rows:
            preview_button.clicked.connect(
                lambda _checked=False, lf=loaded, cidx=effective_channel_idx: (
                    self.toggle_playground_loaded_file_preview(lf, cidx),
                    refresh_preview_buttons(),
                )
            )

        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)

        button_box = QDialogButtonBox(dialog)
        finalize_btn = button_box.addButton("Finalize Reports", QDialogButtonBox.AcceptRole)
        cancel_btn = button_box.addButton(QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        def finalize_clicked() -> None:
            for loaded_path, selector in pending_selectors:
                value = selector.currentData()
                if value is None:
                    QMessageBox.warning(
                        dialog,
                        "Expected outcomes incomplete",
                        f"There are still outcomes to identify. Missing: {Path(loaded_path).name}",
                    )
                    return
            for row in analysis_rows:
                inferred_expected = row["inferred_expected"]
                if inferred_expected is not None:
                    row["expected_glitch"] = bool(inferred_expected)
                    continue
                loaded = row["loaded"]
                assert isinstance(loaded, LoadedWavFile)
                chosen = None
                for loaded_path, selector in pending_selectors:
                    if loaded_path == loaded.path:
                        chosen = selector.currentData()
                        break
                row["expected_glitch"] = bool(chosen)
            dialog.accept()

        finalize_btn.clicked.connect(finalize_clicked)
        cancel_btn.clicked.connect(lambda: dialog.reject())
        dialog.finished.connect(lambda _code: self.stop_playground_audio())
        dialog.finished.connect(lambda _code: refresh_preview_buttons())

        if allow_preview and SOUNDDEVICE_AVAILABLE:
            first = analysis_rows[0]
            loaded = first["loaded"]
            assert isinstance(loaded, LoadedWavFile)
            self._play_playground_loaded_file(loaded, int(first["effective_channel_idx"]))
            refresh_preview_buttons()

        if dialog.exec() != QDialog.Accepted:
            self.playground_analysis_summary.setPlainText("Batch analysis canceled before report finalization.")
            return

        primary_report_paths: list[str] = []
        prod_report_paths: list[str] = []
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        first_rendered = False
        for row in analysis_rows:
            expected_glitch = bool(row.get("expected_glitch", False))
            result = row["result"]
            assert result is not None
            report_stem = Path(result.file.path).stem
            report_path, eval_label, _ = self.render_playground_result(
                result,
                expected_glitch=expected_glitch,
                report_stem=report_stem,
                source_label="batch-file",
                markers_ms=list(row.get("markers_ms", [])),
                update_ui=not first_rendered,
            )
            first_rendered = True
            primary_report_paths.append(str(report_path))
            if eval_label == "TP":
                true_pos += 1
            elif eval_label == "TN":
                true_neg += 1
            elif eval_label == "FP":
                false_pos += 1
            elif eval_label == "FN":
                false_neg += 1

            prod_result = row["prod_result"]
            if prod_result is not None:
                prod_report_path, _, _ = self.render_playground_result(
                    prod_result,
                    expected_glitch=expected_glitch,
                    report_stem=report_stem,
                    source_label="batch-file-legacy",
                    markers_ms=list(row.get("markers_ms", [])),
                    update_ui=False,
                )
                prod_report_paths.append(str(prod_report_path))

        summary = (
            f"Batch complete: {len(analysis_rows)} files | "
            f"TP: {true_pos} TN: {true_neg} FP: {false_pos} FN: {false_neg} | "
            f"Reports: {len(primary_report_paths)}"
        )
        if prod_report_paths:
            summary = f"{summary} | Legacy reports: {len(prod_report_paths)}"
        self.playground_analysis_summary.setPlainText(summary)

    def start_live_playground_report(self) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            QMessageBox.warning(self, "Live report unavailable", "sounddevice is not available.")
            return
        if self._playground_live_running:
            return
        if self._playground_is_playing:
            self.stop_playground_audio()
        selected_device = self.selected_combo_device()
        if selected_device is None or not selected_device.is_monitorable:
            QMessageBox.warning(self, "Live report", "Select a monitorable input/capture device first.")
            return

        self._playground_live_paused_monitoring = bool(self.runtime.is_running)
        self._playground_live_baseline_profile = None
        if self._playground_live_paused_monitoring:
            self._playground_live_baseline_profile = self._snapshot_runtime_baseline_profile()
            self.stop_monitoring()
            self.append_console("Main monitoring paused for playground live recording.")

        channel_index = self.channel_combo.currentData()
        channel_idx = int(channel_index if channel_index is not None else self.settings.selected_channel_index)
        channel_idx = max(0, channel_idx)
        channels = channel_idx + 1
        sample_rate = int(round(float(selected_device.default_samplerate or 44100)))

        with self._playground_live_lock:
            self._playground_live_chunks = []
        self._playground_live_sample_rate = sample_rate
        self._playground_live_channel_index = channel_idx
        self._playground_live_started_at_monotonic = time.monotonic()

        try:
            self._playground_live_stream = sd.InputStream(
                device=selected_device.portaudio_index,
                samplerate=sample_rate,
                channels=channels,
                blocksize=1024,
                callback=self._playground_live_callback,
            )
            self._playground_live_stream.start()
        except Exception as exc:
            self._playground_live_stream = None
            self._resume_main_monitoring_after_live_capture()
            QMessageBox.warning(self, "Live report start failed", str(exc))
            return

        self._playground_live_running = True
        self.playground_live_timer.start(200)
        self.playground_live_status.setText("Recording live...")
        self.append_console(
            f"Playground live report started on {selected_device.full_label} "
            f"(channel {channel_idx + 1}, {sample_rate} Hz)"
        )
        self.update_playground_controls()

    def _playground_live_callback(self, indata, frames, time_info, status) -> None:
        if not self._playground_live_running:
            return
        if status:
            return
        try:
            if indata.ndim > 1:
                idx = min(self._playground_live_channel_index, indata.shape[1] - 1)
                audio_data = indata[:, idx]
            else:
                audio_data = indata
            chunk = np.array(audio_data, dtype=np.float32, copy=True)
            with self._playground_live_lock:
                self._playground_live_chunks.append(chunk)
        except Exception:
            return

    def refresh_live_playground_status(self) -> None:
        if not self._playground_live_running:
            return
        elapsed = max(0.0, time.monotonic() - self._playground_live_started_at_monotonic)
        self.playground_live_status.setText(f"Recording live... {elapsed:.1f}s")

    def stop_live_playground_report(self, *, save_report: bool = True, resume_main_monitoring: bool = True) -> None:
        if not self._playground_live_running:
            return
        self._playground_live_running = False
        self.playground_live_timer.stop()
        stream = self._playground_live_stream
        self._playground_live_stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        with self._playground_live_lock:
            chunks = list(self._playground_live_chunks)
            self._playground_live_chunks = []
        self.update_playground_controls()
        self.playground_live_status.setText("Live report idle")

        if not save_report:
            if resume_main_monitoring:
                self._resume_main_monitoring_after_live_capture()
            return

        if not chunks:
            if resume_main_monitoring:
                self._resume_main_monitoring_after_live_capture()
            QMessageBox.information(self, "Live report", "No audio captured.")
            return

        captured = np.concatenate(chunks).astype(np.float32, copy=False)
        if captured.size == 0:
            if resume_main_monitoring:
                self._resume_main_monitoring_after_live_capture()
            QMessageBox.information(self, "Live report", "No audio captured.")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"live_capture_{timestamp}.wav"
            wav_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Live Recording WAV",
                default_name,
                "WAV files (*.wav);;All files (*)",
            )
            if not wav_path:
                self.append_console("Live recording canceled before save.")
                return
            if not str(wav_path).lower().endswith(".wav"):
                wav_path = f"{wav_path}.wav"

            saved_wav = write_mono_wav_file(
                wav_path,
                captured,
                int(self._playground_live_sample_rate or 44100),
            )
            baseline_payload = self._playground_live_baseline_profile
            baseline_sidecar = None
            if isinstance(baseline_payload, dict):
                baseline_sidecar = save_baseline_sidecar(saved_wav, baseline_payload)
            self.append_console(f"Live recording saved: {saved_wav}")
            if baseline_sidecar is not None:
                self.append_console(f"Live baseline profile saved: {baseline_sidecar}")
            self.load_playground_files([str(saved_wav)])
            self.playground_analysis_summary.setPlainText(
                "Live recording saved. Add markers if needed, then run Analyze File."
            )
        except Exception as exc:
            QMessageBox.warning(self, "Live report save failed", str(exc))
        finally:
            self._playground_live_baseline_profile = None
            if resume_main_monitoring:
                self._resume_main_monitoring_after_live_capture()

    def infer_expected_outcome_from_filename(self, path_text: str) -> tuple[bool, str] | None:
        filename = Path(path_text).name
        lowered = filename.lower()

        if re.search(r"\[(?:\s*no[\s_-]*glitch\s*)\]", lowered):
            return False, "[no glitch]"
        if re.search(r"\[(?:\s*continuous[\s_-]*glitch(?:y)?\s*)\]", lowered):
            return True, "[continuous glitchy]"
        if re.search(r"\[(?:\s*glitch(?:y)?\s*)\]", lowered):
            return True, "[glitch]"
        return None

    def resolve_playground_expected_outcome(
        self,
        file_path: str,
        *,
        allow_preview: bool,
    ) -> tuple[bool, str | None]:
        inferred = self.infer_expected_outcome_from_filename(file_path)
        if inferred is not None:
            if allow_preview and SOUNDDEVICE_AVAILABLE and self._playground_audio_file is not None:
                self.stop_playground_audio()
                self.play_playground_audio()
            return inferred[0], inferred[1]
        return self.prompt_playground_expected_outcome(allow_preview=allow_preview), None

    def prompt_playground_expected_outcome(self, *, allow_preview: bool) -> bool:
        started_preview = False
        if allow_preview and SOUNDDEVICE_AVAILABLE and self._playground_audio_file is not None:
            self.stop_playground_audio()
            self.play_playground_audio()
            started_preview = self._playground_is_playing

        box = QMessageBox(self)
        box.setWindowTitle("Expected Outcome")
        box.setIcon(QMessageBox.Question)
        box.setText("Analysis has finished. What is the expected output?")
        no_btn = box.addButton("No glitch detected", QMessageBox.AcceptRole)
        yes_btn = box.addButton("Glitch detected", QMessageBox.AcceptRole)
        box.exec()
        chosen = box.clickedButton()

        if started_preview and self._playground_is_playing:
            self.stop_playground_audio()

        return bool(chosen == yes_btn)

    def render_playground_result(
        self,
        result,
        *,
        expected_glitch: bool,
        report_stem: str,
        source_label: str,
        markers_ms: list[int] | None = None,
        update_ui: bool = True,
    ) -> tuple[Path, str, bool]:
        rows = result.rows
        if update_ui:
            self.playground_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = [
                    str(row.index),
                    str(row.start_ms),
                    str(row.end_ms),
                    f"{row.rms_dbfs:.2f}",
                    f"{row.confidence_pct:.1f}",
                    "Y" if row.high_confidence else "",
                    "Y" if row.primary_hit else "",
                    "Y" if row.deduped_detection else "",
                    "Y" if row.suppressed_by_warmup else "",
                    row.methods,
                    f"{row.silence_ratio:.3f}",
                    str(row.gap_count),
                    f"{row.max_gap_ms:.1f}",
                    f"{row.envelope_score:.3f}",
                    f"{row.modulation_strength:.3f}",
                    f"{row.modulation_freq_hz:.2f}",
                    f"{row.modulation_depth:.3f}",
                    f"{row.modulation_peak_concentration:.3f}",
                    row.reasons,
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    self.playground_table.setItem(row_index, col, item)

            self.playground_table.resizeColumnsToContents()
            self.playground_table.resizeRowsToContents()
            self.update_playground_table_height()
        report_path = write_compact_report(
            result,
            self.settings,
            reports_dir=Path.cwd() / "Reports",
            expected_glitch=expected_glitch,
            report_stem=report_stem,
            extended_report=bool(self.playground_extended_report.isChecked()),
            markers_ms=markers_ms,
            marker_window_ms=int(self.playground_marker_match_ms_spin.value()),
            marker_latency_ms=int(self.playground_marker_latency_ms_spin.value()),
        )
        detected_glitch = result.deduped_detection_count > 0
        if expected_glitch and detected_glitch:
            eval_label = "TP"
        elif (not expected_glitch) and (not detected_glitch):
            eval_label = "TN"
        elif (not expected_glitch) and detected_glitch:
            eval_label = "FP"
        else:
            eval_label = "FN"
        if update_ui:
            self.playground_analysis_summary.setPlainText(
                f"Source: {source_label} | Windows: {len(rows)} | High-conf windows: {result.high_confidence_count} | "
                f"Dedup detections: {result.deduped_detection_count} | "
                f"Warm-up suppressed: {result.warmup_suppressed_count} | "
                f"Max conf: {result.max_confidence_pct:.1f}% | Avg conf: {result.average_confidence_pct:.1f}% | "
                f"Baseline: {result.baseline_source} | Outcome: {eval_label} | Report: {report_path}"
            )
        self.append_console(
            f"Playground {source_label} analysis complete: {result.file.path} "
            f"(channel {result.channel_index + 1}, window={result.window_ms}ms, step={result.step_ms}ms)"
        )
        self.append_console(
            f"Playground expectation check: expected={'glitch' if expected_glitch else 'clean'}, "
            f"detected={'glitch' if detected_glitch else 'clean'} ({eval_label})"
        )
        self.append_console(f"Playground report saved: {report_path}")
        return report_path, eval_label, detected_glitch

    def save_templates(self) -> None:
        templates = collect_templates_from_controls(self)
        errors = templates.validate_all()
        if errors:
            QMessageBox.warning(self, "Template validation failed", "\n".join(errors))
            return
        self.settings.alert_templates = templates
        rebuild_template = self.template_rebuild_response.text().strip()
        self.settings.chat_commands.rebuild_response_template = (
            rebuild_template or "Baseline relearn started."
        )
        save_settings(self.settings)
        self.update_command_service()
        self.append_console("Response templates saved.")

    def preview_templates(self) -> None:
        preview_text, _ = build_preview_text(self)
        self.template_preview.setPlainText(preview_text)

    def reset_template_to_default(self, template_key: str) -> None:
        reset_template_to_default_controller(self, template_key)

    def copy_template_token(self, token: str) -> None:
        QApplication.clipboard().setText(token)
        self.append_console(f"Copied token {token}")

    def save_all_settings(self) -> None:
        save_all_settings_service(self)

    def update_obs_bundle_network_notice(self) -> None:
        update_obs_bundle_network_notice_controller(self)

    def save_obs_settings(self) -> None:
        collect_obs_from_controls_controller(self)
        save_settings(self.settings)
        self.append_console("WebSocket settings saved.")
        self.update_obs_controls_enabled()

    def handle_tab_changed(self, new_index: int) -> None:
        handle_tab_changed_service(self, new_index, self._confirm_save_prompt)

    def _confirm_save_prompt(self, title: str, message: str) -> bool:
        answer = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return bool(answer == QMessageBox.Yes)

    def set_obs_status(self, label: str, color_hex: str) -> None:
        self.obs_status.setText(label)
        self.obs_status.setStyleSheet(f"color: {color_hex}; font-weight: 700;")
        self.obs_badge_presenter.apply(label, color_hex)

    def start_obs_auto_connect(self) -> None:
        start_obs_auto_connect_service(self, max_attempts=OBS_AUTO_CONNECT_MAX_ATTEMPTS)

    def attempt_obs_auto_connect(self) -> None:
        attempt_obs_auto_connect_service(self, max_attempts=OBS_AUTO_CONNECT_MAX_ATTEMPTS)

    def cancel_obs_auto_connect_retry(self) -> None:
        self._obs_auto_connect_retry_timer.stop()
        self._obs_auto_connect_attempt = 0

    def connect_obs(self) -> None:
        connect_obs_service(self)

    def test_obs_connection(self) -> None:
        test_obs_connection_service(self)

    def disconnect_obs(self) -> None:
        disconnect_obs_service(self)

    def refresh_obs_scenes(self) -> None:
        selected_before = self.obs_target_scene.currentText().strip() or self.settings.obs_websocket.target_scene
        watched_before = (
            self.obs_baseline_scene.currentText().strip()
            or self.settings.obs_websocket.baseline_rebuild_scene
            or "Any Scene"
        )
        self.obs_target_scene.blockSignals(True)
        self.obs_baseline_scene.blockSignals(True)
        self.obs_target_scene.clear()
        self.obs_target_scene.addItem("All Scenes")
        self.obs_baseline_scene.clear()
        self.obs_baseline_scene.addItem("Any Scene")
        scenes = self.obs_service.list_scenes()
        if scenes:
            self.obs_target_scene.addItems(scenes)
            self.obs_baseline_scene.addItems(scenes)
            idx = self.obs_target_scene.findText(selected_before)
            self.obs_target_scene.setCurrentIndex(0 if idx < 0 else idx)
            watched_idx = self.obs_baseline_scene.findText(watched_before)
            self.obs_baseline_scene.setCurrentIndex(0 if watched_idx < 0 else watched_idx)
        else:
            if watched_before and watched_before not in {"Any Scene", ""}:
                self.obs_baseline_scene.addItem(watched_before)
                self.obs_baseline_scene.setCurrentIndex(self.obs_baseline_scene.count() - 1)
        self.obs_target_scene.blockSignals(False)
        self.obs_baseline_scene.blockSignals(False)

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
        refresh_obs_source_now_service(self)

    def queue_obs_refresh_request(self, source: str, action: str = "refresh") -> None:
        queue_obs_refresh_request_service(self, source=source, action=action)

    def _show_obs_disabled_message(self) -> None:
        QMessageBox.information(self, "OBS WebSocket Disabled", "Enable OBS WebSocket integration first.")

    def _show_obs_source_required_message(self) -> None:
        QMessageBox.warning(self, "No Source Selected", "Choose an OBS source first.")

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
            self._reset_obs_scene_watch_state()
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
            self.obs_baseline_rebuild_enabled,
            self.obs_baseline_scene,
            self.obs_baseline_min_dwell_sec,
            self.obs_baseline_exit_delay_sec,
            self.obs_baseline_cooldown_sec,
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

    def _reset_obs_scene_watch_state(self) -> None:
        self._obs_scene_watch_last_scene = ""
        self._obs_scene_watch_entered_at = 0.0
        self._obs_scene_watch_pending_rebuild_at = 0.0
        self._obs_scene_watch_pending_from_scene = ""
        self._obs_scene_watch_pending_to_scene = ""
        self._obs_scene_watch_last_rebuild_at = 0.0

    def _schedule_obs_scene_rebuild(
        self,
        *,
        from_scene: str,
        to_scene: str,
        when_monotonic: float,
    ) -> None:
        self._obs_scene_watch_pending_rebuild_at = when_monotonic
        self._obs_scene_watch_pending_from_scene = from_scene
        self._obs_scene_watch_pending_to_scene = to_scene
        delay_seconds = max(0.0, when_monotonic - time.monotonic())
        self.append_console(
            f"OBS scene automation armed: baseline relearn in {delay_seconds:.1f}s "
            f"(exited '{from_scene}' -> '{to_scene}')."
        )

    def check_obs_scene_rebuild_automation(self) -> None:
        settings = self.settings.obs_websocket
        if (
            not settings.enabled
            or not settings.baseline_rebuild_on_scene_exit_enabled
            or not self.obs_service.is_connected
            or not self.runtime.is_running
        ):
            self._reset_obs_scene_watch_state()
            return

        now = time.monotonic()
        current_scene = self.obs_service.current_program_scene().strip()
        if not current_scene:
            return

        if not self._obs_scene_watch_last_scene:
            self._obs_scene_watch_last_scene = current_scene
            self._obs_scene_watch_entered_at = now
        elif current_scene != self._obs_scene_watch_last_scene:
            previous_scene = self._obs_scene_watch_last_scene
            dwell_sec = max(0.0, now - self._obs_scene_watch_entered_at)
            self._obs_scene_watch_last_scene = current_scene
            self._obs_scene_watch_entered_at = now
            watched_scene = settings.baseline_rebuild_scene.strip()
            matches_scene = not watched_scene or watched_scene == previous_scene
            meets_dwell = dwell_sec >= float(settings.baseline_rebuild_min_dwell_sec)
            cooldown_elapsed = now - float(self._obs_scene_watch_last_rebuild_at or 0.0)
            cooldown_ready = (
                self._obs_scene_watch_last_rebuild_at <= 0
                or cooldown_elapsed >= float(settings.baseline_rebuild_cooldown_sec)
            )
            if matches_scene and meets_dwell and cooldown_ready:
                trigger_at = now + float(settings.baseline_rebuild_delay_sec)
                self._schedule_obs_scene_rebuild(
                    from_scene=previous_scene,
                    to_scene=current_scene,
                    when_monotonic=trigger_at,
                )

        pending_at = float(self._obs_scene_watch_pending_rebuild_at or 0.0)
        if pending_at > 0.0 and now >= pending_at:
            from_scene = self._obs_scene_watch_pending_from_scene
            to_scene = self._obs_scene_watch_pending_to_scene
            self._obs_scene_watch_pending_rebuild_at = 0.0
            self._obs_scene_watch_pending_from_scene = ""
            self._obs_scene_watch_pending_to_scene = ""
            ok, message = self.runtime.rebuild_baseline(source="obs_scene_exit")
            if ok:
                self._obs_scene_watch_last_rebuild_at = now
                self.append_console(
                    f"OBS scene automation executed baseline relearn after scene exit "
                    f"('{from_scene}' -> '{to_scene}')."
                )
            else:
                self.append_console(f"OBS scene automation skipped baseline relearn: {message}")

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

    def save_advanced_settings(self) -> None:
        collect_advanced_from_controls_controller(self)
        save_settings(self.settings)
        self.append_console("Advanced settings saved.")

    def reset_advanced_defaults(self) -> None:
        reset_advanced_defaults_controller(self)
        apply_advanced_controls_controller(self)
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
        maybe_trigger_obs_auto_refresh_service(self, glitch_data)

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
        if event_type == "baseline.rebuild_requested":
            source = str(data.get("source") or data.get("reason") or "unknown")
            return f"Baseline relearn requested ({source})."
        if event_type == "baseline.rebuild_failed":
            return f"Baseline relearn failed: {data.get('error')}"
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
        self.obs_scene_watch_timer.stop()
        self.stop_live_playground_report(save_report=False, resume_main_monitoring=False)
        self.stop_playground_audio()
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
