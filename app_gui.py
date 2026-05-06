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
    from PySide6.QtCore import QObject, QTimer, Signal, Qt, QRectF
    from PySide6.QtGui import QColor, QPainter, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFrame,
        QDoubleSpinBox,
        QFormLayout,
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
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "PySide6 is required for the GUI. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from choppy_detector_gui.alert_templates import AlertTemplates, severity_for_detection_count
from choppy_detector_gui.file_logging import AppFileLogger
from choppy_detector_gui.obs_websocket_service import ObsConnectionConfig, ObsWebSocketService
from choppy_detector_gui.runtime import DetectorRuntime
from choppy_detector_gui.settings import (
    AppSettings,
    DEFAULT_ALERT_CONFIG,
    DEFAULT_APPROACHES,
    DEFAULT_THRESHOLDS,
    load_settings,
    save_settings,
)
from choppy_detector_gui.twitch_command_service import TwitchCommandService

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


class ObsLevelMeter(QWidget):
    """Simple OBS-style horizontal meter with green/yellow/red zones."""

    def __init__(self, show_ruler: bool = True, overlay_label: str = "", parent=None):
        super().__init__(parent)
        self._show_ruler = show_ruler
        self._overlay_label = overlay_label
        self._dbfs = -120.0
        self._peak_hold_dbfs = -120.0
        self._peak_hold_seen_at = 0.0
        self._peak_hold_seconds = 1.2
        self._peak_hold_decay_db_per_sec = 18.0
        self._scale_min_db = -60.0
        self._scale_max_db = 0.0
        self._yellow_start_db = -20.0
        self._red_start_db = -9.0
        if self._show_ruler:
            self.setMinimumHeight(44)
            self.setMaximumHeight(44)
        else:
            self.setMinimumHeight(30)
            self.setMaximumHeight(30)

    def set_level_dbfs(self, dbfs: float, peak_source: bool = False) -> None:
        self._dbfs = max(-120.0, min(0.0, float(dbfs)))
        if peak_source:
            now = time.monotonic()
            if self._dbfs >= self._peak_hold_dbfs:
                self._peak_hold_dbfs = self._dbfs
                self._peak_hold_seen_at = now
            else:
                elapsed = now - self._peak_hold_seen_at
                if elapsed > self._peak_hold_seconds:
                    decay_elapsed = elapsed - self._peak_hold_seconds
                    decayed = self._peak_hold_dbfs - (decay_elapsed * self._peak_hold_decay_db_per_sec)
                    self._peak_hold_dbfs = max(self._dbfs, decayed, self._scale_min_db)
                    if self._peak_hold_dbfs <= self._dbfs + 0.25:
                        self._peak_hold_dbfs = self._dbfs
                        self._peak_hold_seen_at = now
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        full_rect = self.rect().adjusted(0, 2, -1, -2)
        # Keep left/right padding so edge tick labels are not clipped.
        inner_rect = full_rect.adjusted(18, 0, -18, 0)
        meter_rect = QRectF(inner_rect.x(), inner_rect.y(), inner_rect.width(), 20)

        # Background zones: dark green, dark yellow, dark red.
        green_end = meter_rect.width() * self._db_to_ratio(self._yellow_start_db)
        yellow_end = meter_rect.width() * self._db_to_ratio(self._red_start_db)
        self._fill_segment(painter, meter_rect, 0.0, green_end, QColor("#154e1f"))
        self._fill_segment(painter, meter_rect, green_end, yellow_end, QColor("#5c5712"))
        self._fill_segment(painter, meter_rect, yellow_end, meter_rect.width(), QColor("#5a1818"))

        fill_ratio = max(0.0, min(1.0, (self._dbfs - self._scale_min_db) / (self._scale_max_db - self._scale_min_db)))
        fill_width = meter_rect.width() * fill_ratio
        if fill_width <= 0:
            self._draw_zone_lines(painter, meter_rect)
            if self._show_ruler:
                self._draw_db_ruler(painter, full_rect, meter_rect)
            return

        self._fill_segment(painter, meter_rect, 0.0, min(fill_width, green_end), QColor("#28c840"))
        if fill_width > green_end:
            self._fill_segment(painter, meter_rect, green_end, min(fill_width, yellow_end), QColor("#d8d230"))
        if fill_width > yellow_end:
            self._fill_segment(painter, meter_rect, yellow_end, fill_width, QColor("#e05050"))

        # Current level indicator.
        marker_x = meter_rect.x() + fill_width
        painter.setPen(QColor("#f5f5f5"))
        painter.drawLine(int(marker_x), int(meter_rect.y()), int(marker_x), int(meter_rect.y() + meter_rect.height()))

        if self._peak_hold_dbfs > self._scale_min_db:
            hold_ratio = self._db_to_ratio(self._peak_hold_dbfs)
            hold_x = meter_rect.x() + meter_rect.width() * hold_ratio
            painter.setPen(QColor("#ffffff"))
            painter.drawLine(int(hold_x), int(meter_rect.y() - 1), int(hold_x), int(meter_rect.y() + meter_rect.height() + 1))

        self._draw_zone_lines(painter, meter_rect)
        if self._show_ruler:
            self._draw_db_ruler(painter, full_rect, meter_rect)
        if self._overlay_label:
            painter.setPen(QColor("#000000"))
            painter.drawText(int(meter_rect.x() + 6), int(meter_rect.y() + 14), self._overlay_label)

    def _fill_segment(self, painter: QPainter, rect, start: float, end: float, color: QColor) -> None:
        width = max(0.0, end - start)
        if width <= 0:
            return
        painter.fillRect(QRectF(rect.x() + start, rect.y(), width, rect.height()), color)

    def _draw_zone_lines(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QColor("#d0d0d0"))
        for ratio in (self._db_to_ratio(self._yellow_start_db), self._db_to_ratio(self._red_start_db)):
            x = rect.x() + rect.width() * ratio
            painter.drawLine(int(x), rect.y(), int(x), rect.y() + rect.height())

    def _draw_db_ruler(self, painter: QPainter, full_rect: QRectF, meter_rect: QRectF) -> None:
        tick_values = tuple(range(-60, 1, 6))
        tick_top = meter_rect.y() + meter_rect.height() + 3
        tick_bottom = tick_top + 5
        label_y = tick_bottom + 12
        painter.setPen(QColor("#bdbdbd"))
        for db_value in tick_values:
            ratio = self._db_to_ratio(float(db_value))
            x = meter_rect.x() + meter_rect.width() * ratio
            painter.drawLine(int(x), int(tick_top), int(x), int(tick_bottom))
            label = str(db_value)
            metrics = painter.fontMetrics()
            label_w = metrics.horizontalAdvance(label)
            painter.drawText(int(x - label_w / 2), int(label_y), label)

    def _db_to_ratio(self, db_value: float) -> float:
        return max(0.0, min(1.0, (db_value - self._scale_min_db) / (self._scale_max_db - self._scale_min_db)))


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
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        form = QFormLayout()
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self.device_selection_changed)
        device_col = QWidget()
        device_col_layout = QVBoxLayout(device_col)
        device_col_layout.setContentsMargins(0, 0, 0, 0)
        device_col_layout.setSpacing(4)
        device_col_layout.addWidget(self.device_combo)
        self.device_hint_label = QLabel("")
        self.device_hint_label.setWordWrap(True)
        self.device_hint_label.setStyleSheet("color: #bdbdbd;")
        self.device_hint_label.hide()
        device_col_layout.addWidget(self.device_hint_label)
        form.addRow("Audio input", device_col)
        self.channel_combo = QComboBox()
        self.channel_combo.currentIndexChanged.connect(self.channel_selection_changed)
        form.addRow("Channel", self.channel_combo)
        top_row.addLayout(form, 3)

        toggle_col = QVBoxLayout()
        self.twitch_enabled = QCheckBox("Enable Twitch alerts")
        self.twitch_enabled.setChecked(self.settings.twitch_enabled)
        self.twitch_enabled.stateChanged.connect(self.save_main_settings)
        toggle_col.addWidget(self.twitch_enabled)

        self.chat_commands_enabled = QCheckBox("Enable Twitch chat commands")
        self.chat_commands_enabled.setChecked(self.settings.chat_commands.chat_commands_enabled)
        self.chat_commands_enabled.stateChanged.connect(self.save_main_settings)
        toggle_col.addWidget(self.chat_commands_enabled)
        toggle_col.addStretch(1)
        top_row.addLayout(toggle_col, 2)
        layout.addLayout(top_row)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart_monitoring)
        self.refresh_button = QPushButton("Refresh Devices")
        self.refresh_button.clicked.connect(self.refresh_devices)
        for button in (self.start_button, self.stop_button, self.restart_button, self.refresh_button):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Stopped")
        status_row.addWidget(self.status_label, 1)

        status_right_col = QVBoxLayout()
        status_right_col.setContentsMargins(0, 0, 0, 0)
        status_right_col.addStretch(1)
        self.level_text = QLabel("Peak -inf dBFS | RMS -inf dBFS")
        status_right_col.addWidget(self.level_text, 0, Qt.AlignRight)
        status_row.addLayout(status_right_col, 0)
        layout.addLayout(status_row)

        meter_stack = QWidget()
        meter_stack_layout = QVBoxLayout(meter_stack)
        meter_stack_layout.setContentsMargins(0, 0, 0, 0)
        meter_stack_layout.setSpacing(0)
        self.peak_meter = ObsLevelMeter(show_ruler=False, overlay_label="Peak")
        self.rms_meter = ObsLevelMeter(show_ruler=True, overlay_label="RMS")
        meter_stack_layout.addWidget(self.peak_meter)
        meter_stack_layout.addWidget(self.rms_meter)
        layout.addWidget(meter_stack)

        layout.addWidget(QLabel("Recent events"))
        self.recent_events = QPlainTextEdit()
        self.recent_events.setReadOnly(True)
        layout.addWidget(self.recent_events, stretch=1)
        self.tabs.addTab(tab, "Main")

    def build_templates_tab(self) -> None:
        tab = QWidget()
        self.templates_tab = tab
        layout = QVBoxLayout(tab)
        self.template_first_minor = QTextEdit()
        self.template_first_moderate = QTextEdit()
        self.template_first_severe = QTextEdit()
        self.template_ongoing = QTextEdit()
        for editor in (
            self.template_first_minor,
            self.template_first_moderate,
            self.template_first_severe,
            self.template_ongoing,
        ):
            editor.setFixedHeight(76)

        content_row = QHBoxLayout()
        left_col = QVBoxLayout()
        form = QFormLayout()
        form.addRow("First minor", self._template_row(self.template_first_minor, "first_minor"))
        form.addRow("First moderate", self._template_row(self.template_first_moderate, "first_moderate"))
        form.addRow("First severe", self._template_row(self.template_first_severe, "first_severe"))
        form.addRow("Ongoing", self._template_row(self.template_ongoing, "ongoing"))
        left_col.addLayout(form)

        button_row = QHBoxLayout()
        self.preview_templates_button = QPushButton("Preview")
        self.preview_templates_button.clicked.connect(self.preview_templates)
        self.save_templates_button = QPushButton("Save Templates")
        self.save_templates_button.clicked.connect(self.save_templates)
        button_row.addWidget(self.preview_templates_button)
        button_row.addWidget(self.save_templates_button)
        left_col.addLayout(button_row)

        self.template_preview = QPlainTextEdit()
        self.template_preview.setReadOnly(True)
        left_col.addWidget(self.template_preview, 1)

        guide_col = QVBoxLayout()
        guide_col.addWidget(QLabel("Template variables"))
        guide_col.addWidget(self._template_reference_guide(), 1)

        content_row.addLayout(left_col, 5)
        content_row.addLayout(guide_col, 2)
        layout.addLayout(content_row, 1)
        self.tabs.addTab(tab, "Responses")

    def _template_reference_guide(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)

        entries = (
            ("{severity}", "Alert severity label such as [MINOR], [MODERATE], [SEVERE]."),
            ("{detection_count}", "How many detections were counted in the alert window."),
            ("{time_span_minutes}", "Window length in minutes, for example 1.5."),
            ("{confidence_threshold}", "Configured confidence threshold percentage."),
            ("{device_name}", "Active device name used by monitoring."),
            ("{timestamp}", "Local timestamp at render time."),
        )
        for token, description in entries:
            panel_layout.addWidget(self._reference_token_row(token, description))
        panel_layout.addStretch(1)
        return panel

    def _reference_token_row(self, token: str, description: str) -> QWidget:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)

        top = QHBoxLayout()
        token_label = QLabel(token)
        token_label.setStyleSheet("font-family: Menlo, Monaco, Courier New, monospace; font-weight: 600;")
        copy_btn = QToolButton()
        copy_btn.setText("Copy")
        copy_btn.clicked.connect(lambda: self.copy_template_token(token))
        top.addWidget(token_label)
        top.addStretch(1)
        top.addWidget(copy_btn)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #bdbdbd;")

        row_layout.addLayout(top)
        row_layout.addWidget(desc_label)
        return row

    def _template_row(self, editor: QTextEdit, template_key: str) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(editor, 1)
        reset_button = QToolButton()
        reset_button.setText("Default")
        reset_button.clicked.connect(lambda: self.reset_template_to_default(template_key))
        row_layout.addWidget(reset_button, 0, Qt.AlignTop)
        return row_widget

    def build_settings_tab(self) -> None:
        tab = QWidget()
        self.settings_tab = tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)
        columns = QHBoxLayout()
        group_title_style = "QGroupBox { font-size: 19px; font-weight: 600; }"

        general_group = QGroupBox("General")
        general_group.setStyleSheet(group_title_style)
        general_form = QFormLayout(general_group)
        self.auto_restart_minutes = QSpinBox()
        self.auto_restart_minutes.setRange(5, 1440)
        general_form.addRow("Auto-restart minutes", self.auto_restart_minutes)
        self.auto_start_monitoring = QCheckBox("Start monitoring on app start")
        general_form.addRow("Auto-start monitoring", self.auto_start_monitoring)
        self.obs_auto_connect_on_launch = QCheckBox("Start websocket on app start")
        general_form.addRow("Auto-connect OBS", self.obs_auto_connect_on_launch)
        self.obs_auto_connect_retry_enabled = QCheckBox("5 retries, 60 sec cooldown")
        general_form.addRow("OBS auto-connect retry", self.obs_auto_connect_retry_enabled)

        self.alert_cooldown_ms = QSpinBox()
        self.alert_cooldown_ms.setRange(1000, 3600000)
        self.alert_cooldown_ms.setSingleStep(1000)
        general_form.addRow("Alert cooldown ms", self.alert_cooldown_ms)

        self.logs_enabled = QCheckBox("Write log files")
        general_form.addRow("Logs", self.logs_enabled)
        self.log_window_retention_minutes = QSpinBox()
        self.log_window_retention_minutes.setRange(1, 10080)
        self.log_window_retention_minutes.setSuffix(" min")
        general_form.addRow("Log window retention", self.log_window_retention_minutes)
        self.keep_preview_while_monitoring = QCheckBox("Keep Preview Running")
        general_form.addRow("Preview mode", self.keep_preview_while_monitoring)
        self.dark_mode_enabled = QCheckBox("Enable dark mode")
        general_form.addRow("Theme", self.dark_mode_enabled)
        self.smooth_preview_meter = QCheckBox("Smooth preview animation")
        general_form.addRow("Meter smoothing", self.smooth_preview_meter)
        self.preview_meter_fps = QSpinBox()
        self.preview_meter_fps.setRange(5, 60)
        general_form.addRow("Preview meter FPS", self.preview_meter_fps)

        self.log_directory = QLineEdit()
        self.log_directory.setPlaceholderText("Leave blank for ./Log")
        self.log_directory.setMinimumWidth(220)
        general_form.addRow("Log directory", self.log_directory)

        twitch_group = QGroupBox("Twitch Connection")
        twitch_group.setStyleSheet(group_title_style)
        twitch_layout = QVBoxLayout(twitch_group)
        twitch_layout.setSpacing(8)
        self.twitch_channel = QLineEdit()
        self.twitch_channel.setPlaceholderText("Channel name, without #")
        self.twitch_channel.setMinimumWidth(250)
        twitch_layout.addWidget(QLabel("Twitch channel"))
        twitch_layout.addWidget(self.twitch_channel)

        self.twitch_bot_username = QLineEdit()
        self.twitch_bot_username.setPlaceholderText("Bot username")
        self.twitch_bot_username.setMinimumWidth(250)
        twitch_layout.addWidget(QLabel("Twitch bot username"))
        twitch_layout.addWidget(self.twitch_bot_username)

        self.twitch_oauth_token = QLineEdit()
        self.twitch_oauth_token.setPlaceholderText("oauth:...")
        self.twitch_oauth_token.setEchoMode(QLineEdit.Password)
        self.twitch_oauth_token.setMinimumWidth(250)
        twitch_layout.addWidget(QLabel("Twitch OAuth token"))
        twitch_layout.addWidget(self.twitch_oauth_token)

        commands_group = QGroupBox("Chat Commands and Permissions")
        commands_group.setStyleSheet(group_title_style)
        commands_form = QFormLayout(commands_group)
        self.start_command = QLineEdit()
        self.stop_command = QLineEdit()
        self.restart_command = QLineEdit()
        self.status_command = QLineEdit()
        self.list_devices_command = QLineEdit()
        self.fix_command = QLineEdit()
        self.switch_device_command_prefix = QLineEdit()
        for cmd_input in (
            self.start_command,
            self.stop_command,
            self.restart_command,
            self.status_command,
            self.list_devices_command,
            self.fix_command,
            self.switch_device_command_prefix,
        ):
            cmd_input.setMinimumWidth(150)
        commands_form.addRow("Start command", self.start_command)
        commands_form.addRow("Stop command", self.stop_command)
        commands_form.addRow("Restart command", self.restart_command)
        commands_form.addRow("Status command", self.status_command)
        commands_form.addRow("List devices command", self.list_devices_command)
        commands_form.addRow("Refresh OBS Source", self.fix_command)
        commands_form.addRow("Switch device prefix", self.switch_device_command_prefix)

        self.allowed_chat_users = QPlainTextEdit()
        self.allowed_chat_users.setPlaceholderText("One Twitch username per line")
        self.allowed_chat_users.setFixedHeight(120)
        self.allowed_chat_users.setMinimumWidth(0)
        allowed_users_col = QWidget()
        allowed_users_col_layout = QVBoxLayout(allowed_users_col)
        allowed_users_col_layout.setContentsMargins(0, 0, 0, 0)
        allowed_users_col_layout.setSpacing(4)
        allowed_users_col_layout.addWidget(self.allowed_chat_users)
        allowed_users_hint = QLabel("One Twitch username per line.")
        allowed_users_hint.setStyleSheet("color: #bdbdbd;")
        allowed_users_col_layout.addWidget(allowed_users_hint)
        commands_form.addRow("Allowed users", allowed_users_col)

        self.send_command_responses = QCheckBox("Send to chat")
        commands_form.addRow("Command responses", self.send_command_responses)

        left_col = QVBoxLayout()
        left_col.addWidget(general_group)
        left_col.addWidget(twitch_group)
        left_col.addStretch(1)

        right_col = QVBoxLayout()
        right_col.addWidget(commands_group)
        right_col.addStretch(1)

        columns.addLayout(left_col, 1)
        columns.addLayout(right_col, 1)
        content_layout.addLayout(columns, 1)
        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self.save_all_settings)
        content_layout.addWidget(self.save_settings_button)
        self.tabs.addTab(tab, "Settings")

    def build_support_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        title = QLabel("Support and Links")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        github_link = QLabel('<a href="https://github.com/HeroHarmony/Choppy-Audio-Detector">GitHub Repository</a>')
        github_link.setOpenExternalLinks(True)
        layout.addWidget(github_link)

        twitch_link = QLabel('<a href="https://www.twitch.tv/heroharmony">HeroHarmony on Twitch</a>')
        twitch_link.setOpenExternalLinks(True)
        layout.addWidget(twitch_link)

        support_link = QLabel('<a href="https://streamelements.com/heroharmony/tip">Buy Me a Coffee / Support</a>')
        support_link.setOpenExternalLinks(True)
        layout.addWidget(support_link)

        layout.addStretch(1)
        self.tabs.addTab(tab, "Support")

    def build_websocket_tab(self) -> None:
        tab = QWidget()
        self.websocket_tab = tab
        root_layout = QVBoxLayout(tab)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        connection_group = QGroupBox("OBS Connection")
        connection_form = QFormLayout(connection_group)
        connection_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        connection_form.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.obs_enabled = QCheckBox("Enable OBS WebSocket integration")
        self.obs_enabled.stateChanged.connect(self.update_obs_controls_enabled)
        connection_form.addRow("Enabled", self.obs_enabled)
        self.obs_host = QLineEdit()
        self.obs_host.setPlaceholderText("127.0.0.1")
        connection_form.addRow("Host", self.obs_host)
        self.obs_port = QSpinBox()
        self.obs_port.setRange(1, 65535)
        connection_form.addRow("Port", self.obs_port)
        self.obs_password = QLineEdit()
        self.obs_password.setEchoMode(QLineEdit.Password)
        self.obs_password.setPlaceholderText("OBS WebSocket password")
        connection_form.addRow("Password", self.obs_password)
        self.obs_status = QLabel("Disconnected")
        self.obs_status.setStyleSheet("color: #ff9c4a; font-weight: 700;")
        connection_form.addRow("Status", self.obs_status)

        connection_buttons = QHBoxLayout()
        self.obs_connect_button = QPushButton("Connect")
        self.obs_disconnect_button = QPushButton("Disconnect")
        self.obs_test_button = QPushButton("Test OBS Connection")
        self.obs_connect_button.clicked.connect(self.connect_obs)
        self.obs_disconnect_button.clicked.connect(self.disconnect_obs)
        self.obs_test_button.clicked.connect(self.test_obs_connection)
        connection_buttons.addWidget(self.obs_connect_button)
        connection_buttons.addWidget(self.obs_disconnect_button)
        connection_buttons.addWidget(self.obs_test_button)
        connection_form.addRow("", self._wrap_layout(connection_buttons))

        target_group = QGroupBox("Target Source")
        target_form = QFormLayout(target_group)
        target_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        target_form.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.obs_target_scene = QComboBox()
        self.obs_target_scene.setEditable(False)
        self.obs_refresh_scenes_button = QPushButton("Refresh Scenes")
        self.obs_refresh_scenes_button.clicked.connect(self.refresh_obs_scenes)
        scene_row = QHBoxLayout()
        scene_row.addWidget(self.obs_target_scene, 1)
        scene_row.addWidget(self.obs_refresh_scenes_button)
        scene_field = self._wrap_layout(scene_row)
        scene_field.setToolTip("Optional. Leave as 'All Scenes' to search all scenes.")
        target_form.addRow("Scene", scene_field)
        self.obs_target_source = QComboBox()
        self.obs_target_source.setEditable(False)
        self.obs_refresh_sources_button = QPushButton("Refresh Sources")
        self.obs_refresh_sources_button.clicked.connect(self.refresh_obs_sources)
        source_row = QHBoxLayout()
        source_row.addWidget(self.obs_target_source, 1)
        source_row.addWidget(self.obs_refresh_sources_button)
        target_form.addRow("Source", self._wrap_layout(source_row))

        automation_group = QGroupBox("Automation")
        automation_form = QFormLayout(automation_group)
        automation_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        automation_form.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.obs_auto_refresh_enabled = QCheckBox("Auto refresh on detected issues")
        automation_form.addRow("Auto refresh", self.obs_auto_refresh_enabled)
        self.obs_auto_refresh_min_severity = QComboBox()
        self.obs_auto_refresh_min_severity.addItems(["minor", "moderate", "severe"])
        self.obs_auto_refresh_min_severity.setMinimumContentsLength(10)
        self.obs_auto_refresh_min_severity.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.obs_auto_refresh_min_severity.setMinimumWidth(140)
        automation_form.addRow("Min severity", self.obs_auto_refresh_min_severity)
        self.obs_auto_refresh_cooldown_sec = QSpinBox()
        self.obs_auto_refresh_cooldown_sec.setRange(0, 86400)
        self.obs_auto_refresh_cooldown_sec.setSuffix(" sec")
        automation_form.addRow("Cooldown", self.obs_auto_refresh_cooldown_sec)
        self.obs_refresh_off_on_delay_ms = QSpinBox()
        self.obs_refresh_off_on_delay_ms.setRange(0, 10000)
        self.obs_refresh_off_on_delay_ms.setSingleStep(50)
        self.obs_refresh_off_on_delay_ms.setSuffix(" ms")
        automation_form.addRow("Off/on delay", self.obs_refresh_off_on_delay_ms)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.obs_refresh_now_button = QPushButton("Refresh Source Now")
        self.obs_refresh_now_button.clicked.connect(self.refresh_obs_source_now)
        self.obs_save_button = QPushButton("Save WebSocket Settings")
        self.obs_save_button.clicked.connect(self.save_obs_settings)
        actions_layout.addWidget(self.obs_refresh_now_button)
        actions_layout.addWidget(self.obs_save_button)

        if not self.obs_service.available:
            warning = QLabel("obsws-python is not installed. Install dependencies to enable OBS controls.")
            warning.setStyleSheet("color: #f0c04a;")
            layout.addWidget(warning)

        layout.addWidget(connection_group)
        layout.addWidget(target_group)
        layout.addWidget(automation_group)
        layout.addWidget(actions_group)
        layout.addStretch(1)
        self.tabs.addTab(tab, "WebSocket")

    def _wrap_layout(self, inner_layout: QHBoxLayout) -> QWidget:
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(8)
        wrapper = QWidget()
        wrapper.setLayout(inner_layout)
        return wrapper

    def build_console_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        layout.addWidget(self.console_output)
        self.clear_console_button = QPushButton("Clear Console")
        self.clear_console_button.clicked.connect(self.clear_console_messages)
        layout.addWidget(self.clear_console_button)
        self.tabs.addTab(tab, "Console")

    def build_advanced_tab(self) -> None:
        tab = QWidget()
        self.advanced_tab = tab
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)
        group_title_style = "QGroupBox { font-size: 19px; font-weight: 600; }"

        self.advanced_widgets: dict[str, QWidget] = {}
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        alert_group = QGroupBox("Alert Config")
        alert_group.setStyleSheet(group_title_style)
        alert_layout = QVBoxLayout(alert_group)
        alert_layout.setSpacing(8)
        for key, desc, value_type, min_v, max_v, step in self.alert_config_schema():
            row = self._advanced_row(
                key,
                desc,
                value_type,
                min_v,
                max_v,
                step,
                DEFAULT_ALERT_CONFIG.get(key),
            )
            alert_layout.addWidget(row)

        threshold_group = QGroupBox("Thresholds")
        threshold_group.setStyleSheet(group_title_style)
        threshold_layout = QVBoxLayout(threshold_group)
        threshold_layout.setSpacing(8)
        for key, desc, value_type, min_v, max_v, step in self.threshold_schema():
            row = self._advanced_row(
                key,
                desc,
                value_type,
                min_v,
                max_v,
                step,
                DEFAULT_THRESHOLDS.get(key),
            )
            threshold_layout.addWidget(row)

        methods_group = QGroupBox("Detection Methods")
        methods_group.setStyleSheet(group_title_style)
        methods_layout = QVBoxLayout(methods_group)
        methods_layout.setSpacing(8)
        for key, desc in self.methods_schema():
            checkbox = QCheckBox()
            self.advanced_widgets[f"method:{key}"] = checkbox
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
            left_layout.addWidget(checkbox)
            left_layout.addStretch(1)

            default_value = bool(DEFAULT_APPROACHES.get(key, False))
            desc_label = QLabel(f"{desc} (Default: {self._format_default_value(default_value)})")
            desc_label.setStyleSheet("color: #bdbdbd;")
            desc_label.setWordWrap(True)
            row_layout.addWidget(left, 3)
            row_layout.addWidget(desc_label, 5)
            methods_layout.addWidget(row_widget)

        content_layout.addWidget(alert_group)
        content_layout.addSpacing(12)
        content_layout.addWidget(threshold_group)
        content_layout.addSpacing(12)
        content_layout.addWidget(methods_group)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        actions = QHBoxLayout()
        self.reset_advanced_defaults_btn = QPushButton("Reset Defaults")
        self.reset_advanced_defaults_btn.clicked.connect(self.reset_advanced_defaults)
        self.save_advanced_btn = QPushButton("Save Advanced Settings")
        self.save_advanced_btn.clicked.connect(self.save_advanced_settings)
        actions.addWidget(self.reset_advanced_defaults_btn)
        actions.addWidget(self.save_advanced_btn)
        layout.addLayout(actions)
        self.tabs.addTab(tab, "Advanced")

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
        templates = self.settings.alert_templates
        self.template_first_minor.setPlainText(templates.first_minor)
        self.template_first_moderate.setPlainText(templates.first_moderate)
        self.template_first_severe.setPlainText(templates.first_severe)
        self.template_ongoing.setPlainText(templates.ongoing)

        self.auto_restart_minutes.setValue(self.settings.auto_restart_minutes)
        self.auto_start_monitoring.setChecked(self.settings.auto_start_monitoring)
        self.alert_cooldown_ms.setValue(self.settings.alert_cooldown_ms)
        self.twitch_channel.setText(self.settings.twitch_channel)
        self.twitch_bot_username.setText(self.settings.twitch_bot_username)
        self.twitch_oauth_token.setText(self.settings.twitch_oauth_token)
        commands = self.settings.chat_commands
        self.start_command.setText(commands.start_command)
        self.stop_command.setText(commands.stop_command)
        self.restart_command.setText(commands.restart_command)
        self.status_command.setText(commands.status_command)
        self.list_devices_command.setText(commands.list_devices_command)
        self.fix_command.setText(commands.fix_command)
        self.switch_device_command_prefix.setText(commands.switch_device_command_prefix)
        self.allowed_chat_users.setPlainText("\n".join(commands.allowed_chat_users))
        self.send_command_responses.setChecked(commands.send_command_responses)
        self.logs_enabled.setChecked(self.settings.log_settings.logs_enabled)
        self.log_window_retention_minutes.setValue(self.settings.log_settings.log_window_retention_minutes)
        self.keep_preview_while_monitoring.setChecked(self.settings.keep_preview_while_monitoring)
        self.smooth_preview_meter.setChecked(self.settings.smooth_preview_meter)
        self.preview_meter_fps.setValue(self.settings.preview_meter_fps)
        self.dark_mode_enabled.setChecked(self.settings.dark_mode_enabled)
        self.obs_auto_connect_on_launch.setChecked(self.settings.obs_websocket.auto_connect_on_launch)
        self.obs_auto_connect_retry_enabled.setChecked(self.settings.obs_websocket.auto_connect_retry_enabled)
        self.log_directory.setText(self.settings.log_settings.log_directory)
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
        self.global_twitch_status_badge.setText(f"Twitch: {label}")
        self.global_twitch_status_badge.setStyleSheet(
            f"color: {color_hex}; font-weight: 700; border: 1px solid {color_hex};"
            " border-radius: 8px; padding: 2px 8px;"
        )

    def update_twitch_status_from_settings(self) -> None:
        if not self.settings.twitch_enabled and not self.settings.chat_commands.chat_commands_enabled:
            self.set_twitch_status_badge("Disabled", "#8f8f8f")
        else:
            self.set_twitch_status_badge("Idle", "#bdbdbd")

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
        templates = self.collect_templates()
        errors = templates.validate_all()
        if errors:
            self.template_preview.setPlainText("\n".join(errors))
            return
        previews = [
            templates.render(
                detection_count=count,
                time_span_minutes=1.5,
                is_first_alert=is_first,
                confidence_threshold=70,
                device_name=self.runtime.device_summary(),
            )
            for count, is_first in ((6, True), (8, True), (12, True), (9, False))
        ]
        self.template_preview.setPlainText("\n".join(previews))

    def collect_templates(self) -> AlertTemplates:
        return AlertTemplates(
            first_minor=self.template_first_minor.toPlainText().strip(),
            first_moderate=self.template_first_moderate.toPlainText().strip(),
            first_severe=self.template_first_severe.toPlainText().strip(),
            ongoing=self.template_ongoing.toPlainText().strip(),
        )

    def is_templates_dirty(self) -> bool:
        current = self.collect_templates()
        saved = self.settings.alert_templates
        return any(
            (
                current.first_minor != saved.first_minor,
                current.first_moderate != saved.first_moderate,
                current.first_severe != saved.first_severe,
                current.ongoing != saved.ongoing,
            )
        )

    def reset_template_to_default(self, template_key: str) -> None:
        defaults = AlertTemplates()
        if template_key == "first_minor":
            self.template_first_minor.setPlainText(defaults.first_minor)
        elif template_key == "first_moderate":
            self.template_first_moderate.setPlainText(defaults.first_moderate)
        elif template_key == "first_severe":
            self.template_first_severe.setPlainText(defaults.first_severe)
        elif template_key == "ongoing":
            self.template_ongoing.setPlainText(defaults.ongoing)

    def copy_template_token(self, token: str) -> None:
        QApplication.clipboard().setText(token)
        self.append_console(f"Copied token {token}")

    def save_all_settings(self) -> None:
        self.settings.auto_restart_minutes = self.auto_restart_minutes.value()
        self.settings.auto_start_monitoring = self.auto_start_monitoring.isChecked()
        self.settings.alert_cooldown_ms = self.alert_cooldown_ms.value()
        self.settings.twitch_channel = self.twitch_channel.text().strip().lstrip("#")
        self.settings.twitch_bot_username = self.twitch_bot_username.text().strip()
        self.settings.twitch_oauth_token = self.twitch_oauth_token.text().strip()
        commands = self.settings.chat_commands
        commands.chat_commands_enabled = self.chat_commands_enabled.isChecked()
        commands.start_command = self.start_command.text().strip()
        commands.stop_command = self.stop_command.text().strip()
        commands.restart_command = self.restart_command.text().strip()
        commands.status_command = self.status_command.text().strip()
        commands.list_devices_command = self.list_devices_command.text().strip()
        commands.fix_command = self.fix_command.text().strip()
        commands.switch_device_command_prefix = self.switch_device_command_prefix.text().strip()
        commands.allowed_chat_users = [
            line.strip().lower()
            for line in self.allowed_chat_users.toPlainText().splitlines()
            if line.strip()
        ]
        commands.send_command_responses = self.send_command_responses.isChecked()
        self.settings.log_settings.logs_enabled = self.logs_enabled.isChecked()
        self.settings.log_settings.log_window_retention_minutes = self.log_window_retention_minutes.value()
        self.settings.keep_preview_while_monitoring = self.keep_preview_while_monitoring.isChecked()
        self.settings.smooth_preview_meter = self.smooth_preview_meter.isChecked()
        self.settings.preview_meter_fps = self.preview_meter_fps.value()
        self.settings.dark_mode_enabled = self.dark_mode_enabled.isChecked()
        self.settings.obs_websocket.auto_connect_on_launch = self.obs_auto_connect_on_launch.isChecked()
        self.settings.obs_websocket.auto_connect_retry_enabled = self.obs_auto_connect_retry_enabled.isChecked()
        self.settings.log_settings.log_directory = self.log_directory.text().strip()
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
        commands = self.settings.chat_commands
        allowed_users = [
            line.strip().lower()
            for line in self.allowed_chat_users.toPlainText().splitlines()
            if line.strip()
        ]
        return any(
            (
                self.auto_restart_minutes.value() != self.settings.auto_restart_minutes,
                self.auto_start_monitoring.isChecked() != self.settings.auto_start_monitoring,
                self.alert_cooldown_ms.value() != self.settings.alert_cooldown_ms,
                self.twitch_channel.text().strip().lstrip("#") != self.settings.twitch_channel,
                self.twitch_bot_username.text().strip() != self.settings.twitch_bot_username,
                self.twitch_oauth_token.text().strip() != self.settings.twitch_oauth_token,
                self.chat_commands_enabled.isChecked() != commands.chat_commands_enabled,
                self.start_command.text().strip() != commands.start_command,
                self.stop_command.text().strip() != commands.stop_command,
                self.restart_command.text().strip() != commands.restart_command,
                self.status_command.text().strip() != commands.status_command,
                self.list_devices_command.text().strip() != commands.list_devices_command,
                self.fix_command.text().strip() != commands.fix_command,
                self.switch_device_command_prefix.text().strip() != commands.switch_device_command_prefix,
                allowed_users != commands.allowed_chat_users,
                self.send_command_responses.isChecked() != commands.send_command_responses,
                self.logs_enabled.isChecked() != self.settings.log_settings.logs_enabled,
                self.log_window_retention_minutes.value() != self.settings.log_settings.log_window_retention_minutes,
                self.keep_preview_while_monitoring.isChecked() != self.settings.keep_preview_while_monitoring,
                self.smooth_preview_meter.isChecked() != self.settings.smooth_preview_meter,
                self.preview_meter_fps.value() != self.settings.preview_meter_fps,
                self.dark_mode_enabled.isChecked() != self.settings.dark_mode_enabled,
                self.obs_auto_connect_on_launch.isChecked() != self.settings.obs_websocket.auto_connect_on_launch,
                self.obs_auto_connect_retry_enabled.isChecked()
                != self.settings.obs_websocket.auto_connect_retry_enabled,
                self.log_directory.text().strip() != self.settings.log_settings.log_directory,
            )
        )

    def apply_obs_settings_to_controls(self) -> None:
        obs_settings = self.settings.obs_websocket
        self.obs_enabled.setChecked(obs_settings.enabled)
        self.obs_host.setText(obs_settings.host)
        self.obs_port.setValue(obs_settings.port)
        self.obs_password.setText(obs_settings.password)
        self.obs_auto_refresh_enabled.setChecked(obs_settings.auto_refresh_enabled)
        idx = self.obs_auto_refresh_min_severity.findText(obs_settings.auto_refresh_min_severity)
        self.obs_auto_refresh_min_severity.setCurrentIndex(max(0, idx))
        self.obs_auto_refresh_cooldown_sec.setValue(obs_settings.auto_refresh_cooldown_sec)
        self.obs_refresh_off_on_delay_ms.setValue(obs_settings.refresh_off_on_delay_ms)
        self.refresh_obs_scenes()
        if obs_settings.target_scene:
            idx_scene = self.obs_target_scene.findText(obs_settings.target_scene)
            if idx_scene >= 0:
                self.obs_target_scene.setCurrentIndex(idx_scene)
        self.refresh_obs_sources()
        self.update_obs_controls_enabled()

    def collect_obs_from_controls(self) -> None:
        obs_settings = self.settings.obs_websocket
        obs_settings.enabled = self.obs_enabled.isChecked()
        obs_settings.host = self.obs_host.text().strip()
        obs_settings.port = self.obs_port.value()
        obs_settings.password = self.obs_password.text()
        obs_settings.auto_refresh_enabled = self.obs_auto_refresh_enabled.isChecked()
        obs_settings.auto_refresh_min_severity = self.obs_auto_refresh_min_severity.currentText().strip().lower()
        obs_settings.auto_refresh_cooldown_sec = self.obs_auto_refresh_cooldown_sec.value()
        obs_settings.refresh_off_on_delay_ms = self.obs_refresh_off_on_delay_ms.value()
        scene_value = self.obs_target_scene.currentText().strip()
        obs_settings.target_scene = "" if scene_value == "All Scenes" else scene_value
        selected_source = self.obs_target_source.currentText().strip()
        if selected_source:
            obs_settings.target_source = selected_source

    def save_obs_settings(self) -> None:
        self.collect_obs_from_controls()
        save_settings(self.settings)
        self.append_console("WebSocket settings saved.")
        self.update_obs_controls_enabled()

    def is_websocket_dirty(self) -> bool:
        obs_settings = self.settings.obs_websocket
        selected_source = self.obs_target_source.currentText().strip()
        selected_scene = self.obs_target_scene.currentText().strip()
        normalized_scene = "" if selected_scene == "All Scenes" else selected_scene
        return any(
            (
                self.obs_enabled.isChecked() != obs_settings.enabled,
                self.obs_host.text().strip() != obs_settings.host,
                self.obs_port.value() != obs_settings.port,
                self.obs_password.text() != obs_settings.password,
                self.obs_auto_refresh_enabled.isChecked() != obs_settings.auto_refresh_enabled,
                self.obs_auto_refresh_min_severity.currentText().strip().lower() != obs_settings.auto_refresh_min_severity,
                self.obs_auto_refresh_cooldown_sec.value() != obs_settings.auto_refresh_cooldown_sec,
                self.obs_refresh_off_on_delay_ms.value() != obs_settings.refresh_off_on_delay_ms,
                normalized_scene != obs_settings.target_scene,
                selected_source != obs_settings.target_source,
            )
        )

    def is_advanced_dirty(self) -> bool:
        for key, *_ in self.alert_config_schema():
            widget = self.advanced_widgets.get(f"value:{key}")
            if widget is None:
                continue
            current = self._get_advanced_widget_value(widget, DEFAULT_ALERT_CONFIG[key])
            saved = self.settings.advanced_alert_config.get(key, DEFAULT_ALERT_CONFIG[key])
            if current != saved:
                return True
        for key, *_ in self.threshold_schema():
            widget = self.advanced_widgets.get(f"value:{key}")
            if widget is None:
                continue
            current = self._get_advanced_widget_value(widget, DEFAULT_THRESHOLDS[key])
            saved = self.settings.advanced_thresholds.get(key, DEFAULT_THRESHOLDS[key])
            if current != saved:
                return True
        for key, _ in self.methods_schema():
            widget = self.advanced_widgets.get(f"method:{key}")
            if widget is None:
                continue
            if bool(widget.isChecked()) != bool(self.settings.detection_methods.get(key, DEFAULT_APPROACHES[key])):
                return True
        return False

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
        self.global_obs_status_badge.setText(f"OBS: {label}")
        self.global_obs_status_badge.setStyleSheet(
            f"color: {color_hex}; font-weight: 700; border: 1px solid {color_hex};"
            " border-radius: 8px; padding: 2px 8px;"
        )

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
        cfg = ObsConnectionConfig(
            host=self.settings.obs_websocket.host,
            port=self.settings.obs_websocket.port,
            password=self.settings.obs_websocket.password,
        )
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
        cfg = ObsConnectionConfig(
            host=self.settings.obs_websocket.host,
            port=self.settings.obs_websocket.port,
            password=self.settings.obs_websocket.password,
        )
        self.set_obs_status("Connecting", "#4aa3ff")
        self.set_obs_busy(True)
        self._run_obs_task("connect", lambda: self.obs_service.connect(cfg))

    def test_obs_connection(self) -> None:
        if not self.obs_enabled.isChecked():
            QMessageBox.information(self, "OBS WebSocket Disabled", "Enable OBS WebSocket integration first.")
            return
        self.collect_obs_from_controls()
        cfg = ObsConnectionConfig(
            host=self.settings.obs_websocket.host,
            port=self.settings.obs_websocket.port,
            password=self.settings.obs_websocket.password,
        )
        self.set_obs_status("Testing", "#4aa3ff")
        self.set_obs_busy(True)

        def _test_once() -> tuple[bool, str]:
            temp = ObsWebSocketService()
            ok, message = temp.connect(cfg)
            if ok:
                temp.disconnect()
                return True, f"Connection test successful to {cfg.host}:{cfg.port} (test socket closed)."
            return False, message

        self._run_obs_task("test", _test_once)

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

        if action == "connect":
            if ok:
                self.cancel_obs_auto_connect_retry()
                self.set_obs_status("Connected", "#3fcf5e")
                self.append_console(message)
                self.refresh_obs_sources()
            else:
                self.set_obs_status("Error", "#ff6a6a")
                QMessageBox.warning(self, "OBS Connection Failed", message)
                self.append_console(message)
            return

        if action == "connect_auto":
            attempt = int(data.get("attempt") or self._obs_auto_connect_attempt or 1)
            max_attempts = int(data.get("max_attempts") or OBS_AUTO_CONNECT_MAX_ATTEMPTS)
            if ok:
                self.cancel_obs_auto_connect_retry()
                self.set_obs_status("Connected", "#3fcf5e")
                self.append_console(f"OBS auto-connect succeeded on attempt {attempt}/{max_attempts}.")
                self.append_console(message)
                self.refresh_obs_sources()
            else:
                self.set_obs_status("Error", "#ff6a6a")
                self.append_console(f"OBS auto-connect attempt {attempt}/{max_attempts} failed: {message}")
                retry_enabled = bool(self.settings.obs_websocket.auto_connect_retry_enabled)
                if retry_enabled and attempt < max_attempts:
                    retry_seconds = OBS_AUTO_CONNECT_RETRY_DELAY_MS // 1000
                    self.append_console(f"Retrying OBS auto-connect in {retry_seconds} seconds.")
                    self._obs_auto_connect_retry_timer.start(OBS_AUTO_CONNECT_RETRY_DELAY_MS)
                else:
                    if not retry_enabled and attempt == 1:
                        self.append_console("OBS auto-connect retry is disabled; no further attempts will be made.")
                    else:
                        self.append_console(
                            f"OBS auto-connect stopped after {attempt}/{max_attempts} attempts."
                        )
                    self.cancel_obs_auto_connect_retry()
                self.update_obs_controls_enabled()
            return

        if action == "test":
            if ok:
                self.set_obs_status("Test OK", "#3fcf5e")
                QMessageBox.information(self, "OBS Connection Test", message)
                self.append_console(message)
            else:
                self.set_obs_status("Test Failed", "#ff6a6a")
                QMessageBox.warning(self, "OBS Connection Test Failed", message)
                self.append_console(message)
            self.update_obs_controls_enabled()
            return

        if action == "refresh":
            if ok:
                self.set_obs_status("Connected", "#3fcf5e")
                self.append_console(message)
                self.append_event(message)
            else:
                self.set_obs_status("Error", "#ff6a6a")
                QMessageBox.warning(self, "OBS Refresh Failed", message)
                self.append_console(message)
            self.update_obs_controls_enabled()
            return

        if action == "chat_refresh":
            if ok:
                self.set_obs_status("Connected", "#3fcf5e")
                self.append_console(f"OBS chat refresh succeeded: {message}")
                self.append_event(f"OBS chat refresh: {message}")
            else:
                self.set_obs_status("Error", "#ff6a6a")
                self.append_console(f"OBS chat refresh failed: {message}")
                self.append_event(f"OBS chat refresh failed: {message}")
            self.update_obs_controls_enabled()
            return

        if action == "auto_refresh":
            if ok:
                self.set_obs_status("Connected", "#3fcf5e")
                self.append_console(f"OBS auto-refresh succeeded: {message}")
                self.append_event(f"OBS auto-refresh: {message}")
            else:
                self.set_obs_status("Error", "#ff6a6a")
                self.append_console(f"OBS auto-refresh failed: {message}")
                self.append_event(f"OBS auto-refresh failed: {message}")
            self.update_obs_controls_enabled()
            return

    def apply_advanced_to_controls(self) -> None:
        for key, *_ in self.alert_config_schema():
            widget = self.advanced_widgets.get(f"value:{key}")
            if widget is None:
                continue
            value = self.settings.advanced_alert_config.get(key, DEFAULT_ALERT_CONFIG[key])
            self._set_advanced_widget_value(widget, value)
        for key, *_ in self.threshold_schema():
            widget = self.advanced_widgets.get(f"value:{key}")
            if widget is None:
                continue
            value = self.settings.advanced_thresholds.get(key, DEFAULT_THRESHOLDS[key])
            self._set_advanced_widget_value(widget, value)
        for key, _ in self.methods_schema():
            widget = self.advanced_widgets.get(f"method:{key}")
            if widget is None:
                continue
            widget.setChecked(bool(self.settings.detection_methods.get(key, DEFAULT_APPROACHES[key])))

    def collect_advanced_from_controls(self) -> None:
        alert_config = dict(DEFAULT_ALERT_CONFIG)
        for key, *_ in self.alert_config_schema():
            widget = self.advanced_widgets.get(f"value:{key}")
            if widget is None:
                continue
            alert_config[key] = self._get_advanced_widget_value(widget, DEFAULT_ALERT_CONFIG[key])
        thresholds = dict(DEFAULT_THRESHOLDS)
        for key, *_ in self.threshold_schema():
            widget = self.advanced_widgets.get(f"value:{key}")
            if widget is None:
                continue
            thresholds[key] = self._get_advanced_widget_value(widget, DEFAULT_THRESHOLDS[key])
        methods = dict(DEFAULT_APPROACHES)
        for key, _ in self.methods_schema():
            widget = self.advanced_widgets.get(f"method:{key}")
            if widget is None:
                continue
            methods[key] = bool(widget.isChecked())

        self.settings.advanced_alert_config = alert_config
        self.settings.advanced_thresholds = thresholds
        self.settings.detection_methods = methods
        self.settings.alert_cooldown_ms = int(alert_config.get("alert_cooldown_ms", self.settings.alert_cooldown_ms))

    def save_advanced_settings(self) -> None:
        self.collect_advanced_from_controls()
        save_settings(self.settings)
        self.append_console("Advanced settings saved.")

    def reset_advanced_defaults(self) -> None:
        self.settings.advanced_alert_config = dict(DEFAULT_ALERT_CONFIG)
        self.settings.advanced_thresholds = dict(DEFAULT_THRESHOLDS)
        self.settings.detection_methods = dict(DEFAULT_APPROACHES)
        self.settings.alert_cooldown_ms = int(DEFAULT_ALERT_CONFIG["alert_cooldown_ms"])
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
        self.command_service.settings = self.settings
        if self.settings.chat_commands.chat_commands_enabled:
            if self.command_service.running:
                self.command_service.stop()
            self.command_service.start()
        else:
            self.command_service.stop()

    def handle_runtime_event(self, event_type: str, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        if event_type in {"twitch.connected", "chat_commands.connected"}:
            self.set_twitch_status_badge("Connected", "#3fcf5e")
        elif event_type in {"twitch.connecting", "chat_commands.connecting"}:
            self.set_twitch_status_badge("Connecting", "#4aa3ff")
        elif event_type in {"chat_commands.reconnecting", "chat_commands.reconnect_scheduled"}:
            self.set_twitch_status_badge("Reconnecting", "#f0c04a")
        elif event_type in {"twitch.connection_failed", "chat_commands.connection_failed"}:
            error_text = str(data.get("error", "")).lower()
            if "auth" in error_text or "login" in error_text or "token" in error_text:
                self.set_twitch_status_badge("Auth failed", "#ff6a6a")
            else:
                self.set_twitch_status_badge("Disconnected", "#ff9c4a")
        elif event_type == "twitch.send_circuit_open":
            self.set_twitch_status_badge("Paused", "#ff9c4a")
        elif event_type == "twitch.send_resumed":
            self.set_twitch_status_badge("Connected", "#3fcf5e")
        elif event_type == "chat_commands.disconnected":
            self.set_twitch_status_badge("Disconnected", "#ff9c4a")

        if event_type == "chat_commands.fix_requested":
            user = str(data.get("user", "")).strip() or "unknown"
            if not self.obs_enabled.isChecked():
                self.append_console(f"Chat fix by {user} skipped: OBS WebSocket integration disabled.")
                return
            if not self.obs_service.is_connected:
                self.append_console(f"Chat fix by {user} skipped: OBS is not connected.")
                return
            source = self.obs_target_source.currentText().strip() or self.settings.obs_websocket.target_source.strip()
            if not source:
                self.append_console(f"Chat fix by {user} skipped: no OBS source selected.")
                return
            self.append_console(f"Chat fix requested by {user}: refreshing source '{source}'.")
            self.queue_obs_refresh_request(source=source, action="chat_refresh")
            return

        if event_type == "audio.level":
            self._last_audio_level_seen_at = datetime.now()
            self._audio_watchdog_warned = False
            if self.runtime.is_running and not self.settings.keep_preview_while_monitoring:
                return
            peak_dbfs = float(data.get("peak_dbfs", data.get("dbfs", -120.0)))
            rms_dbfs = float(data.get("rms_dbfs", data.get("dbfs", -120.0)))
            smooth_peak_dbfs, smooth_rms_dbfs = self._smooth_display_levels(peak_dbfs, rms_dbfs)
            self.peak_meter.set_level_dbfs(smooth_peak_dbfs, peak_source=True)
            self.rms_meter.set_level_dbfs(smooth_rms_dbfs, peak_source=False)
            self.level_text.setText(f"Peak {peak_dbfs:.1f} dBFS | RMS {rms_dbfs:.1f} dBFS")
            return

        if event_type == "glitch.detected":
            self.maybe_trigger_obs_auto_refresh(data)

        if event_type == "runtime.console":
            line = str(data.get("line", "")).rstrip()
            if line:
                self.append_console(line)
                self.append_event(line)
            return

        if event_type in {"monitoring.started", "audio.stream_opened"}:
            self._monitoring_ui_active = True
            self.status_label.setText(f"Running - {self.runtime.device_summary()}")
            self.update_meter_display_mode()
            self.update_device_controls()
        elif event_type in {"monitoring.stopped", "monitoring.stopped_by_request"}:
            self._monitoring_ui_active = False
            self.status_label.setText("Stopped")
            self._audio_watchdog_warned = False
            self._monitoring_started_at = None
            self.restart_meter_preview()
            self.update_meter_display_mode()
            self.update_device_controls()
        elif event_type in {"monitoring.error", "audio.stream_error", "detector.error"}:
            self._monitoring_ui_active = False
            self.status_label.setText("Error")
            self.update_device_controls()

        message = self.format_event(event_type, data)
        self.append_event(message)
        self.append_console(message)

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
