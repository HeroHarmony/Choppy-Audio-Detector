#!/usr/bin/env python3
"""PySide6 GUI for Choppy Audio Detector."""

from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime

try:
    from PySide6.QtCore import QObject, QTimer, Signal, Qt, QRectF
    from PySide6.QtGui import QColor, QPainter
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

from choppy_detector_gui.alert_templates import AlertTemplates
from choppy_detector_gui.file_logging import AppFileLogger
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


class RuntimeSignals(QObject):
    event = Signal(str, object)


class ObsLevelMeter(QWidget):
    """Simple OBS-style horizontal meter with green/yellow/red zones."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dbfs = -120.0
        self._scale_min_db = -60.0
        self._scale_max_db = 0.0
        self._yellow_start_db = -20.0
        self._red_start_db = -9.0
        self.setMinimumHeight(44)
        self.setMaximumHeight(44)

    def set_level_dbfs(self, dbfs: float) -> None:
        self._dbfs = max(-120.0, min(0.0, float(dbfs)))
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

        self._draw_zone_lines(painter, meter_rect)
        self._draw_db_ruler(painter, full_rect, meter_rect)

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
        self._loading_devices = False
        self._last_audio_level_seen_at: datetime | None = None
        self._monitoring_started_at: datetime | None = None
        self._audio_watchdog_warned = False
        self._meter_preview_stream = None
        self._meter_preview_lock = threading.Lock()
        self._meter_preview_dbfs = -120.0
        self._meter_preview_last_update = datetime.min

        self.auto_restart_timer = QTimer(self)
        self.auto_restart_timer.timeout.connect(self.auto_restart)
        self.audio_watchdog_timer = QTimer(self)
        self.audio_watchdog_timer.timeout.connect(self.check_audio_watchdog)
        self.audio_watchdog_timer.start(1000)
        self.meter_preview_timer = QTimer(self)
        self.meter_preview_timer.timeout.connect(self.refresh_meter_preview_ui)
        self.meter_preview_timer.start(80)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.build_main_tab()
        self.build_templates_tab()
        self.build_settings_tab()
        self.build_advanced_tab()
        self.build_console_tab()
        self.refresh_devices()
        self.apply_settings_to_controls()
        self.apply_theme()
        self.update_auto_restart_timer()
        self.update_command_service()
        self.restart_meter_preview()
        self.append_console("GUI started.")
        if self.auto_start_requested:
            QTimer.singleShot(120, self.start_monitoring)

    def apply_launch_options(self, options: argparse.Namespace | None) -> None:
        if options is None:
            return
        if getattr(options, "audio_device", None) is not None:
            self.settings.selected_device_id = int(options.audio_device)
        if getattr(options, "channel", None) is not None:
            self.settings.selected_channel_index = max(0, int(options.channel))
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
        form.addRow("Audio input", self.device_combo)
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

        self.status_label = QLabel("Stopped")
        layout.addWidget(self.status_label)
        self.device_hint_label = QLabel("")
        self.device_hint_label.setWordWrap(True)
        layout.addWidget(self.device_hint_label)

        self.level_meter = ObsLevelMeter()
        self.level_text = QLabel("-inf dBFS")
        layout.addWidget(self.level_meter)
        layout.addWidget(self.level_text)

        layout.addWidget(QLabel("Recent events"))
        self.recent_events = QPlainTextEdit()
        self.recent_events.setReadOnly(True)
        layout.addWidget(self.recent_events, stretch=1)
        self.tabs.addTab(tab, "Main")

    def build_templates_tab(self) -> None:
        tab = QWidget()
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
        layout = QVBoxLayout(tab)
        columns = QHBoxLayout()
        group_title_style = "QGroupBox { font-size: 19px; font-weight: 600; }"

        general_group = QGroupBox("General")
        general_group.setStyleSheet(group_title_style)
        general_form = QFormLayout(general_group)
        self.auto_restart_minutes = QSpinBox()
        self.auto_restart_minutes.setRange(5, 1440)
        general_form.addRow("Auto-restart minutes", self.auto_restart_minutes)

        self.alert_cooldown_ms = QSpinBox()
        self.alert_cooldown_ms.setRange(1000, 3600000)
        self.alert_cooldown_ms.setSingleStep(1000)
        general_form.addRow("Alert cooldown ms", self.alert_cooldown_ms)

        self.logs_enabled = QCheckBox("Write log files")
        general_form.addRow("Logs", self.logs_enabled)
        self.keep_preview_while_monitoring = QCheckBox("Keep Preview Running (Experimental)")
        general_form.addRow("Preview mode", self.keep_preview_while_monitoring)
        self.dark_mode_enabled = QCheckBox("Enable dark mode")
        general_form.addRow("Theme", self.dark_mode_enabled)

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
        self.switch_device_command_prefix = QLineEdit()
        for cmd_input in (
            self.start_command,
            self.stop_command,
            self.restart_command,
            self.status_command,
            self.list_devices_command,
            self.switch_device_command_prefix,
        ):
            cmd_input.setMinimumWidth(250)
        commands_form.addRow("Start command", self.start_command)
        commands_form.addRow("Stop command", self.stop_command)
        commands_form.addRow("Restart command", self.restart_command)
        commands_form.addRow("Status command", self.status_command)
        commands_form.addRow("List devices command", self.list_devices_command)
        commands_form.addRow("Switch device prefix", self.switch_device_command_prefix)

        self.allowed_chat_users = QPlainTextEdit()
        self.allowed_chat_users.setPlaceholderText("One Twitch username per line")
        self.allowed_chat_users.setFixedHeight(120)
        allowed_users_col = QWidget()
        allowed_users_col_layout = QVBoxLayout(allowed_users_col)
        allowed_users_col_layout.setContentsMargins(0, 0, 0, 0)
        allowed_users_col_layout.setSpacing(4)
        allowed_users_col_layout.addWidget(self.allowed_chat_users)
        allowed_users_hint = QLabel("One Twitch username per line.")
        allowed_users_hint.setStyleSheet("color: #bdbdbd;")
        allowed_users_col_layout.addWidget(allowed_users_hint)
        commands_form.addRow("Allowed users", allowed_users_col)

        self.send_command_responses = QCheckBox("Send command responses to Twitch chat")
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
        layout.addLayout(columns, 1)
        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self.save_all_settings)
        layout.addWidget(self.save_settings_button)
        self.tabs.addTab(tab, "Settings")

    def build_console_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        layout.addWidget(self.console_output)
        self.clear_console_button = QPushButton("Clear Console")
        self.clear_console_button.clicked.connect(self.console_output.clear)
        layout.addWidget(self.clear_console_button)
        self.tabs.addTab(tab, "Console")

    def build_advanced_tab(self) -> None:
        tab = QWidget()
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
            row = self._advanced_row(key, desc, value_type, min_v, max_v, step)
            alert_layout.addWidget(row)

        threshold_group = QGroupBox("Thresholds")
        threshold_group.setStyleSheet(group_title_style)
        threshold_layout = QVBoxLayout(threshold_group)
        threshold_layout.setSpacing(8)
        for key, desc, value_type, min_v, max_v, step in self.threshold_schema():
            row = self._advanced_row(key, desc, value_type, min_v, max_v, step)
            threshold_layout.addWidget(row)

        methods_group = QGroupBox("Detection Methods")
        methods_group.setStyleSheet(group_title_style)
        methods_layout = QVBoxLayout(methods_group)
        methods_layout.setSpacing(8)
        for key, desc in self.methods_schema():
            checkbox = QCheckBox()
            self.advanced_widgets[f"method:{key}"] = checkbox
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)

            title = QLabel(key)
            title.setStyleSheet("font-weight: 600;")
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: #bdbdbd;")
            desc_label.setWordWrap(True)
            row_layout.addWidget(title)
            row_layout.addWidget(checkbox)
            row_layout.addWidget(desc_label)
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

    def _advanced_row(self, key: str, desc: str, value_type: str, min_v: float, max_v: float, step: float) -> QWidget:
        row_widget = QWidget()
        row_layout = QVBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)
        title = QLabel(key)
        title.setStyleSheet("font-weight: 600;")
        row_layout.addWidget(title)

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
        row_layout.addWidget(input_widget)

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #bdbdbd;")
        row_layout.addWidget(desc_label)
        return row_widget

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
        self.switch_device_command_prefix.setText(commands.switch_device_command_prefix)
        self.allowed_chat_users.setPlainText("\n".join(commands.allowed_chat_users))
        self.send_command_responses.setChecked(commands.send_command_responses)
        self.logs_enabled.setChecked(self.settings.log_settings.logs_enabled)
        self.keep_preview_while_monitoring.setChecked(self.settings.keep_preview_while_monitoring)
        self.dark_mode_enabled.setChecked(self.settings.dark_mode_enabled)
        self.log_directory.setText(self.settings.log_settings.log_directory)
        self.apply_advanced_to_controls()
        self.refresh_channel_options()

    def apply_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setStyleSheet(DARK_THEME_STYLESHEET if self.settings.dark_mode_enabled else "")

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
        self.status_label.setText("Starting")
        self.runtime.start(source="gui")
        if self.settings.keep_preview_while_monitoring:
            QTimer.singleShot(700, self.ensure_meter_preview_stream)

    def stop_monitoring(self) -> None:
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
        self.status_label.setText("Restarting")
        self.runtime.restart(source="gui")
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
        self.start_button.setEnabled(is_monitorable)
        self.restart_button.setEnabled(is_monitorable)
        if device and not device.is_monitorable:
            self.device_hint_label.setText(
                "Selected device is output-only. On macOS, route output through BlackHole/Loopback and select the matching input/capture endpoint."
            )
        elif device:
            self.device_hint_label.setText("Selected device can be monitored.")
        else:
            self.device_hint_label.setText("No audio device selected.")

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
        commands.switch_device_command_prefix = self.switch_device_command_prefix.text().strip()
        commands.allowed_chat_users = [
            line.strip().lower()
            for line in self.allowed_chat_users.toPlainText().splitlines()
            if line.strip()
        ]
        commands.send_command_responses = self.send_command_responses.isChecked()
        self.settings.log_settings.logs_enabled = self.logs_enabled.isChecked()
        self.settings.keep_preview_while_monitoring = self.keep_preview_while_monitoring.isChecked()
        self.settings.dark_mode_enabled = self.dark_mode_enabled.isChecked()
        self.settings.log_settings.log_directory = self.log_directory.text().strip()
        self.collect_advanced_from_controls()
        self.file_logger.settings = self.settings.log_settings
        save_settings(self.settings)
        self.apply_theme()
        self.update_auto_restart_timer()
        self.update_command_service()
        self.restart_meter_preview()
        self.update_meter_display_mode()
        self.append_console("Settings saved.")

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
        if event_type == "audio.level":
            self._last_audio_level_seen_at = datetime.now()
            self._audio_watchdog_warned = False
            if self.runtime.is_running and not self.settings.keep_preview_while_monitoring:
                return
            rms = float(data.get("rms", 0.0))
            dbfs = float(data.get("dbfs", -120.0))
            self.level_meter.set_level_dbfs(dbfs)
            self.level_text.setText(f"{dbfs:.1f} dBFS | RMS={rms:.6f}")
            return

        if event_type == "runtime.console":
            line = str(data.get("line", "")).rstrip()
            if line:
                self.append_console(line)
                self.append_event(line)
            return

        if event_type in {"monitoring.started", "audio.stream_opened"}:
            self.status_label.setText(f"Running - {self.runtime.device_summary()}")
            self.update_meter_display_mode()
        elif event_type in {"monitoring.stopped", "monitoring.stopped_by_request"}:
            self.status_label.setText("Stopped")
            self._audio_watchdog_warned = False
            self._monitoring_started_at = None
            self.restart_meter_preview()
            self.update_meter_display_mode()
        elif event_type in {"monitoring.error", "audio.stream_error", "detector.error"}:
            self.status_label.setText("Error")

        message = self.format_event(event_type, data)
        self.append_event(message)
        self.append_console(message)

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
        self.recent_events.appendPlainText(self.timestamped(message))

    def append_console(self, message: str) -> None:
        self.console_output.appendPlainText(self.timestamped(message))

    def timestamped(self, message: str) -> str:
        return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"

    def closeEvent(self, event) -> None:
        self.stop_meter_preview()
        self.command_service.stop()
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
            dbfs = 20.0 * np.log10(rms + 1e-12)
            with self._meter_preview_lock:
                self._meter_preview_dbfs = max(-120.0, min(0.0, dbfs))
                self._meter_preview_last_update = datetime.now()
        except Exception:
            return

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
            dbfs = self._meter_preview_dbfs
            seen_at = self._meter_preview_last_update
        if seen_at == datetime.min:
            return
        self.update_meter_display_mode()
        self.level_meter.set_level_dbfs(dbfs)
        suffix = "preview + monitoring" if self.runtime.is_running else "preview"
        self.level_text.setText(f"{dbfs:.1f} dBFS | {suffix}")

    def update_meter_display_mode(self) -> None:
        preview_disabled = self.runtime.is_running and not self.settings.keep_preview_while_monitoring
        self.level_meter.setEnabled(not preview_disabled)
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
    window = MainWindow(launch_options=args)
    window.show()
    return app.exec()


def parse_gui_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choppy Audio Detector GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_gui.py --audio-device 2
  python app_gui.py --audio-device 2 --channel 1
  python app_gui.py --no-twitch
  python app_gui.py --list-devices
        """,
    )
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--audio-device", type=int, metavar="N", help="Select audio input device by GUI index")
    parser.add_argument("--channel", type=int, metavar="N", help="Select input channel index and auto-start")
    parser.add_argument("--twitch", action="store_true", help="Enable Twitch chat alerts")
    parser.add_argument("--no-twitch", action="store_true", help="Disable Twitch chat alerts")
    parser.add_argument("--twitch-channel", type=str, help="Twitch channel name (without #)")
    parser.add_argument("--twitch-bot-username", type=str, help="Twitch bot username")
    parser.add_argument("--twitch-oauth-token", type=str, help="Twitch OAuth token")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
