"""Main tab UI builder."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from choppy_detector_gui.gui.widgets.obs_level_meter import ObsLevelMeter


def build_main_tab(window) -> None:
    tab = QWidget()
    layout = QVBoxLayout(tab)

    top_row = QHBoxLayout()
    form = QFormLayout()
    window.device_combo = QComboBox()
    window.device_combo.currentIndexChanged.connect(window.device_selection_changed)
    device_col = QWidget()
    device_col_layout = QVBoxLayout(device_col)
    device_col_layout.setContentsMargins(0, 0, 0, 0)
    device_col_layout.setSpacing(4)
    device_col_layout.addWidget(window.device_combo)
    window.device_hint_label = QLabel("")
    window.device_hint_label.setWordWrap(True)
    window.device_hint_label.setStyleSheet("color: #bdbdbd;")
    window.device_hint_label.hide()
    device_col_layout.addWidget(window.device_hint_label)
    form.addRow("Audio input", device_col)
    window.channel_combo = QComboBox()
    window.channel_combo.currentIndexChanged.connect(window.channel_selection_changed)
    form.addRow("Channel", window.channel_combo)
    top_row.addLayout(form, 3)

    toggle_col = QVBoxLayout()
    window.twitch_enabled = QCheckBox("Enable Twitch alerts")
    window.twitch_enabled.setChecked(window.settings.twitch_enabled)
    window.twitch_enabled.stateChanged.connect(window.save_main_settings)
    toggle_col.addWidget(window.twitch_enabled)

    window.chat_commands_enabled = QCheckBox("Enable Twitch chat commands")
    window.chat_commands_enabled.setChecked(window.settings.chat_commands.chat_commands_enabled)
    window.chat_commands_enabled.stateChanged.connect(window.save_main_settings)
    toggle_col.addWidget(window.chat_commands_enabled)
    toggle_col.addStretch(1)
    top_row.addLayout(toggle_col, 2)
    layout.addLayout(top_row)

    button_row = QHBoxLayout()
    window.start_button = QPushButton("Start")
    window.start_button.clicked.connect(window.start_monitoring)
    window.stop_button = QPushButton("Stop")
    window.stop_button.clicked.connect(window.stop_monitoring)
    window.restart_button = QPushButton("Restart")
    window.restart_button.clicked.connect(window.restart_monitoring)
    window.refresh_button = QPushButton("Refresh Devices")
    window.refresh_button.clicked.connect(window.refresh_devices)
    for button in (window.start_button, window.stop_button, window.restart_button, window.refresh_button):
        button_row.addWidget(button)
    layout.addLayout(button_row)

    status_row = QHBoxLayout()
    window.status_label = QLabel("Stopped")
    status_row.addWidget(window.status_label, 1)

    status_right_col = QVBoxLayout()
    status_right_col.setContentsMargins(0, 0, 0, 0)
    status_right_col.addStretch(1)
    window.level_text = QLabel("Peak -inf dBFS | RMS -inf dBFS")
    status_right_col.addWidget(window.level_text, 0, Qt.AlignRight)
    status_row.addLayout(status_right_col, 0)
    layout.addLayout(status_row)

    meter_stack = QWidget()
    meter_stack_layout = QVBoxLayout(meter_stack)
    meter_stack_layout.setContentsMargins(0, 0, 0, 0)
    meter_stack_layout.setSpacing(0)
    window.peak_meter = ObsLevelMeter(show_ruler=False, overlay_label="Peak")
    window.rms_meter = ObsLevelMeter(show_ruler=True, overlay_label="RMS")
    meter_stack_layout.addWidget(window.peak_meter)
    meter_stack_layout.addWidget(window.rms_meter)
    layout.addWidget(meter_stack)

    layout.addWidget(QLabel("Recent events"))
    window.recent_events = QPlainTextEdit()
    window.recent_events.setReadOnly(True)
    layout.addWidget(window.recent_events, stretch=1)
    window.tabs.addTab(tab, "Main")
