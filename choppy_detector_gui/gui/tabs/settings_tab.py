"""Settings tab UI builder."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def build_settings_tab(window) -> None:
    tab = QWidget()
    window.settings_tab = tab
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
    window.auto_restart_minutes = QSpinBox()
    window.auto_restart_minutes.setRange(5, 1440)
    general_form.addRow("Auto-restart minutes", window.auto_restart_minutes)
    window.auto_start_monitoring = QCheckBox("Start monitoring on app start")
    general_form.addRow("Auto-start monitoring", window.auto_start_monitoring)
    window.obs_auto_connect_on_launch = QCheckBox("Start websocket on app start")
    general_form.addRow("Auto-connect OBS", window.obs_auto_connect_on_launch)
    window.obs_auto_connect_retry_enabled = QCheckBox("5 retries, 60 sec cooldown")
    general_form.addRow("OBS auto-connect retry", window.obs_auto_connect_retry_enabled)

    window.alert_cooldown_ms = QSpinBox()
    window.alert_cooldown_ms.setRange(1000, 3600000)
    window.alert_cooldown_ms.setSingleStep(1000)
    general_form.addRow("Alert cooldown ms", window.alert_cooldown_ms)

    window.logs_enabled = QCheckBox("Write log files")
    general_form.addRow("Logs", window.logs_enabled)
    window.log_window_retention_minutes = QSpinBox()
    window.log_window_retention_minutes.setRange(1, 10080)
    window.log_window_retention_minutes.setSuffix(" min")
    general_form.addRow("Log window retention", window.log_window_retention_minutes)
    window.keep_preview_while_monitoring = QCheckBox("Keep Preview Running")
    general_form.addRow("Preview mode", window.keep_preview_while_monitoring)
    window.dark_mode_enabled = QCheckBox("Enable dark mode")
    general_form.addRow("Theme", window.dark_mode_enabled)
    window.enable_clip_capture_buffer = QCheckBox("Enable !choppy clip rolling capture")
    general_form.addRow("Clip capture buffer", window.enable_clip_capture_buffer)
    window.smooth_preview_meter = QCheckBox("Smooth preview animation")
    general_form.addRow("Meter smoothing", window.smooth_preview_meter)
    window.preview_meter_fps = QSpinBox()
    window.preview_meter_fps.setRange(5, 60)
    general_form.addRow("Preview meter FPS", window.preview_meter_fps)

    window.log_directory = QLineEdit()
    window.log_directory.setPlaceholderText("Leave blank for ./Log")
    window.log_directory.setMinimumWidth(220)
    general_form.addRow("Log directory", window.log_directory)

    twitch_group = QGroupBox("Twitch Connection")
    twitch_group.setStyleSheet(group_title_style)
    twitch_layout = QVBoxLayout(twitch_group)
    twitch_layout.setSpacing(8)
    window.twitch_channel = QLineEdit()
    window.twitch_channel.setPlaceholderText("Channel name, without #")
    window.twitch_channel.setMinimumWidth(250)
    twitch_layout.addWidget(QLabel("Twitch channel"))
    twitch_layout.addWidget(window.twitch_channel)

    window.twitch_bot_username = QLineEdit()
    window.twitch_bot_username.setPlaceholderText("Bot username")
    window.twitch_bot_username.setMinimumWidth(250)
    twitch_layout.addWidget(QLabel("Twitch bot username"))
    twitch_layout.addWidget(window.twitch_bot_username)

    window.twitch_oauth_token = QLineEdit()
    window.twitch_oauth_token.setPlaceholderText("oauth:...")
    window.twitch_oauth_token.setEchoMode(QLineEdit.Password)
    window.twitch_oauth_token.setMinimumWidth(250)
    twitch_layout.addWidget(QLabel("Twitch OAuth token"))
    twitch_layout.addWidget(window.twitch_oauth_token)
    oauth_link = QLabel(
        '<a href="https://twitchtokengenerator.com/">Get Twitch OAuth token / client ID</a>'
    )
    oauth_link.setOpenExternalLinks(True)
    oauth_link.setTextFormat(Qt.RichText)
    oauth_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
    oauth_link.setStyleSheet("color: #7db8ff;")
    twitch_layout.addWidget(oauth_link)

    commands_group = QGroupBox("Chat Commands and Permissions")
    commands_group.setStyleSheet(group_title_style)
    commands_form = QFormLayout(commands_group)
    window.start_command = QLineEdit()
    window.stop_command = QLineEdit()
    window.restart_command = QLineEdit()
    window.status_command = QLineEdit()
    window.list_devices_command = QLineEdit()
    window.fix_command = QLineEdit()
    window.rebuild_command = QLineEdit()
    window.clip_command = QLineEdit()
    window.switch_device_command_prefix = QLineEdit()
    for cmd_input in (
        window.start_command,
        window.stop_command,
        window.restart_command,
        window.status_command,
        window.list_devices_command,
        window.fix_command,
        window.rebuild_command,
        window.clip_command,
        window.switch_device_command_prefix,
    ):
        cmd_input.setMinimumWidth(150)
    commands_form.addRow("Start command", window.start_command)
    commands_form.addRow("Stop command", window.stop_command)
    commands_form.addRow("Restart command", window.restart_command)
    commands_form.addRow("Status command", window.status_command)
    commands_form.addRow("List devices command", window.list_devices_command)
    commands_form.addRow("Refresh OBS Source", window.fix_command)
    commands_form.addRow("Rebuild baseline", window.rebuild_command)
    commands_form.addRow("Capture clip", window.clip_command)
    commands_form.addRow("Switch device prefix", window.switch_device_command_prefix)

    window.allow_broadcaster_commands = QCheckBox("Allow broadcaster")
    window.allow_moderator_commands = QCheckBox("Allow moderators")
    permission_row = QWidget()
    permission_row_layout = QHBoxLayout(permission_row)
    permission_row_layout.setContentsMargins(0, 0, 0, 0)
    permission_row_layout.setSpacing(12)
    permission_row_layout.addWidget(window.allow_broadcaster_commands)
    permission_row_layout.addWidget(window.allow_moderator_commands)
    permission_row_layout.addStretch(1)
    commands_form.addRow("Role bypass", permission_row)

    window.allowed_chat_users = QPlainTextEdit()
    window.allowed_chat_users.setPlaceholderText("One Twitch username per line")
    window.allowed_chat_users.setFixedHeight(120)
    window.allowed_chat_users.setMinimumWidth(0)
    allowed_users_col = QWidget()
    allowed_users_col_layout = QVBoxLayout(allowed_users_col)
    allowed_users_col_layout.setContentsMargins(0, 0, 0, 0)
    allowed_users_col_layout.setSpacing(4)
    allowed_users_col_layout.addWidget(window.allowed_chat_users)
    allowed_users_hint = QLabel("One Twitch username per line.")
    allowed_users_hint.setStyleSheet("color: #bdbdbd;")
    allowed_users_col_layout.addWidget(allowed_users_hint)
    commands_form.addRow("Allowed users", allowed_users_col)

    window.send_command_responses = QCheckBox("Send to chat")
    commands_form.addRow("Command responses", window.send_command_responses)

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
    window.save_settings_button = QPushButton("Save Settings")
    window.save_settings_button.clicked.connect(window.save_all_settings)
    content_layout.addWidget(window.save_settings_button)
    window.tabs.addTab(tab, "Settings")
