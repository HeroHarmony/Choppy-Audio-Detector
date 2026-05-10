"""WebSocket tab UI builder."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def _wrap_layout(inner_layout: QHBoxLayout) -> QWidget:
    inner_layout.setContentsMargins(0, 0, 0, 0)
    inner_layout.setSpacing(8)
    wrapper = QWidget()
    wrapper.setLayout(inner_layout)
    return wrapper


def build_websocket_tab(window) -> None:
    tab = QWidget()
    window.websocket_tab = tab
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
    window.obs_enabled = QCheckBox("Enable OBS WebSocket integration")
    window.obs_enabled.stateChanged.connect(window.update_obs_controls_enabled)
    connection_form.addRow("Enabled", window.obs_enabled)
    window.obs_host = QLineEdit()
    window.obs_host.setPlaceholderText("127.0.0.1")
    window.obs_host.textChanged.connect(window.update_obs_bundle_network_notice)
    connection_form.addRow("Host", window.obs_host)
    window.obs_port = QSpinBox()
    window.obs_port.setRange(1, 65535)
    connection_form.addRow("Port", window.obs_port)
    window.obs_password = QLineEdit()
    window.obs_password.setEchoMode(QLineEdit.Password)
    window.obs_password.setPlaceholderText("OBS WebSocket password")
    connection_form.addRow("Password", window.obs_password)
    window.obs_status = QLabel("Disconnected")
    window.obs_status.setStyleSheet("color: #ff9c4a; font-weight: 700;")
    connection_form.addRow("Status", window.obs_status)
    window.obs_bundle_network_notice = QLabel(
        "Note: macOS packaged app builds without an Apple signing certificate may fail to reach LAN OBS hosts "
        "(\"No route to host\") even when Local Network access appears enabled. "
        "Use `run_gui.command`, use localhost when OBS is local, or use a properly signed build."
    )
    window.obs_bundle_network_notice.setWordWrap(True)
    window.obs_bundle_network_notice.setStyleSheet("color: #f0c04a;")
    window.obs_bundle_network_notice.hide()
    connection_form.addRow("", window.obs_bundle_network_notice)

    connection_buttons = QHBoxLayout()
    window.obs_connect_button = QPushButton("Connect")
    window.obs_disconnect_button = QPushButton("Disconnect")
    window.obs_test_button = QPushButton("Test OBS Connection")
    window.obs_connect_button.clicked.connect(window.connect_obs)
    window.obs_disconnect_button.clicked.connect(window.disconnect_obs)
    window.obs_test_button.clicked.connect(window.test_obs_connection)
    connection_buttons.addWidget(window.obs_connect_button)
    connection_buttons.addWidget(window.obs_disconnect_button)
    connection_buttons.addWidget(window.obs_test_button)
    connection_form.addRow("", _wrap_layout(connection_buttons))

    auto_refresh_group = QGroupBox("Automation - Auto Refresh")
    auto_refresh_layout = QVBoxLayout(auto_refresh_group)
    auto_refresh_desc = QLabel(
        "Automatically refreshes the selected OBS source after detected audio issues."
    )
    auto_refresh_desc.setWordWrap(True)
    auto_refresh_desc.setStyleSheet("color: #bdbdbd;")
    auto_refresh_layout.addWidget(auto_refresh_desc)
    auto_refresh_form = QFormLayout()
    auto_refresh_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    auto_refresh_form.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    window.obs_auto_refresh_enabled = QCheckBox("Auto refresh on detected issues")
    auto_refresh_form.addRow("Auto refresh", window.obs_auto_refresh_enabled)
    window.obs_target_scene = QComboBox()
    window.obs_target_scene.setEditable(False)
    window.obs_refresh_scenes_button = QPushButton("Refresh Scenes")
    window.obs_refresh_scenes_button.clicked.connect(window.refresh_obs_scenes)
    scene_row = QHBoxLayout()
    scene_row.addWidget(window.obs_target_scene, 1)
    scene_row.addWidget(window.obs_refresh_scenes_button)
    scene_field = _wrap_layout(scene_row)
    scene_field.setToolTip("Optional. Leave as 'All Scenes' to search all scenes.")
    auto_refresh_form.addRow("Scene", scene_field)
    window.obs_target_source = QComboBox()
    window.obs_target_source.setEditable(False)
    window.obs_refresh_sources_button = QPushButton("Refresh Sources")
    window.obs_refresh_sources_button.clicked.connect(window.refresh_obs_sources)
    source_row = QHBoxLayout()
    source_row.addWidget(window.obs_target_source, 1)
    source_row.addWidget(window.obs_refresh_sources_button)
    auto_refresh_form.addRow("Source", _wrap_layout(source_row))
    window.obs_auto_refresh_min_severity = QComboBox()
    window.obs_auto_refresh_min_severity.addItems(["minor", "moderate", "severe"])
    window.obs_auto_refresh_min_severity.setMinimumContentsLength(10)
    window.obs_auto_refresh_min_severity.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    window.obs_auto_refresh_min_severity.setMinimumWidth(140)
    auto_refresh_form.addRow("Min severity", window.obs_auto_refresh_min_severity)
    window.obs_auto_refresh_cooldown_sec = QSpinBox()
    window.obs_auto_refresh_cooldown_sec.setRange(0, 86400)
    window.obs_auto_refresh_cooldown_sec.setSuffix(" sec")
    auto_refresh_form.addRow("Cooldown", window.obs_auto_refresh_cooldown_sec)
    window.obs_refresh_off_on_delay_ms = QSpinBox()
    window.obs_refresh_off_on_delay_ms.setRange(0, 10000)
    window.obs_refresh_off_on_delay_ms.setSingleStep(50)
    window.obs_refresh_off_on_delay_ms.setSuffix(" ms")
    auto_refresh_form.addRow("Off/on delay", window.obs_refresh_off_on_delay_ms)
    auto_refresh_layout.addLayout(auto_refresh_form)

    baseline_relearn_group = QGroupBox("Automation - Baseline Relearn")
    baseline_relearn_layout = QVBoxLayout(baseline_relearn_group)
    baseline_relearn_desc = QLabel(
        "Baseline is the app's reference for what clean audio sounds like; this can relearn it after a scene change."
    )
    baseline_relearn_desc.setWordWrap(True)
    baseline_relearn_desc.setStyleSheet("color: #bdbdbd;")
    baseline_relearn_layout.addWidget(baseline_relearn_desc)
    baseline_relearn_form = QFormLayout()
    baseline_relearn_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    baseline_relearn_form.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    window.obs_baseline_rebuild_enabled = QCheckBox("Relearn baseline after scene exit")
    baseline_relearn_form.addRow("Baseline relearn", window.obs_baseline_rebuild_enabled)
    window.obs_baseline_scene = QComboBox()
    window.obs_baseline_scene.setEditable(False)
    window.obs_baseline_scene.addItem("Any Scene")
    baseline_relearn_form.addRow("Watch scene", window.obs_baseline_scene)
    window.obs_baseline_min_dwell_sec = QSpinBox()
    window.obs_baseline_min_dwell_sec.setRange(1, 3600)
    window.obs_baseline_min_dwell_sec.setSuffix(" sec")
    baseline_relearn_form.addRow("Min dwell", window.obs_baseline_min_dwell_sec)
    window.obs_baseline_exit_delay_sec = QSpinBox()
    window.obs_baseline_exit_delay_sec.setRange(0, 3600)
    window.obs_baseline_exit_delay_sec.setSuffix(" sec")
    baseline_relearn_form.addRow("Exit delay", window.obs_baseline_exit_delay_sec)
    window.obs_baseline_cooldown_sec = QSpinBox()
    window.obs_baseline_cooldown_sec.setRange(0, 86400)
    window.obs_baseline_cooldown_sec.setSuffix(" sec")
    baseline_relearn_form.addRow("Cooldown", window.obs_baseline_cooldown_sec)
    baseline_relearn_layout.addLayout(baseline_relearn_form)

    actions_group = QGroupBox("Actions")
    actions_layout = QVBoxLayout(actions_group)
    window.obs_refresh_now_button = QPushButton("Refresh Source Now")
    window.obs_refresh_now_button.clicked.connect(window.refresh_obs_source_now)
    window.obs_save_button = QPushButton("Save WebSocket Settings")
    window.obs_save_button.clicked.connect(window.save_obs_settings)
    actions_layout.addWidget(window.obs_refresh_now_button)
    actions_layout.addWidget(window.obs_save_button)

    if not window.obs_service.available:
        warning = QLabel(window.obs_service.unavailable_reason or "obsws-python is not installed.")
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #f0c04a;")
        layout.addWidget(warning)

    layout.addWidget(connection_group)
    layout.addWidget(auto_refresh_group)
    layout.addWidget(baseline_relearn_group)
    layout.addWidget(actions_group)
    layout.addStretch(1)
    window.tabs.addTab(tab, "WebSocket")
