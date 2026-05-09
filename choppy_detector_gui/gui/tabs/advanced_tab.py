"""Advanced settings tab UI builder."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from choppy_detector_gui.settings import DEFAULT_ALERT_CONFIG, DEFAULT_APPROACHES, DEFAULT_THRESHOLDS


def build_advanced_tab(window) -> None:
    tab = QWidget()
    window.advanced_tab = tab
    layout = QVBoxLayout(tab)
    layout.setSpacing(10)
    layout.setContentsMargins(8, 8, 8, 8)
    group_title_style = "QGroupBox { font-size: 19px; font-weight: 600; }"

    window.advanced_widgets = {}
    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(0)

    alert_group = QGroupBox("Alert Config")
    alert_group.setStyleSheet(group_title_style)
    alert_layout = QVBoxLayout(alert_group)
    alert_layout.setSpacing(8)
    for key, desc, value_type, min_v, max_v, step in window.alert_config_schema():
        row = window._advanced_row(
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
    for key, desc, value_type, min_v, max_v, step in window.threshold_schema():
        row = window._advanced_row(
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
    for key, desc in window.methods_schema():
        checkbox = QCheckBox()
        window.advanced_widgets[f"method:{key}"] = checkbox
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
        desc_label = QLabel(f"{desc} (Default: {window._format_default_value(default_value)})")
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
    window.reset_advanced_defaults_btn = QPushButton("Reset Defaults")
    window.reset_advanced_defaults_btn.clicked.connect(window.reset_advanced_defaults)
    window.save_advanced_btn = QPushButton("Save Advanced Settings")
    window.save_advanced_btn.clicked.connect(window.save_advanced_settings)
    actions.addWidget(window.reset_advanced_defaults_btn)
    actions.addWidget(window.save_advanced_btn)
    layout.addLayout(actions)
    window.tabs.addTab(tab, "Advanced")
