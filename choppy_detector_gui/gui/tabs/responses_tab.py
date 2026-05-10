"""Responses tab UI builder."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def build_responses_tab(window) -> None:
    tab = QWidget()
    window.templates_tab = tab
    layout = QVBoxLayout(tab)
    window.template_first_minor = QTextEdit()
    window.template_first_moderate = QTextEdit()
    window.template_first_severe = QTextEdit()
    window.template_ongoing = QTextEdit()
    for editor in (
        window.template_first_minor,
        window.template_first_moderate,
        window.template_first_severe,
        window.template_ongoing,
    ):
        editor.setFixedHeight(76)

    content_row = QHBoxLayout()
    left_col = QVBoxLayout()
    form = QFormLayout()
    form.addRow("First minor", _template_row(window, window.template_first_minor, "first_minor"))
    form.addRow("First moderate", _template_row(window, window.template_first_moderate, "first_moderate"))
    form.addRow("First severe", _template_row(window, window.template_first_severe, "first_severe"))
    form.addRow("Ongoing", _template_row(window, window.template_ongoing, "ongoing"))
    window.template_rebuild_response = QLineEdit()
    window.template_rebuild_response.setPlaceholderText("Baseline relearn started.")
    window.template_rebuild_response.setToolTip("Twitch chat response for !choppy rebuild. Token: {user}")
    form.addRow("Rebuild reply", window.template_rebuild_response)
    left_col.addLayout(form)

    button_row = QHBoxLayout()
    window.preview_templates_button = QPushButton("Preview")
    window.preview_templates_button.clicked.connect(window.preview_templates)
    window.save_templates_button = QPushButton("Save Templates")
    window.save_templates_button.clicked.connect(window.save_templates)
    button_row.addWidget(window.preview_templates_button)
    button_row.addWidget(window.save_templates_button)
    left_col.addLayout(button_row)

    window.template_preview = QPlainTextEdit()
    window.template_preview.setReadOnly(True)
    left_col.addWidget(window.template_preview, 1)

    guide_col = QVBoxLayout()
    guide_col.addWidget(QLabel("Template variables"))
    guide_col.addWidget(_template_reference_guide(window), 1)

    content_row.addLayout(left_col, 5)
    content_row.addLayout(guide_col, 2)
    layout.addLayout(content_row, 1)
    window.tabs.addTab(tab, "Responses")


def _template_reference_guide(window) -> QWidget:
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
        panel_layout.addWidget(_reference_token_row(window, token, description))
    panel_layout.addStretch(1)
    return panel


def _reference_token_row(window, token: str, description: str) -> QWidget:
    row = QWidget()
    row_layout = QVBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(2)

    top = QHBoxLayout()
    token_label = QLabel(token)
    token_label.setStyleSheet("font-family: Menlo, Monaco, Courier New, monospace; font-weight: 600;")
    copy_btn = QToolButton()
    copy_btn.setText("Copy")
    copy_btn.clicked.connect(lambda: window.copy_template_token(token))
    top.addWidget(token_label)
    top.addStretch(1)
    top.addWidget(copy_btn)

    desc_label = QLabel(description)
    desc_label.setWordWrap(True)
    desc_label.setStyleSheet("color: #bdbdbd;")

    row_layout.addLayout(top)
    row_layout.addWidget(desc_label)
    return row


def _template_row(window, editor: QTextEdit, template_key: str) -> QWidget:
    row_widget = QWidget()
    row_layout = QHBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.addWidget(editor, 1)
    reset_button = QToolButton()
    reset_button.setText("Default")
    reset_button.clicked.connect(lambda: window.reset_template_to_default(template_key))
    row_layout.addWidget(reset_button, 0, Qt.AlignTop)
    return row_widget
