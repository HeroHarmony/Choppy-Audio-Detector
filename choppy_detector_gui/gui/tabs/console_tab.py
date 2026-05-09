"""Console tab UI builder."""

from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


def build_console_tab(window) -> None:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    window.console_output = QPlainTextEdit()
    window.console_output.setReadOnly(True)
    layout.addWidget(window.console_output)
    window.clear_console_button = QPushButton("Clear Console")
    window.clear_console_button.clicked.connect(window.clear_console_messages)
    layout.addWidget(window.clear_console_button)
    window.tabs.addTab(tab, "Console")
