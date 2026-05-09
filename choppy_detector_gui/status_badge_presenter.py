"""Small presenter for global status-bar badges."""

from __future__ import annotations


class StatusBadgePresenter:
    def __init__(self, label_widget, *, prefix: str):
        self._label_widget = label_widget
        self._prefix = prefix

    def apply(self, label: str, color_hex: str) -> None:
        self._label_widget.setText(f"{self._prefix}: {label}")
        self._label_widget.setStyleSheet(
            f"color: {color_hex}; font-weight: 700; border: 1px solid {color_hex};"
            " border-radius: 8px; padding: 2px 8px;"
        )
