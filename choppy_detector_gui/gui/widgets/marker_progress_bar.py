"""Marker-aware playback progress widget for Playground preview."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Signal, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class MarkerProgressBar(QWidget):
    markerClicked = Signal(int)
    seekRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration_ms = 1
        self._position_ms = 0
        self._markers_ms: list[int] = []
        self._active_marker_ms: int | None = None
        self._flash_on = False
        self._marker_click_tolerance_px = 9.0
        self.setMinimumHeight(24)
        self.setMaximumHeight(24)

    def set_duration_ms(self, duration_ms: int) -> None:
        self._duration_ms = max(1, int(duration_ms))
        self._position_ms = max(0, min(self._position_ms, self._duration_ms))
        self.update()

    def set_position_ms(self, position_ms: int) -> None:
        self._position_ms = max(0, min(int(position_ms), self._duration_ms))
        self.update()

    def set_markers_ms(self, markers_ms: list[int]) -> None:
        cleaned = sorted({max(0, min(int(v), self._duration_ms)) for v in markers_ms})
        self._markers_ms = cleaned
        self.update()

    def set_marker_alert(self, *, active_marker_ms: int | None, flash_on: bool) -> None:
        self._active_marker_ms = None if active_marker_ms is None else int(active_marker_ms)
        self._flash_on = bool(flash_on and active_marker_ms is not None)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        track_rect = self._track_rect()
        if self._markers_ms:
            hit = self._find_marker_near_point(event.position(), track_rect)
            if hit is not None:
                self.markerClicked.emit(hit)
                event.accept()
                return
        seek_ms = self._position_to_ms(event.position().x(), track_rect)
        self.seekRequested.emit(seek_ms)
        event.accept()
        return

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track_rect = self._track_rect()

        painter.setBrush(QColor("#2a2a2a"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track_rect, 4, 4)

        ratio = max(0.0, min(1.0, self._position_ms / float(max(1, self._duration_ms))))
        fill_width = track_rect.width() * ratio
        if fill_width > 0:
            fill_rect = QRectF(track_rect.x(), track_rect.y(), fill_width, track_rect.height())
            painter.setBrush(QColor("#2a82da"))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(fill_rect, 4, 4)

        border_color = QColor("#ff5c5c") if self._flash_on else QColor("#5a5a5a")
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(border_color, 1.6))
        painter.drawRoundedRect(track_rect, 4, 4)

        for marker_ms in self._markers_ms:
            marker_ratio = max(0.0, min(1.0, marker_ms / float(max(1, self._duration_ms))))
            x = track_rect.x() + track_rect.width() * marker_ratio
            y = track_rect.center().y()
            is_active = self._active_marker_ms is not None and marker_ms == int(self._active_marker_ms)
            marker_color = QColor("#ff5c5c") if is_active else QColor("#c8d9f2")
            painter.setBrush(marker_color)
            painter.setPen(QPen(QColor("#141414"), 1.0))
            painter.drawEllipse(QPointF(x, y), 4.2, 4.2)

    def _track_rect(self) -> QRectF:
        return QRectF(self.rect().adjusted(1, 4, -1, -4))

    def _find_marker_near_point(self, point: QPointF, track_rect: QRectF) -> int | None:
        closest_marker: int | None = None
        closest_dist = float("inf")
        for marker_ms in self._markers_ms:
            ratio = max(0.0, min(1.0, marker_ms / float(max(1, self._duration_ms))))
            x = track_rect.x() + track_rect.width() * ratio
            dist = abs(point.x() - x)
            if dist <= self._marker_click_tolerance_px and dist < closest_dist:
                closest_dist = dist
                closest_marker = marker_ms
        return closest_marker

    def _position_to_ms(self, x: float, track_rect: QRectF) -> int:
        width = max(1.0, float(track_rect.width()))
        ratio = (float(x) - float(track_rect.x())) / width
        ratio = max(0.0, min(1.0, ratio))
        return int(round(ratio * float(self._duration_ms)))
