"""OBS-style audio level meter widget."""

from __future__ import annotations

import time

from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget


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
        disabled = not self.isEnabled()

        if disabled:
            dark_green = QColor("#3a3a3a")
            dark_yellow = QColor("#484848")
            dark_red = QColor("#555555")
            fill_green = QColor("#6e6e6e")
            fill_yellow = QColor("#828282")
            fill_red = QColor("#989898")
            line_color = QColor("#9a9a9a")
            text_color = QColor("#a5a5a5")
            overlay_color = QColor("#d0d0d0")
        else:
            dark_green = QColor("#154e1f")
            dark_yellow = QColor("#5c5712")
            dark_red = QColor("#5a1818")
            fill_green = QColor("#28c840")
            fill_yellow = QColor("#d8d230")
            fill_red = QColor("#e05050")
            line_color = QColor("#d0d0d0")
            text_color = QColor("#bdbdbd")
            overlay_color = QColor("#000000")

        # Background zones: dark green, dark yellow, dark red.
        green_end = meter_rect.width() * self._db_to_ratio(self._yellow_start_db)
        yellow_end = meter_rect.width() * self._db_to_ratio(self._red_start_db)
        self._fill_segment(painter, meter_rect, 0.0, green_end, dark_green)
        self._fill_segment(painter, meter_rect, green_end, yellow_end, dark_yellow)
        self._fill_segment(painter, meter_rect, yellow_end, meter_rect.width(), dark_red)

        fill_ratio = max(0.0, min(1.0, (self._dbfs - self._scale_min_db) / (self._scale_max_db - self._scale_min_db)))
        fill_width = meter_rect.width() * fill_ratio
        if fill_width <= 0:
            self._draw_zone_lines(painter, meter_rect, line_color)
            if self._show_ruler:
                self._draw_db_ruler(painter, full_rect, meter_rect, text_color)
            return

        self._fill_segment(painter, meter_rect, 0.0, min(fill_width, green_end), fill_green)
        if fill_width > green_end:
            self._fill_segment(painter, meter_rect, green_end, min(fill_width, yellow_end), fill_yellow)
        if fill_width > yellow_end:
            self._fill_segment(painter, meter_rect, yellow_end, fill_width, fill_red)

        # Current level indicator.
        marker_x = meter_rect.x() + fill_width
        painter.setPen(QColor("#cfcfcf") if disabled else QColor("#f5f5f5"))
        painter.drawLine(int(marker_x), int(meter_rect.y()), int(marker_x), int(meter_rect.y() + meter_rect.height()))

        if self._peak_hold_dbfs > self._scale_min_db:
            hold_ratio = self._db_to_ratio(self._peak_hold_dbfs)
            hold_x = meter_rect.x() + meter_rect.width() * hold_ratio
            painter.setPen(QColor("#d8d8d8") if disabled else QColor("#ffffff"))
            painter.drawLine(int(hold_x), int(meter_rect.y() - 1), int(hold_x), int(meter_rect.y() + meter_rect.height() + 1))

        self._draw_zone_lines(painter, meter_rect, line_color)
        if self._show_ruler:
            self._draw_db_ruler(painter, full_rect, meter_rect, text_color)
        if self._overlay_label:
            painter.setPen(overlay_color)
            painter.drawText(int(meter_rect.x() + 6), int(meter_rect.y() + 14), self._overlay_label)

    def _fill_segment(self, painter: QPainter, rect, start: float, end: float, color: QColor) -> None:
        width = max(0.0, end - start)
        if width <= 0:
            return
        painter.fillRect(QRectF(rect.x() + start, rect.y(), width, rect.height()), color)

    def _draw_zone_lines(self, painter: QPainter, rect: QRectF, color: QColor) -> None:
        painter.setPen(color)
        for ratio in (self._db_to_ratio(self._yellow_start_db), self._db_to_ratio(self._red_start_db)):
            x = rect.x() + rect.width() * ratio
            painter.drawLine(int(x), rect.y(), int(x), rect.y() + rect.height())

    def _draw_db_ruler(self, painter: QPainter, full_rect: QRectF, meter_rect: QRectF, color: QColor) -> None:
        tick_values = tuple(range(-60, 1, 6))
        tick_top = meter_rect.y() + meter_rect.height() + 3
        tick_bottom = tick_top + 5
        label_y = tick_bottom + 12
        painter.setPen(color)
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
