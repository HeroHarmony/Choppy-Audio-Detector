"""Alert message template rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from string import Formatter
from typing import Any


ALLOWED_TEMPLATE_FIELDS = {
    "severity",
    "detection_count",
    "time_span_minutes",
    "confidence_threshold",
    "device_name",
    "timestamp",
}


DEFAULT_FIRST_ALERT_TEMPLATE = (
    "{severity} Audio issues detected! {detection_count} glitches in "
    "{time_span_minutes:.1f} minutes. Stream audio may be choppy! modCheck"
)

DEFAULT_ONGOING_ALERT_TEMPLATE = (
    "{severity} Ongoing audio issue: {detection_count} glitches in last "
    "{time_span_minutes:.1f} minutes. Still unstable... modCheck"
)


@dataclass
class AlertTemplates:
    first_minor: str = DEFAULT_FIRST_ALERT_TEMPLATE
    first_moderate: str = DEFAULT_FIRST_ALERT_TEMPLATE
    first_severe: str = DEFAULT_FIRST_ALERT_TEMPLATE
    ongoing: str = DEFAULT_ONGOING_ALERT_TEMPLATE
    _formatter: Formatter = field(default_factory=Formatter, init=False, repr=False)

    def template_for(self, detection_count: int, is_first_alert: bool) -> str:
        if not is_first_alert:
            return self.ongoing
        if detection_count >= 12:
            return self.first_severe
        if detection_count >= 8:
            return self.first_moderate
        return self.first_minor

    def render(
        self,
        *,
        detection_count: int,
        time_span_minutes: float,
        is_first_alert: bool,
        confidence_threshold: int,
        device_name: str,
        timestamp: datetime | None = None,
    ) -> str:
        severity = severity_for_detection_count(detection_count)
        context = {
            "severity": severity,
            "detection_count": detection_count,
            "time_span_minutes": time_span_minutes,
            "confidence_threshold": confidence_threshold,
            "device_name": device_name,
            "timestamp": (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
        }
        return self.template_for(detection_count, is_first_alert).format(**context)

    def validate_all(self) -> list[str]:
        errors: list[str] = []
        for name, template in (
            ("first_minor", self.first_minor),
            ("first_moderate", self.first_moderate),
            ("first_severe", self.first_severe),
            ("ongoing", self.ongoing),
        ):
            errors.extend(f"{name}: {error}" for error in validate_template(template))
        return errors

    def to_dict(self) -> dict[str, str]:
        return {
            "first_minor": self.first_minor,
            "first_moderate": self.first_moderate,
            "first_severe": self.first_severe,
            "ongoing": self.ongoing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AlertTemplates":
        if not isinstance(data, dict):
            return cls()
        return cls(
            first_minor=str(data.get("first_minor") or DEFAULT_FIRST_ALERT_TEMPLATE),
            first_moderate=str(data.get("first_moderate") or DEFAULT_FIRST_ALERT_TEMPLATE),
            first_severe=str(data.get("first_severe") or DEFAULT_FIRST_ALERT_TEMPLATE),
            ongoing=str(data.get("ongoing") or DEFAULT_ONGOING_ALERT_TEMPLATE),
        )


def severity_for_detection_count(detection_count: int) -> str:
    if detection_count >= 12:
        return "[SEVERE]"
    if detection_count >= 8:
        return "[MODERATE]"
    return "[MINOR]"


def validate_template(template: str) -> list[str]:
    errors: list[str] = []
    formatter = Formatter()
    try:
        parsed = list(formatter.parse(template))
    except ValueError as exc:
        return [str(exc)]

    for _, field_name, _, _ in parsed:
        if not field_name:
            continue
        root_name = field_name.split(".", 1)[0].split("[", 1)[0]
        if root_name not in ALLOWED_TEMPLATE_FIELDS:
            errors.append(f"unknown field {{{field_name}}}")

    sample = {
        "severity": "[MINOR]",
        "detection_count": 6,
        "time_span_minutes": 1.5,
        "confidence_threshold": 70,
        "device_name": "CABLE Output",
        "timestamp": "2026-04-27 21:43:12",
    }
    try:
        template.format(**sample)
    except Exception as exc:
        errors.append(f"format error: {exc}")

    return errors

