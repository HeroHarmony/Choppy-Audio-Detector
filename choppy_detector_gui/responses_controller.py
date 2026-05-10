"""Controller helpers for response template settings."""

from __future__ import annotations

from choppy_detector_gui.alert_templates import AlertTemplates


def collect_templates_from_controls(window) -> AlertTemplates:
    return AlertTemplates(
        first_minor=window.template_first_minor.toPlainText().strip(),
        first_moderate=window.template_first_moderate.toPlainText().strip(),
        first_severe=window.template_first_severe.toPlainText().strip(),
        ongoing=window.template_ongoing.toPlainText().strip(),
    )


def templates_dirty(window) -> bool:
    current = collect_templates_from_controls(window)
    saved = window.settings.alert_templates
    return any(
        (
            current.first_minor != saved.first_minor,
            current.first_moderate != saved.first_moderate,
            current.first_severe != saved.first_severe,
            current.ongoing != saved.ongoing,
            window.template_rebuild_response.text().strip()
            != window.settings.chat_commands.rebuild_response_template,
        )
    )


def apply_templates_to_controls(window) -> None:
    templates = window.settings.alert_templates
    window.template_first_minor.setPlainText(templates.first_minor)
    window.template_first_moderate.setPlainText(templates.first_moderate)
    window.template_first_severe.setPlainText(templates.first_severe)
    window.template_ongoing.setPlainText(templates.ongoing)
    window.template_rebuild_response.setText(window.settings.chat_commands.rebuild_response_template)


def reset_template_to_default(window, template_key: str) -> None:
    defaults = AlertTemplates()
    if template_key == "first_minor":
        window.template_first_minor.setPlainText(defaults.first_minor)
    elif template_key == "first_moderate":
        window.template_first_moderate.setPlainText(defaults.first_moderate)
    elif template_key == "first_severe":
        window.template_first_severe.setPlainText(defaults.first_severe)
    elif template_key == "ongoing":
        window.template_ongoing.setPlainText(defaults.ongoing)


def build_preview_text(window) -> tuple[str, list[str]]:
    templates = collect_templates_from_controls(window)
    errors = templates.validate_all()
    if errors:
        return ("\n".join(errors), errors)
    previews = [
        templates.render(
            detection_count=count,
            time_span_minutes=1.5,
            is_first_alert=is_first,
            confidence_threshold=70,
            device_name=window.runtime.device_summary(),
        )
        for count, is_first in ((6, True), (8, True), (12, True), (9, False))
    ]
    return ("\n".join(previews), [])
