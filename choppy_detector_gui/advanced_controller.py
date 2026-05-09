"""Controller helpers for advanced settings form."""

from __future__ import annotations

from choppy_detector_gui.settings import DEFAULT_ALERT_CONFIG, DEFAULT_APPROACHES, DEFAULT_THRESHOLDS


def apply_advanced_to_controls(window) -> None:
    for key, *_ in window.alert_config_schema():
        widget = window.advanced_widgets.get(f"value:{key}")
        if widget is None:
            continue
        value = window.settings.advanced_alert_config.get(key, DEFAULT_ALERT_CONFIG[key])
        window._set_advanced_widget_value(widget, value)
    for key, *_ in window.threshold_schema():
        widget = window.advanced_widgets.get(f"value:{key}")
        if widget is None:
            continue
        value = window.settings.advanced_thresholds.get(key, DEFAULT_THRESHOLDS[key])
        window._set_advanced_widget_value(widget, value)
    for key, _ in window.methods_schema():
        widget = window.advanced_widgets.get(f"method:{key}")
        if widget is None:
            continue
        widget.setChecked(bool(window.settings.detection_methods.get(key, DEFAULT_APPROACHES[key])))


def collect_advanced_from_controls(window) -> None:
    alert_config = dict(DEFAULT_ALERT_CONFIG)
    for key, *_ in window.alert_config_schema():
        widget = window.advanced_widgets.get(f"value:{key}")
        if widget is None:
            continue
        alert_config[key] = window._get_advanced_widget_value(widget, DEFAULT_ALERT_CONFIG[key])
    thresholds = dict(DEFAULT_THRESHOLDS)
    for key, *_ in window.threshold_schema():
        widget = window.advanced_widgets.get(f"value:{key}")
        if widget is None:
            continue
        thresholds[key] = window._get_advanced_widget_value(widget, DEFAULT_THRESHOLDS[key])
    methods = dict(DEFAULT_APPROACHES)
    for key, _ in window.methods_schema():
        widget = window.advanced_widgets.get(f"method:{key}")
        if widget is None:
            continue
        methods[key] = bool(widget.isChecked())

    window.settings.advanced_alert_config = alert_config
    window.settings.advanced_thresholds = thresholds
    window.settings.detection_methods = methods
    window.settings.alert_cooldown_ms = int(alert_config.get("alert_cooldown_ms", window.settings.alert_cooldown_ms))


def advanced_dirty(window) -> bool:
    for key, *_ in window.alert_config_schema():
        widget = window.advanced_widgets.get(f"value:{key}")
        if widget is None:
            continue
        current = window._get_advanced_widget_value(widget, DEFAULT_ALERT_CONFIG[key])
        saved = window.settings.advanced_alert_config.get(key, DEFAULT_ALERT_CONFIG[key])
        if current != saved:
            return True
    for key, *_ in window.threshold_schema():
        widget = window.advanced_widgets.get(f"value:{key}")
        if widget is None:
            continue
        current = window._get_advanced_widget_value(widget, DEFAULT_THRESHOLDS[key])
        saved = window.settings.advanced_thresholds.get(key, DEFAULT_THRESHOLDS[key])
        if current != saved:
            return True
    for key, _ in window.methods_schema():
        widget = window.advanced_widgets.get(f"method:{key}")
        if widget is None:
            continue
        if bool(widget.isChecked()) != bool(window.settings.detection_methods.get(key, DEFAULT_APPROACHES[key])):
            return True
    return False


def reset_advanced_defaults(window) -> None:
    window.settings.advanced_alert_config = dict(DEFAULT_ALERT_CONFIG)
    window.settings.advanced_thresholds = dict(DEFAULT_THRESHOLDS)
    window.settings.detection_methods = dict(DEFAULT_APPROACHES)
    window.settings.alert_cooldown_ms = int(DEFAULT_ALERT_CONFIG["alert_cooldown_ms"])
