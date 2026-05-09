"""Controller helpers for OBS WebSocket settings form."""

from __future__ import annotations

from choppy_detector_gui.obs_network_notice import should_show_unsigned_bundle_notice
from choppy_detector_gui.obs_settings_binding import (
    apply_form_values_to_settings,
    is_form_dirty,
    read_form_values,
)


def apply_obs_settings_to_controls(window) -> None:
    obs_settings = window.settings.obs_websocket
    window.obs_enabled.setChecked(obs_settings.enabled)
    window.obs_host.setText(obs_settings.host)
    window.obs_port.setValue(obs_settings.port)
    window.obs_password.setText(obs_settings.password)
    window.obs_auto_refresh_enabled.setChecked(obs_settings.auto_refresh_enabled)
    idx = window.obs_auto_refresh_min_severity.findText(obs_settings.auto_refresh_min_severity)
    window.obs_auto_refresh_min_severity.setCurrentIndex(max(0, idx))
    window.obs_auto_refresh_cooldown_sec.setValue(obs_settings.auto_refresh_cooldown_sec)
    window.obs_refresh_off_on_delay_ms.setValue(obs_settings.refresh_off_on_delay_ms)
    window.refresh_obs_scenes()
    if obs_settings.target_scene:
        idx_scene = window.obs_target_scene.findText(obs_settings.target_scene)
        if idx_scene >= 0:
            window.obs_target_scene.setCurrentIndex(idx_scene)
    window.refresh_obs_sources()
    window.update_obs_controls_enabled()
    update_obs_bundle_network_notice(window)


def collect_obs_from_controls(window) -> None:
    values = read_form_values(
        enabled=window.obs_enabled.isChecked(),
        host=window.obs_host.text(),
        port=window.obs_port.value(),
        password=window.obs_password.text(),
        auto_refresh_enabled=window.obs_auto_refresh_enabled.isChecked(),
        auto_refresh_min_severity=window.obs_auto_refresh_min_severity.currentText(),
        auto_refresh_cooldown_sec=window.obs_auto_refresh_cooldown_sec.value(),
        refresh_off_on_delay_ms=window.obs_refresh_off_on_delay_ms.value(),
        scene_value=window.obs_target_scene.currentText(),
        source_value=window.obs_target_source.currentText(),
    )
    apply_form_values_to_settings(values, window.settings.obs_websocket)
    update_obs_bundle_network_notice(window)


def websocket_dirty(window) -> bool:
    current = read_form_values(
        enabled=window.obs_enabled.isChecked(),
        host=window.obs_host.text(),
        port=window.obs_port.value(),
        password=window.obs_password.text(),
        auto_refresh_enabled=window.obs_auto_refresh_enabled.isChecked(),
        auto_refresh_min_severity=window.obs_auto_refresh_min_severity.currentText(),
        auto_refresh_cooldown_sec=window.obs_auto_refresh_cooldown_sec.value(),
        refresh_off_on_delay_ms=window.obs_refresh_off_on_delay_ms.value(),
        scene_value=window.obs_target_scene.currentText(),
        source_value=window.obs_target_source.currentText(),
    )
    return is_form_dirty(window.settings.obs_websocket, current)


def update_obs_bundle_network_notice(window) -> None:
    if should_show_unsigned_bundle_notice(window.obs_host.text()):
        window.obs_bundle_network_notice.show()
    else:
        window.obs_bundle_network_notice.hide()
