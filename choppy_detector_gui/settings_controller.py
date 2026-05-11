"""Controller helpers for main Settings tab form."""

from __future__ import annotations


def apply_settings_to_controls(window) -> None:
    window.auto_restart_minutes.setValue(window.settings.auto_restart_minutes)
    window.auto_start_monitoring.setChecked(window.settings.auto_start_monitoring)
    window.alert_cooldown_ms.setValue(window.settings.alert_cooldown_ms)
    window.twitch_channel.setText(window.settings.twitch_channel)
    window.twitch_bot_username.setText(window.settings.twitch_bot_username)
    window.twitch_oauth_token.setText(window.settings.twitch_oauth_token)
    commands = window.settings.chat_commands
    window.start_command.setText(commands.start_command)
    window.stop_command.setText(commands.stop_command)
    window.restart_command.setText(commands.restart_command)
    window.status_command.setText(commands.status_command)
    window.list_devices_command.setText(commands.list_devices_command)
    window.fix_command.setText(commands.fix_command)
    window.rebuild_command.setText(commands.rebuild_command)
    window.clip_command.setText(commands.clip_command)
    window.switch_device_command_prefix.setText(commands.switch_device_command_prefix)
    window.allowed_chat_users.setPlainText("\n".join(commands.allowed_chat_users))
    window.send_command_responses.setChecked(commands.send_command_responses)
    window.logs_enabled.setChecked(window.settings.log_settings.logs_enabled)
    window.log_window_retention_minutes.setValue(window.settings.log_settings.log_window_retention_minutes)
    window.keep_preview_while_monitoring.setChecked(window.settings.keep_preview_while_monitoring)
    window.smooth_preview_meter.setChecked(window.settings.smooth_preview_meter)
    window.preview_meter_fps.setValue(window.settings.preview_meter_fps)
    window.dark_mode_enabled.setChecked(window.settings.dark_mode_enabled)
    window.enable_clip_capture_buffer.setChecked(window.settings.enable_clip_capture_buffer)
    window.obs_auto_connect_on_launch.setChecked(window.settings.obs_websocket.auto_connect_on_launch)
    window.obs_auto_connect_retry_enabled.setChecked(window.settings.obs_websocket.auto_connect_retry_enabled)
    window.log_directory.setText(window.settings.log_settings.log_directory)


def collect_settings_from_controls(window) -> None:
    window.settings.auto_restart_minutes = window.auto_restart_minutes.value()
    window.settings.auto_start_monitoring = window.auto_start_monitoring.isChecked()
    window.settings.alert_cooldown_ms = window.alert_cooldown_ms.value()
    window.settings.twitch_channel = window.twitch_channel.text().strip().lstrip("#")
    window.settings.twitch_bot_username = window.twitch_bot_username.text().strip()
    window.settings.twitch_oauth_token = window.twitch_oauth_token.text().strip()
    commands = window.settings.chat_commands
    commands.chat_commands_enabled = window.chat_commands_enabled.isChecked()
    commands.start_command = window.start_command.text().strip()
    commands.stop_command = window.stop_command.text().strip()
    commands.restart_command = window.restart_command.text().strip()
    commands.status_command = window.status_command.text().strip()
    commands.list_devices_command = window.list_devices_command.text().strip()
    commands.fix_command = window.fix_command.text().strip()
    commands.rebuild_command = window.rebuild_command.text().strip()
    commands.clip_command = window.clip_command.text().strip()
    commands.switch_device_command_prefix = window.switch_device_command_prefix.text().strip()
    commands.allowed_chat_users = _allowed_chat_users(window)
    commands.send_command_responses = window.send_command_responses.isChecked()
    window.settings.log_settings.logs_enabled = window.logs_enabled.isChecked()
    window.settings.log_settings.log_window_retention_minutes = window.log_window_retention_minutes.value()
    window.settings.keep_preview_while_monitoring = window.keep_preview_while_monitoring.isChecked()
    window.settings.smooth_preview_meter = window.smooth_preview_meter.isChecked()
    window.settings.preview_meter_fps = window.preview_meter_fps.value()
    window.settings.dark_mode_enabled = window.dark_mode_enabled.isChecked()
    window.settings.enable_clip_capture_buffer = window.enable_clip_capture_buffer.isChecked()
    window.settings.obs_websocket.auto_connect_on_launch = window.obs_auto_connect_on_launch.isChecked()
    window.settings.obs_websocket.auto_connect_retry_enabled = window.obs_auto_connect_retry_enabled.isChecked()
    window.settings.log_settings.log_directory = window.log_directory.text().strip()


def settings_dirty(window) -> bool:
    commands = window.settings.chat_commands
    allowed_users = _allowed_chat_users(window)
    return any(
        (
            window.auto_restart_minutes.value() != window.settings.auto_restart_minutes,
            window.auto_start_monitoring.isChecked() != window.settings.auto_start_monitoring,
            window.alert_cooldown_ms.value() != window.settings.alert_cooldown_ms,
            window.twitch_channel.text().strip().lstrip("#") != window.settings.twitch_channel,
            window.twitch_bot_username.text().strip() != window.settings.twitch_bot_username,
            window.twitch_oauth_token.text().strip() != window.settings.twitch_oauth_token,
            window.chat_commands_enabled.isChecked() != commands.chat_commands_enabled,
            window.start_command.text().strip() != commands.start_command,
            window.stop_command.text().strip() != commands.stop_command,
            window.restart_command.text().strip() != commands.restart_command,
            window.status_command.text().strip() != commands.status_command,
            window.list_devices_command.text().strip() != commands.list_devices_command,
            window.fix_command.text().strip() != commands.fix_command,
            window.rebuild_command.text().strip() != commands.rebuild_command,
            window.clip_command.text().strip() != commands.clip_command,
            window.switch_device_command_prefix.text().strip() != commands.switch_device_command_prefix,
            allowed_users != commands.allowed_chat_users,
            window.send_command_responses.isChecked() != commands.send_command_responses,
            window.logs_enabled.isChecked() != window.settings.log_settings.logs_enabled,
            window.log_window_retention_minutes.value() != window.settings.log_settings.log_window_retention_minutes,
            window.keep_preview_while_monitoring.isChecked() != window.settings.keep_preview_while_monitoring,
            window.smooth_preview_meter.isChecked() != window.settings.smooth_preview_meter,
            window.preview_meter_fps.value() != window.settings.preview_meter_fps,
            window.dark_mode_enabled.isChecked() != window.settings.dark_mode_enabled,
            window.enable_clip_capture_buffer.isChecked() != window.settings.enable_clip_capture_buffer,
            window.obs_auto_connect_on_launch.isChecked() != window.settings.obs_websocket.auto_connect_on_launch,
            window.obs_auto_connect_retry_enabled.isChecked() != window.settings.obs_websocket.auto_connect_retry_enabled,
            window.log_directory.text().strip() != window.settings.log_settings.log_directory,
        )
    )


def _allowed_chat_users(window) -> list[str]:
    return [
        line.strip().lower()
        for line in window.allowed_chat_users.toPlainText().splitlines()
        if line.strip()
    ]
