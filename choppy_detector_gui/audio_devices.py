"""Shared audio device discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sounddevice as sd


@dataclass(frozen=True)
class AudioDeviceInfo:
    selection_index: int
    portaudio_index: int
    name: str
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default: bool
    obs_monitoring_hint: str | None = None

    @property
    def is_monitorable(self) -> bool:
        return self.max_input_channels > 0

    @property
    def display_name(self) -> str:
        default_text = " (default)" if self.is_default else ""
        kind = "input" if self.is_monitorable else "output only - routing reference"
        return f"{self.selection_index}: {self.name} [{kind}]{default_text}"

    @property
    def full_label(self) -> str:
        return f"{self.selection_index} - {self.name}"


def get_hostapi_name(hostapi_index: int | None) -> str:
    try:
        return str(sd.query_hostapis(hostapi_index)["name"])
    except Exception:
        return "Unknown"


def infer_obs_monitoring_device(device_name: str) -> str | None:
    normalized = " ".join(str(device_name).split())
    replacements = (
        ("CABLE Output", "CABLE Input"),
        ("Voicemeeter Output", "Voicemeeter Input"),
        ("Voicemeeter AUX Output", "Voicemeeter AUX Input"),
        ("Voicemeeter VAIO3 Output", "Voicemeeter VAIO3 Input"),
    )
    for source, target in replacements:
        if source in normalized:
            return normalized.replace(source, target)
    return None


def list_audio_devices(include_output_only: bool = True) -> list[AudioDeviceInfo]:
    devices = sd.query_devices()
    default_input = _default_input_device()
    default_output = _default_output_device()
    audio_devices: list[AudioDeviceInfo] = []

    for portaudio_index, raw_device in enumerate(devices):
        device: dict[str, Any] = dict(raw_device)
        max_input_channels = int(device.get("max_input_channels", 0) or 0)
        max_output_channels = int(device.get("max_output_channels", 0) or 0)
        if max_input_channels <= 0 and (not include_output_only or max_output_channels <= 0):
            continue

        is_default = (
            portaudio_index == default_input
            if max_input_channels > 0
            else portaudio_index == default_output
        )
        audio_devices.append(
            AudioDeviceInfo(
                selection_index=len(audio_devices),
                portaudio_index=portaudio_index,
                name=str(device.get("name") or f"Device {portaudio_index}"),
                hostapi_name=get_hostapi_name(device.get("hostapi")),
                max_input_channels=max_input_channels,
                max_output_channels=max_output_channels,
                default_samplerate=float(device.get("default_samplerate", 0.0) or 0.0),
                is_default=is_default,
                obs_monitoring_hint=infer_obs_monitoring_device(str(device.get("name") or "")),
            )
        )

    audio_devices.sort(key=lambda item: (not item.is_monitorable, item.selection_index))
    return [
        AudioDeviceInfo(
            selection_index=index,
            portaudio_index=device.portaudio_index,
            name=device.name,
            hostapi_name=device.hostapi_name,
            max_input_channels=device.max_input_channels,
            max_output_channels=device.max_output_channels,
            default_samplerate=device.default_samplerate,
            is_default=device.is_default,
            obs_monitoring_hint=device.obs_monitoring_hint,
        )
        for index, device in enumerate(audio_devices)
    ]


def list_input_devices() -> list[AudioDeviceInfo]:
    return list_audio_devices(include_output_only=False)


def resolve_selection_to_portaudio_index(selection_index: int | None) -> int | None:
    if selection_index is None:
        return None
    devices = list_input_devices()
    if 0 <= selection_index < len(devices):
        return devices[selection_index].portaudio_index
    return None


def device_label_for_portaudio_index(portaudio_index: int | None) -> str:
    if portaudio_index is None:
        return "Default input device"
    for device in list_audio_devices(include_output_only=True):
        if device.portaudio_index == portaudio_index:
            return device.full_label
    return f"PortAudio device {portaudio_index}"


def _default_input_device() -> int | None:
    try:
        default_device = sd.default.device
        if isinstance(default_device, (list, tuple)):
            return int(default_device[0])
        return int(default_device)
    except Exception:
        return None


def _default_output_device() -> int | None:
    try:
        default_device = sd.default.device
        if isinstance(default_device, (list, tuple)):
            return int(default_device[1])
        return int(default_device)
    except Exception:
        return None
