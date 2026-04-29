"""OBS WebSocket integration for GUI-triggered source refresh."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Any


try:
    import obsws_python as obs
except Exception:  # pragma: no cover - optional dependency at runtime
    obs = None


@dataclass
class ObsConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 4455
    password: str = ""


class ObsWebSocketService:
    def __init__(self):
        self._lock = RLock()
        self._client = None
        self._connected = False
        self._last_error = ""

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def available(self) -> bool:
        return obs is not None

    def connect(self, config: ObsConnectionConfig, timeout_seconds: float = 3.0) -> tuple[bool, str]:
        if obs is None:
            return False, "OBS WebSocket client dependency missing. Install obsws-python."

        host = str(config.host).strip() or "127.0.0.1"
        try:
            port = int(config.port)
        except (TypeError, ValueError):
            return False, "Invalid OBS port."

        if not 1 <= port <= 65535:
            return False, "OBS port must be between 1 and 65535."

        with self._lock:
            try:
                self._client = obs.ReqClient(
                    host=host,
                    port=port,
                    password=config.password or "",
                    timeout=float(timeout_seconds),
                )
                self._connected = True
                self._last_error = ""
                return True, f"Connected to OBS at {host}:{port}"
            except Exception as exc:
                self._client = None
                self._connected = False
                self._last_error = str(exc)
                return False, f"Failed to connect to OBS: {exc}"

    def disconnect(self) -> None:
        with self._lock:
            self._client = None
            self._connected = False

    def list_sources(self) -> list[str]:
        with self._lock:
            client = self._client
            connected = self._connected
        if not connected or client is None:
            return []

        try:
            inputs = client.get_input_list().inputs
            names = [str(item.get("inputName", "")).strip() for item in inputs]
            return sorted([name for name in names if name])
        except Exception:
            return []

    def list_scenes(self) -> list[str]:
        with self._lock:
            client = self._client
            connected = self._connected
        if not connected or client is None:
            return []

        try:
            scenes = client.get_scene_list().scenes
            names = [str(item.get("sceneName", "")).strip() for item in scenes]
            return [name for name in names if name]
        except Exception:
            return []

    def refresh_source(self, source_name: str, off_on_delay_ms: int = 500) -> tuple[bool, str]:
        return self.refresh_source_in_scene(source_name=source_name, scene_name="", off_on_delay_ms=off_on_delay_ms)

    def refresh_source_in_scene(self, source_name: str, scene_name: str = "", off_on_delay_ms: int = 500) -> tuple[bool, str]:
        with self._lock:
            client = self._client
            connected = self._connected
        if not connected or client is None:
            return False, "OBS is not connected."

        source_name = str(source_name).strip()
        scene_name = str(scene_name).strip()
        if not source_name:
            return False, "No OBS source selected."

        delay_ms = max(0, min(10000, int(off_on_delay_ms or 0)))

        # Preferred path for this app: toggle scene items off then on so delay is honored.
        try:
            toggled = 0
            scene_candidates = [scene_name] if scene_name else self.list_scenes()
            for scene in scene_candidates:
                items = client.get_scene_item_list(scene).scene_items
                for item in items:
                    if str(item.get("sourceName", "")).strip() != source_name:
                        continue
                    scene_item_id = int(item.get("sceneItemId"))
                    client.set_scene_item_enabled(scene, scene_item_id, False)
                    if delay_ms > 0:
                        time.sleep(delay_ms / 1000.0)
                    client.set_scene_item_enabled(scene, scene_item_id, True)
                    toggled += 1

            if toggled > 0:
                delay_sec = delay_ms / 1000.0
                return True, (
                    f"Refreshed source '{source_name}' by toggling {toggled} scene item(s) "
                    f"with {delay_sec:.2f}s off/on delay"
                    + (f" in scene '{scene_name}'." if scene_name else ".")
                )
        except Exception:
            # Continue to media restart fallback below.
            pass

        # Fallback: restart media input directly when no scene-item toggle target is found.
        try:
            client.trigger_media_input_action(source_name, "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART")
            return True, (
                f"Refreshed source '{source_name}' via media restart fallback. "
                "Note: restart fallback does not keep the source off for the configured delay."
            )
        except Exception as exc:
            return False, (
                f"Failed to refresh source '{source_name}'. "
                "No scene item toggle target was found and media restart fallback failed: "
                f"{exc}"
            )
