"""Composed runtime event pipeline for GUI orchestration."""

from __future__ import annotations

from collections.abc import Callable

from choppy_detector_gui.runtime_event_router import RuntimeEventContext, route_runtime_event


class RuntimeEventPipeline:
    def __init__(
        self,
        *,
        twitch_status,
        set_twitch_status_badge: Callable[[str, str], None],
        is_alerts_enabled: Callable[[], bool],
        is_chat_enabled: Callable[[], bool],
        context_provider: Callable[[], RuntimeEventContext],
        presenter,
        queue_obs_refresh_request: Callable[[str, str], None],
    ) -> None:
        self.twitch_status = twitch_status
        self.set_twitch_status_badge = set_twitch_status_badge
        self.is_alerts_enabled = is_alerts_enabled
        self.is_chat_enabled = is_chat_enabled
        self.context_provider = context_provider
        self.presenter = presenter
        self.queue_obs_refresh_request = queue_obs_refresh_request

    def handle(self, event_type: str, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        twitch_state_event, badge = self.twitch_status.on_event(
            event_type,
            data,
            alerts_enabled=self.is_alerts_enabled(),
            chat_enabled=self.is_chat_enabled(),
        )
        if twitch_state_event:
            self.set_twitch_status_badge(*badge)

        route = route_runtime_event(event_type, data, self.context_provider())
        self.presenter.apply_route(route, data)

        if route.obs_refresh_request is not None:
            self.queue_obs_refresh_request(
                route.obs_refresh_request.source,
                route.obs_refresh_request.action,
            )

        if route.consume_event or not route.append_formatted_event:
            return
        self.presenter.append_formatted_event(event_type, data)
