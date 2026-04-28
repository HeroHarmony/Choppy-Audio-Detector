# OBS WebSocket Tab Plan

## Goal

Add a new `WebSocket` tab to the GUI so users can connect to OBS via OBS WebSocket and refresh a selected media source (toggle off then on) to force reconnect to SRTLA/RTMP when audio desync is detected.

Primary outcomes:

- Users can configure and test OBS WebSocket connection from the GUI.
- Users can select which OBS media source is refreshed.
- Users can manually trigger a refresh from the GUI.
- Users can enable automatic refresh based on detector severity.
- Users can control safety timing with cooldown and off/on delay.

## User Flow

1. Open `WebSocket` tab.
2. Enter OBS connection settings:
- Host (default `127.0.0.1`)
- Port (default `4455`)
- Password
- Connect/Disconnect button with status indicator
3. After connected, load OBS objects:
- Scene list
- Source list (media/input sources)
4. Choose refresh target source.
5. Configure auto-refresh policy:
- Minimum severity that can trigger refresh
- Cooldown between refreshes
- Delay between source off and source on
6. Save settings and enable automation.
7. Use `Refresh Now` for manual test.

## Scene Requirement Decision

Short answer: selecting scene should not be required in v1 unless implementation API requires scene context.

Reasoning:

- In OBS WebSocket v5, input enable/disable can be handled with input-centric methods, which usually do not require the scene to be selected by user.
- For sources duplicated across scenes or scene-item-specific control, scene context may be needed.

Plan:

- V1 default: source-only selection.
- If OBS API call fails because scene-item context is required, show a clear error and prompt scene selection.
- Optional enhancement: add `Auto-detect scene containing source` helper.

## Functional Requirements

- New GUI tab: `WebSocket`.
- Connection controls:
- `Connect`, `Disconnect`, `Test Connection`.
- Display connection state (`Disconnected`, `Connecting`, `Connected`, `Error`).
- OBS discovery controls:
- `Refresh Scenes` and `Refresh Sources`.
- Source targeting:
- Select one source to refresh.
- Persist selected source ID/name in settings.
- Manual action:
- `Refresh Source Now` button.
- Automatic action:
- Enable toggle: `Auto refresh on audio issue`.
- Severity threshold picker (for example: `minor`, `moderate`, `severe`).
- Cooldown input (seconds).
- Off/on delay input (milliseconds).
- Runtime behavior:
- When detector emits severity >= threshold and cooldown elapsed, execute refresh.
- Write action log entries to Console/log file with reason and timing.

## Non-Functional Requirements

- Never block GUI thread for WebSocket operations.
- Connection retries should be explicit/manual in v1 (no aggressive reconnect loop).
- Obvious operator feedback for all failures.
- Keep credentials out of log output.
- Persist settings across restarts.

## Proposed Architecture

### New module

- Add `choppy_detector_gui/obs_websocket_service.py`:
- Owns connect/disconnect, OBS API calls, and source refresh orchestration.
- Exposes methods such as:
- `connect(config)`
- `disconnect()`
- `list_scenes()`
- `list_sources()`
- `refresh_source(source_ref, off_on_delay_ms)`

### Settings model updates

- Extend GUI settings with:
- `obs_enabled` (bool)
- `obs_host` (str)
- `obs_port` (int)
- `obs_password` (str, persisted carefully)
- `obs_target_source` (str)
- `obs_auto_refresh_enabled` (bool)
- `obs_auto_refresh_min_severity` (enum)
- `obs_auto_refresh_cooldown_sec` (int)
- `obs_refresh_off_on_delay_ms` (int)

### Runtime integration

- Hook detector severity events from runtime into an `OBSRefreshController` check:
- Gate by `auto_refresh_enabled`.
- Gate by severity threshold.
- Gate by cooldown clock.
- Trigger `obs_websocket_service.refresh_source(...)`.
- Emit structured runtime events/logs:
- `obs_connect_success`
- `obs_connect_failed`
- `obs_refresh_triggered`
- `obs_refresh_skipped_cooldown`
- `obs_refresh_failed`

## UX Details

- Tab sections:
- `Connection`
- `Target Source`
- `Automation`
- `Actions`
- Show last refresh timestamp and next eligible refresh time.
- Disable source controls when disconnected.
- Confirm when user clicks manual refresh if no source is selected.

## Validation Rules

- Host required.
- Port must be valid range.
- Cooldown must be >= 0.
- Off/on delay must be >= 0 and capped to sane upper bound (for example 10,000 ms).
- Source must be selected before allowing auto-refresh enable.

## Testing Plan

- Unit tests:
- Cooldown gating logic.
- Severity threshold comparison.
- Refresh orchestration call order (off -> wait -> on).
- Settings serialization/deserialization.
- Integration/manual tests:
- Connect to local OBS with/without password.
- Select media source and run manual refresh.
- Verify source visibility/enable toggles correctly.
- Trigger simulated severities and validate threshold behavior.
- Confirm cooldown prevents repeated toggles.
- Confirm logs contain action details but no password.

## Rollout Phases

### Phase 1: Core Connection + Manual Refresh

- Add tab UI scaffold.
- Implement connect/disconnect and source listing.
- Implement manual refresh button.
- Persist connection and source settings.

### Phase 2: Auto-Refresh Policy

- Add severity threshold, cooldown, and off/on delay controls.
- Integrate runtime-triggered refresh with gating.
- Add console/file log entries.

### Phase 3: Hardening

- Improve error messages for scene-context edge cases.
- Add optional scene picker only if needed by real OBS behavior.
- Add guardrails for reconnect and stale source selection.

## Open Questions

- Should OBS password be stored plaintext in current settings, or moved to OS keychain/credential manager?
- Which detector severity signal should be authoritative for auto-refresh (`minor/moderate/severe` currently expected)?
- Should first auto-refresh after app start ignore cooldown, or respect persisted last-run timestamp?
- For duplicated source names across scenes, should we store OBS UUID/reference instead of display name?
