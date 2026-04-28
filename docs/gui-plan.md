# Cross-Platform GUI Plan

## Goal

Build a desktop GUI for Choppy Audio Detector that works on Windows and macOS, with Windows as the first-class target. The GUI should make the detector easier to run during a stream without editing Python files or watching a terminal.

Primary features:

- Select the monitored audio input device from a drop-down list.
- Show a live audio level meter so the user can immediately verify that the selected input has audio.
- Edit Twitch warning message templates for minor, moderate, severe, and ongoing alerts.
- Restart the `live_analysis.py` engine automatically on a schedule, defaulting to every 60 minutes and configurable in settings.
- Start and stop detection from Twitch chat commands, with configurable command text.
- Start, stop, and restart monitoring from the GUI.
- Automatically restart detection when the selected audio input device changes while detection is running.
- Write local log files for glitches and operator actions such as start, stop, restart, and audio device changes.
- Surface detector status, current device, recent warnings, and useful error messages.

## Current Project Shape

The repo is currently a script-first Python app:

- `live_analysis.py` owns device listing, interactive prompts, detector state, audio stream lifecycle, glitch detection, Twitch alert construction, logging, and command-line parsing.
- `twitch_chat.py` owns Twitch IRC connection and message sending.
- Optional runtime config is loaded from `config.py`, including Twitch credentials and `ALERT_COOLDOWN_MS`.

This is a good starting point, but the GUI will need a cleaner boundary around the detector runtime. The current `BalancedChoppyDetector.start_detection()` method blocks, prints status to stdout, and owns its own retry loop. That is fine for CLI use, but the GUI needs non-blocking status updates and a controlled way to change settings.

The CLI must remain a supported entry point after the GUI refactor. Existing commands should keep working:

- `python live_analysis.py`
- `python live_analysis.py --list-devices`
- `python live_analysis.py --audio-device 2 --no-twitch`
- `python live_analysis.py --twitch --audio-device 1`

## Recommendation

Use a Python-native GUI with PySide6.

Reasoning:

- The existing detector is already Python and depends on `numpy` and `sounddevice`.
- PySide6 is the official Qt for Python binding and gives us mature native desktop widgets on Windows and macOS.
- Qt has strong built-in support for long-running background work, timers, settings, menus, tray icons, and native-looking controls.
- Packaging can be handled by PyInstaller or `pyside6-deploy`, with Windows builds produced on Windows and macOS builds produced on macOS.

Recommended initial stack:

- `PySide6` for GUI.
- `sounddevice` for audio device listing and audio capture, retained from the current project.
- `numpy` for existing detector math.
- `platformdirs` for user config paths.
- `pydantic` or `dataclasses` plus JSON for settings validation. Prefer `dataclasses` first unless validation becomes complex.
- `PyInstaller` for first packaging pass, especially for Windows.

Official references checked:

- PySide6 deployment docs: https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html
- Qt for Python PyInstaller notes: https://doc.qt.io/qtforpython-6.9/deployment/deployment-pyinstaller.html
- PyInstaller usage docs: https://www.pyinstaller.org/en/stable/usage.html

## GUI Library Options

### Option A: PySide6

Best fit for this project.

Pros:

- Native desktop UI without adding a web frontend stack.
- Works on Windows and macOS.
- Good support for real-time UI updates with signals, slots, timers, and worker threads.
- Easy to build a compact operational tool: device selector, meters, log pane, settings dialog, status bar.
- Keeps the detector and GUI in one language.

Cons:

- Packaging Qt apps can need some tuning.
- GUI code should avoid blocking calls on the main thread.
- Commercial licensing is not needed for normal LGPL usage, but distribution should respect Qt/PySide license obligations.

### Option B: Tkinter

Possible, but not recommended.

Pros:

- Bundled with most Python installs.
- Lightweight.

Cons:

- Less polished UI.
- More awkward for modern settings forms, meters, and background process supervision.
- Packaging still needs work, so the main advantage is limited.

### Option C: Tauri/Electron Frontend plus Python Backend

Possible, but heavier than needed right now.

Pros:

- Rich UI possibilities.
- Tauri can supervise sidecar processes.

Cons:

- Adds JavaScript/TypeScript, Node tooling, and a second application runtime.
- Requires an IPC contract between frontend and Python.
- More moving pieces before the detector itself has stable GUI-facing APIs.

## Proposed Architecture

### Phase 1: Make the Detector GUI-Friendly

Refactor without changing detection behavior:

- Move audio device discovery into an importable function that returns structured device records.
- Move alert message construction into a small template renderer.
- Add a settings model for GUI-editable values.
- Add callbacks or an event queue for status updates:
  - audio level updates
  - device open/close
  - glitch detections
  - alert queued/sent/failed
  - runtime errors
- Keep the CLI path working.

Suggested modules:

- `choppy_detector_gui/audio_devices.py`
- `choppy_detector_gui/detector.py`
- `choppy_detector_gui/settings.py`
- `choppy_detector_gui/alert_templates.py`
- `choppy_detector_gui/runtime.py`
- `app_gui.py`

### Phase 2: Build a Minimal GUI Shell

Keep the first GUI simple and organized around tabs.

Tab 1: Main

- Device drop-down.
- Twitch enabled toggle.
- Twitch chat commands enabled toggle.
- Start/Stop button.
- Restart button.
- Live audio meter.
- Current status label.
- Recent events log.

Tab 2: Response Templates

- First minor alert template.
- First moderate alert template.
- First severe alert template.
- Ongoing alert template.
- Template preview using sample values.
- Validation for unknown template variables.

Tab 3: Settings

- Twitch channel name.
- Auto-restart interval in minutes.
- Alert cooldown.
- Twitch chat command settings.
- Detection count/window/confidence settings, if desired.
- Optional detection method toggles, if exposed.

Tab 4: Console

- Append-only console output that mirrors what the user would see when running `live_analysis.py` in a terminal.
- Include detector startup, selected device, audio stream status, volume heartbeat, glitch detections, alert queue/send results, Twitch connection messages, restarts, and errors.
- Allow copy/select text.
- Include a clear-console button.
- Consider a save-log button later, but it is not required for the first version.

Possible later tabs or sections:

- Diagnostics: audio callback health, current sample rate, callback freshness, queue drops, detector thread state.
- About/help: version, config path, build info, links to README.

These can wait unless the first GUI feels too cramped.

### Phase 2.5: Twitch Chat Remote Controls

Add optional Twitch chat command listening so trusted chat users can start and stop detection without touching the machine running the GUI.

Recommended default commands:

- Start detection: `!choppy start`
- Stop detection: `!choppy stop`
- Restart detection: `!choppy restart`
- Status: `!choppy status`
- List audio devices: `!choppy devices`
- Switch audio device: `!choppy device 2`

Settings should allow the user to change these command strings. For example, the user may prefer `!audio start`, `!audio stop`, or separate compact commands like `!startdetector`.

Recommended command settings:

- `chat_commands_enabled`: default `false`
- `start_command`: default `!choppy start`
- `stop_command`: default `!choppy stop`
- `restart_command`: default `!choppy restart`
- `status_command`: default `!choppy status`
- `list_devices_command`: default `!choppy devices`
- `switch_device_command_prefix`: default `!choppy device`
- `allowed_chat_users`: list of Twitch usernames allowed to control detection
- `allow_broadcaster`: default `true`, if IRC badges are available
- `allow_moderators`: default `true`, if IRC badges are available
- `send_command_responses`: default `true`

Important safety behavior:

- Do not allow every chatter to start/stop detection by default.
- The simplest reliable permission model is an explicit allowed-user list in Settings.
- Broadcaster/moderator permissions can be added if Twitch IRC badges are parsed reliably.
- Normalize command matching by trimming whitespace and comparing case-insensitively.
- Log every accepted and rejected remote command in the GUI event log.
- If a command is received while the GUI is already in the requested state, respond with the current state instead of restarting unnecessarily.
- Manual GUI controls and Twitch commands should call the same runtime methods so state stays consistent.
- Audio device switching by chat command should use the same restart behavior as selecting a device in the GUI.
- Device switch commands should reference the displayed device enum from the GUI/device list, not raw PortAudio indexes.
- After a successful chat-initiated device switch, announce the selected device in Twitch chat if command responses are enabled.
- If the requested device number is invalid or cannot be opened, announce a short failure message in Twitch chat and write the full error to the GUI console/log.

Implementation detail:

- Extend `twitch_chat.py` so it can listen continuously and parse IRC metadata well enough to identify username and moderator/broadcaster badges.
- Keep Twitch message sending and command listening on background threads, never on the GUI thread.
- Emit runtime events such as `remote_start_requested`, `remote_stop_requested`, `remote_restart_requested`, and `remote_device_switch_requested`; the GUI/runtime controller should decide whether to execute them.
- If Twitch disconnects, detection should continue. The GUI should show chat control status separately from detector status.

Example chat responses:

- `Monitoring started.`
- `Monitoring stopped.`
- `Monitoring restarted.`
- `Current device: 2 - CABLE Output (VB-Audio Virtual Cable).`
- `Available devices: 0: Microphone, 1: Line In, 2: CABLE Output.`
- `Switching monitor device to 2 - CABLE Output (VB-Audio Virtual Cable).`
- `Could not switch device: invalid device number.`

### Phase 3: Runtime Supervision

Preferred long-term approach: run the detector in-process on worker threads and restart the audio stream/detector object on schedule.

Short-term fallback: run `live_analysis.py` as a subprocess with non-interactive flags. This is lower-risk for an initial GUI prototype, but it limits live meter quality and settings integration unless stdout becomes structured.

Recommended path:

1. Add importable runtime controls.
2. Use PySide6 `QThread` or `QThreadPool` for detector runtime.
3. Use a `QTimer` for scheduled restart.
4. On restart:
   - stop detection
   - close Twitch/audio resources
   - clear transient runtime state
   - start detection again with the same saved settings

The restart interval should be stored in settings as minutes, default `60`, with validation such as min `5` and max `1440`.

Device changes should use the same restart path:

- If detection is stopped, changing the device only updates the pending selection.
- If detection is running, changing the device should prompt an automatic restart of the detector runtime.
- The restart should close the current audio stream, reset the baseline, open the new device, and resume monitoring.
- The console tab should log both the old and new device names.
- If the new device fails to open, the GUI should show a clear error and leave detection stopped rather than silently falling back to the old device.
- Output/render devices should be shown for routing reference, but should be marked as output-only unless the platform exposes a real capture/loopback endpoint for them.
- Chat and GUI device switching should reject output-only rows with a clear message that the user must select a matching input/capture endpoint or route the output through a virtual cable.

## File Logging

The GUI should write persistent log files under a repo/local app `Log` folder by default, with local timestamps.

Recommended path during development:

- `Log/choppy-audio-detector-YYYY-MM-DD.log`

Recommended installed-app path later:

- Windows: `%LOCALAPPDATA%/ChoppyAudioDetector/Log/choppy-audio-detector-YYYY-MM-DD.log`
- macOS: `~/Library/Logs/ChoppyAudioDetector/choppy-audio-detector-YYYY-MM-DD.log`

Events to log:

- App start and shutdown.
- Detection start, stop, manual restart, scheduled restart, and chat-command restart.
- Audio device selection changes, including old device and new device.
- Audio stream opened/closed/retry failures.
- Twitch connect/disconnect/reconnect.
- Accepted and rejected Twitch commands, including sender username and reason.
- Glitch detections with local timestamp, confidence, reasons, active detection methods, current audio device, and whether a Twitch alert was queued/sent.
- Alert send failures.
- Unhandled detector/runtime errors.

Log format:

- Use normal text logs first because they are easy to read.
- Include enough structure to search/filter later.
- Use local time with timezone offset, for example `2026-04-27 21:43:12.385 -0400`.

Example lines:

```text
2026-04-27 21:43:12.385 -0400 INFO monitoring.started device="2 - CABLE Output (VB-Audio Virtual Cable)" source="gui"
2026-04-27 22:05:44.091 -0400 WARN glitch.detected confidence=80.0 device="2 - CABLE Output (VB-Audio Virtual Cable)" reasons="Audio envelope break detected"
2026-04-27 22:10:03.522 -0400 INFO device.changed old="2 - CABLE Output" new="1 - Line In" source="twitch" user="hero_mod"
```

The Console tab should show the same important messages, but the file log is the durable record.

## Live Audio Meter

The existing detector already calculates RMS in the detection loop and reports volume every 10 seconds. The GUI should update much more frequently, around 10 to 20 times per second.

Implementation options:

- Compute RMS in the audio callback and emit a lightweight meter event.
- Smooth the visible meter in the GUI to avoid jitter.
- Show both:
  - a moving bar meter
  - a small text value like `-32.4 dBFS`

Important detail: keep the audio callback lightweight. It should copy audio into the buffer and push simple level information only. Detection work should remain outside the callback.

## Alert Templates

Current hard-coded messages:

- First alert: `{severity} Audio issues detected! {detection_count} glitches in {time_span_minutes:.1f} minutes. Stream audio may be choppy! modCheck`
- Ongoing alert: `{severity} Ongoing audio issue: {detection_count} glitches in last {time_span_minutes:.1f} minutes. Still unstable... modCheck`

Proposed template variables:

- `{severity}`: `[MINOR]`, `[MODERATE]`, or `[SEVERE]`
- `{detection_count}`
- `{time_span_minutes}`
- `{confidence_threshold}`
- `{device_name}`
- `{timestamp}`

Settings should store templates separately from Twitch credentials. The renderer should validate templates before saving so a typo like `{detecton_count}` is caught in the GUI.

## Settings Storage

Do not require users to edit `config.py` for normal GUI settings.

Use a JSON settings file under the user config directory:

- Windows: `%APPDATA%/ChoppyAudioDetector/settings.json`
- macOS: `~/Library/Application Support/ChoppyAudioDetector/settings.json`

Keep Twitch credentials support compatible with the current `config.py` in the first pass. Later, the GUI can add a credentials screen if desired, but credentials storage needs a separate decision because OAuth tokens are sensitive.

Chat command settings can be stored in the normal JSON settings file because command text and allowed usernames are not secrets. OAuth tokens should still be treated separately.

Logging settings can also be stored in JSON:

- `logs_enabled`: default `true`
- `log_directory`: optional override
- `log_retention_days`: optional, default `30`

## Packaging

Windows is the priority.

Recommended first packaging target:

- Windows `.exe` or folder-based app using PyInstaller `--onedir`.
- Avoid `--onefile` initially because audio/Qt dependencies are easier to diagnose in folder mode.
- Build Windows packages on Windows. PyInstaller is not a cross-compiler.

macOS target:

- Build on macOS.
- Use `--windowed` to produce a `.app` bundle.
- Signing/notarization can be deferred unless distributing broadly outside local use.

Potential packaging blockers to test early:

- PortAudio/sounddevice binary bundling.
- Qt platform plugins.
- Access to audio input devices on macOS privacy/security prompts.
- Antivirus false positives for unsigned Windows builds.

## Open Questions

- Should the first GUI run the detector in-process, or is a subprocess wrapper acceptable for the first prototype?
- Should Twitch credentials remain only in `config.py`, or should the GUI eventually manage them?
- Are alert templates enough, or should the GUI also expose thresholds like `detections_for_alert`, `confidence_threshold`, and detection method toggles?
- Should auto-restart restart only the audio stream, or the entire detector runtime including Twitch reconnect and baseline reset?
- Should Twitch chat commands use only an explicit allow-list first, then add broadcaster/moderator badge support later?
- Do we need a tray icon/minimize-to-tray behavior for stream use?
- Should the app remember the selected device by PortAudio index, device name, or both? Device indexes can change after reboot or audio device changes.
- Should the GUI include a test alert button that sends a Twitch message without waiting for glitches?
- What is the minimum Windows version to support?
- Should the Console tab be an exact stdout mirror only, or should it also include structured GUI-only events such as settings saves?
- Should log files live beside the app in `Log/` for portable installs, or under the OS-specific app data/log directory for installed builds?

## Implementation Milestones

### Milestone 1: Project Structure and Settings

- Add `requirements.txt` or `pyproject.toml`.
- Create `choppy_detector_gui/` package.
- Move settings/default constants into importable modules.
- Add JSON settings load/save.
- Preserve CLI behavior.

### Milestone 2: Runtime Events

- Add structured runtime events.
- Add audio level events.
- Add alert template renderer.
- Add file logging for runtime events and glitch detections.
- Add tests for template rendering and settings validation.

### Milestone 3: GUI Prototype

- Build PySide6 main window.
- Add four tabs: Main, Response Templates, Settings, Console.
- Device drop-down.
- Start/Stop/Restart controls.
- Live meter.
- Recent event log.
- Response template editor.
- Settings controls for auto-restart and Twitch commands.
- Console output mirror.
- Automatic runtime restart when the selected device changes while detection is running.
- Log file writing under `Log/`.

### Milestone 3.5: Twitch Chat Commands

- Add Twitch chat listener mode.
- Parse command sender and permissions.
- Add settings for command text and allowed users.
- Route accepted commands through the same runtime controller used by GUI buttons.
- Add chat commands for listing audio devices and switching selected audio device by enum.
- Announce successful chat-initiated device switches in Twitch chat.
- Add tests for command matching and permission checks.

### Milestone 4: Scheduled Restart

- Add GUI-controlled restart interval.
- Verify restart closes and reopens audio cleanly.
- Make manual and scheduled restarts use the same code path.

### Milestone 5: Packaging

- Add PyInstaller spec for Windows.
- Build and test on Windows first.
- Add macOS package path after Windows is working.

## Suggested First Implementation Decision

Start with PySide6 and an in-process detector refactor. It is a little more work than wrapping the script as a subprocess, but it directly supports the live meter, settings, and clean scheduled restarts. The subprocess path is useful only as a temporary prototype or diagnostic fallback.
