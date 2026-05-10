# Technical Reference

This page is for advanced tuning and script-first workflows.

For Playground tab workflows and report behavior, see [Playground Reference](playground-reference.md).

## CLI Usage (`live_analysis.py`)

Show available audio devices:

```bash
python live_analysis.py --list-devices
```

Use specific audio device and disable Twitch alerts:

```bash
python live_analysis.py --audio-device 2 --no-twitch
```

Enable Twitch alerts with specific audio device:

```bash
python live_analysis.py --twitch --audio-device 1
```

Use a specific channel within the selected audio device:

```bash
python live_analysis.py --audio-device 2 --audio-channel 1
```

### Command Line Options

- `--list-devices`: Lists all available audio input devices and exits.
- `--audio-device DEVICE_ID`: Specifies the audio input device by ID.
- `--audio-channel N` (legacy alias: `--channel N`): Selects input audio channel index.
- `--twitch`: Enables Twitch chat notifications.
- `--no-twitch`: Disables Twitch chat notifications.
- `--twitch-channel NAME`: Overrides Twitch channel (without `#`).
- `--twitch-bot-username USER`: Overrides Twitch bot username.
- `--twitch-oauth-token TOKEN`: Overrides Twitch OAuth token.

## Runtime Configuration (Script Mode)

When using `live_analysis.py` directly, these dictionaries control behavior.

### Alert Configuration

```python
ALERT_CONFIG = {
    "detections_for_alert": 6,
    "alert_cooldown_ms": 60000,
    "detection_window_seconds": 90,
    "confidence_threshold": 70,
    "clean_audio_reset_seconds": 60,
}
```

- `detections_for_alert`: Detections needed before an alert is sent.
- `alert_cooldown_ms`: Minimum time between alerts.
- `detection_window_seconds`: Window for counting recent detections.
- `confidence_threshold`: Minimum confidence percentage to count a detection.
- `clean_audio_reset_seconds`: Clean-audio duration before a new episode starts.

### Thresholds

```python
THRESHOLDS = {
    "silence_ratio": 0.60,
    "amplitude_jump": 2.5,
    "envelope_discontinuity": 2.0,
    "gap_duration_ms": 100,
    "min_audio_level": 0.005,
    "max_normal_gaps": 2,
    "suspicious_gap_count": 4,
}
```

### Detection Methods

```python
APPROACHES = {
    "silence_gaps": True,
    "amplitude_jumps": False,
    "envelope_discontinuity": True,
    "temporal_consistency": False,
    "energy_variance": False,
    "zero_crossings": False,
    "spectral_rolloff": False,
    "spectral_centroid": False,
}
```

## Detection Method Notes

- `silence_gaps`: Detects dropouts and sustained silent gaps.
- `envelope_discontinuity`: Detects abrupt envelope breaks and drop-return patterns.
- `amplitude_jumps`: Can be noisy with normal speech/music dynamics.
- `temporal_consistency`: Sensitive to natural expressive speech changes.
- `energy_variance`: Often high in games/music, can false-positive.
- `zero_crossings`: Weak correlation with user-audible streaming glitches.
- `spectral_rolloff`: Varies naturally across content types.
- `spectral_centroid`: Also highly content-dependent.

## Example Runtime Output

```text
INFO:twitch_chat:Connecting to irc.chat.twitch.tv:6667...
INFO:twitch_chat:Successfully connected to Twitch
Active detection methods:
  silence_gaps=True
  envelope_discontinuity=True
Listening for streaming audio glitches...
```
