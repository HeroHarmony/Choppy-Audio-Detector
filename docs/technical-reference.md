# Technical Reference

This page is for advanced tuning and script-first workflows.

For Playground tab workflows and report behavior, see [Playground Reference](playground-reference.md).

## Detector Tuning Objective

The detector is tuned for live-stream usefulness, not perfect offline sample classification.

The target outcome is:

- known-bad audio should escalate often enough to produce actionable Twitch chat alerts
- occasional false positives are acceptable if they stay bounded and do not become constant spam
- a single alert is already useful because the streamer can manually verify with viewers and investigate
- missing obvious recurring glitches is worse than producing a rare ignorable alert

In practice, tuning should prefer:

- better recall on real live-stream glitches
- bounded alert noise on changing real-world audio
- operational usefulness over perfect separation on synthetic or offline test sets

This means report review should ask these questions in order:

1. Would this glitchy sample eventually produce a real alert under production timing?
2. Would a clean sample avoid repeated or runaway alerts?
3. If there is a tradeoff, does it still help the live workflow more than it hurts it?

The detector is not optimized for:

- zero false positives at any cost
- perfect agreement with every human-marked subtle edge case
- maximizing offline classification score when that makes live alerts too conservative

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

### Production Timing

Production timing currently uses:

- `production_window_ms = 1000`
- `production_step_ms = 50`

This is the primary truth for live behavior. Longer-window comparisons such as `2000/200` are useful as reference lanes in Playground, but should not override production tuning decisions unless live behavior is also being changed intentionally.

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
