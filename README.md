# Choppy-Audio-Detector
This is a Python-based script that **listens to an audio channel in real-time and detects streaming-related audio glitches**. It will send chat messages to your Twitch channel when it detects these issues.

**Most common use case:** You're streaming remotely from a mobile phone/device and your audio passes through a computer — typically via OBS, SRT ingest, or Belabox. As long as you can isolate and monitor the audio, you can use this script.

I was inspired to create this tool after regularly experiencing audio issues, particularly audio glitches caused by unstable bitrate connections when streaming outdoors. In particular it seems that OBS's "Media Source" is not 100% reliable in playing back SRT (and RTMP?). In my pursuit to find a solution, I realized I'm not alone. Small streamers like myself often don't have the chatters to help identify these issues in real-time. Other times chatters may mention issues, but instructions are not clear. It can be very frustrating to find out hours later that your stream had audio issues.

The audio glitches can be described as "robotic," "choppy," "broken,", and "glitchy." 

Here's one such example: [Glitchy Twitch VOD Audio](https://www.twitch.tv/videos/2529044366?t=1h19m2s)

The script is largely developed with the help of AI. While I am a software engineer, I am not an expert in audio processing.

# Getting Started
## Prerequisites

To get started we need a few things. Be prepared to setup the following:
- Python 3.x
- (Optional) Twitch account and OAuth token for chat integration
- Have an understanding of your audio input devices. If you use Voicemeeter, you're in luck, as I use it too.

## Setup

1. **Install Python**: Make sure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).
2. **Install Required Libraries**: You can install the required libraries using pip. Open a terminal and run the following command:
   ```bash
   pip install numpy sounddevice
   ```
3. **Configure Twitch Integration**: If you want to enable Twitch chat notifications, you'll need to set up the config.py file with your Twitch credentials. 
    ```python
    TWITCH_CHANNEL = "HeroHarmony"
    TWITCH_BOT_USERNAME = "HeroBot"
    TWITCH_OAUTH_TOKEN = "oauth:your_token_here"
    ```
   Replace the placeholders with your actual Twitch channel name, bot username, and the bot's OAuth token. You can generate an OAuth token from [Twitch Token Generator](https://twitchtokengenerator.com/). You want to use the client ID to formulate the OAuth token.

4. **Run the Script**: 
    Open a terminal and navigate the script directory. 
    Run the script using the following command:
   ```bash
   python live_analysis.py
   ```
   Follow the terminal prompts to enable Twitch alerts and select your audio input device.

   Skip the interactive prompts by using command line arguments. [See Advanced Usage section below for details.](#advanced-usage)
5. **That's it!** The script should now be running and monitoring your audio input for glitches.

# Audio Input Device
Audio is different for everyone, so I'll explain my setup. I'm using voicemeeter to manage my audio devices. I've configured Open Broadcaster Software (OBS) to use one of voicemeeter's virtual inputs, namely Cable Input, as the monitoring device. For this, on OBS go to:
`Settings > Audio > Advanced > Monitoring Device > Cable Input (VB-Audio Virtual Cable)`

When you run the live_analysis script, it will list all available audio input devices. Select the one that corresponds to your monitoring device. For me this was listed as "CABLE Output (VB-Audio Virtual Cable)".

### Not Only For The Streamer
I'd like to point out that since this script is designed to monitor audio input, it can be used by users other than the streamer/broadcaster. For example, you could be the moderator of a channel and run this script to monitor the streamer's audio quality. Just be sure to isolate the audio input to the streamer's audio channel. Similiarly, you can test this script with previously recorded audio.

# Advanced Usage

## Command Line Arguments
You can skip the interactive prompts by providing command line arguments when running the script. 

Show available audio devices
```bash
python live_analysis.py --list-devices
```

Use specific audio device and disable Twitch alerts
```bash
python live_analysis.py --audio-device 2 --no-twitch
```

Enable Twitch alerts with specific audio device
```bash
python live_analysis.py --twitch --audio-device 1
```

### Command Line Options
- `--list-devices`: Lists all available audio input devices and exits.
- `--audio-device DEVICE_ID`: Specifies the audio input device to use by its ID.
- `--twitch`: Enables Twitch chat notifications.
- `--no-twitch`: Disables Twitch chat notifications.

## Alert Configuration
You can configure the alert settings in the `live_analysis.py` file. The following parameters can be adjusted:
```python
# Alert configuration
ALERT_CONFIG = {
    'detections_for_alert': 6,
    'alert_cooldown_ms': 60000,
    'detection_window_seconds': 90,
    'confidence_threshold': 70,
    'clean_audio_reset_seconds': 60,
}
```

- `detections_for_alert`: Number of detections needed to trigger an alert.
- `alert_cooldown_ms`: Minimum time in milliseconds between alerts.
- `detection_window_seconds`: After N seconds, the script will reset the detection count.
- `confidence_threshold`: Minimum confidence percentage for a detection to be counted.
- `clean_audio_reset_seconds`: Cosmetic use. After N seconds alert message will read as a new detection or ongoing detection.

## Thresholds for Detection Methods
You can also adjust the thresholds for each detection method in the `live_analysis.py` file.
These options are largely based on my own testing, otherwise untested.
```python
THRESHOLDS = {
    'silence_ratio': 0.60,         # Only flag if >60% silence (major dropouts)
    'amplitude_jump': 2.5,         # Much higher - only flag dramatic jumps
    'envelope_discontinuity': 2.0, # Higher threshold
    'gap_duration_ms': 100,        # Flag gaps longer than 100ms (significant dropouts)
    'min_audio_level': 0.005,      # Minimum RMS to even analyze
    'max_normal_gaps': 2,          # Max gaps allowed in normal audio
    'suspicious_gap_count': 4,     # Number of gaps that suggests real problems
}
```

# In Action
Running the script will look something like this in the terminal.
```
✅ Using device 1: CABLE Output (VB-Audio Virtual
🎧 Balanced Choppy Audio Detector with Twitch Integration
Focused on detecting streaming-related audio glitches
🔗 Attempting to connect to Twitch...
INFO:twitch_chat:Connecting to irc.chat.twitch.tv:6667...
INFO:twitch_chat:✅ Successfully connected to Twitch as beckyaberlin in #heroharmony
✅ Connected to Twitch chat
Press Ctrl+C to stop
 
Active detection methods:
  ✅ silence_gaps
  ❌ amplitude_jumps
  ✅ envelope_discontinuity
  ❌ temporal_consistency
  ❌ energy_variance
  ❌ zero_crossings
  ❌ spectral_rolloff
  ❌ spectral_centroid
 
Alert Configuration:
  🎯 Detections needed for alert: 6
  ⏱️  Detection window: 90s
  Alert cooldown: ~1.0 min
  📊 Confidence threshold: 70%
  🔄 Episode reset after: 60s clean audio
 
🎤 Listening for streaming audio glitches...
Building baseline audio profile...
```

Example of detecting glitching audio.
After 6 recent detections, the script will send a message to Twitch chat. Further detections will be ignored for 60000 ms due to the cooldown period.

```
[20:17:20.626] 🚨 STREAMING GLITCH DETECTED!
  Confidence: 80.0%
  Reasons: Audio envelope break detected
    envelope_discontinuity: Discontinuity score: 22.6
    📊 Recent detections: 5/6 (need 6 for alert)
 
[20:17:21.264] 🚨 STREAMING GLITCH DETECTED!
  Confidence: 80.0%
  Reasons: Audio envelope break detected
    envelope_discontinuity: Discontinuity score: 22.6
INFO:twitch_chat:💬 Sent: 🟠 MINOR Audio issues detected! 6 glitches in 1.5 minutes. Stream audio ma y be choppy! modCheck
📢 Twitch alert sent: 🟠 MINOR Audio issues detected! 6 glitches in 1.5 minutes. Stream audio may be  choppy! modCheck
```

# Detection Methods
You might have noticed that not all detection methods are enabled. This is because some methods are more prone to false positives, especially in dynamic audio environments like gaming or music streaming. The currently enabled methods have been selected for their balance between sensitivity and specificity in detecting streaming-related audio glitches.

You can toggle detection methods in the `live_analysis.py` file by modifying the `APPROACHES` dictionary. Here's a quick overview of the available detection methods:
```python
# Detection approaches - focused on streaming glitches
APPROACHES = {
    'silence_gaps': True,           # Dropouts/gaps from streaming
    'amplitude_jumps': False,       # Sudden level changes
    'envelope_discontinuity': True, # Audio cuts/breaks
    'temporal_consistency': False,  # inconsistent volume changes
    'energy_variance': False,       # changes in overall loudness
    'zero_crossings': False,        # how often waveform crosses zero
    'spectral_rolloff': False,      # frequency energy drop-off
    'spectral_centroid': False,     # center of mass of frequencies
}
```

In any case, lets review what each detection method does. The following is a big dump of AI generated explanations - so don't waste too much time reading it all! I'm putting it here for future reference.

## Silence Gaps

Purpose: Detects sudden gaps/dropouts in audio, like the sound cutting out briefly.

### How it works:
- Monitors for unusually long stretches of silence.
- Compares silence levels to your normal speaking volume (using RMS).

### Flags a problem if:
- Silence exceeds 60% of the window, or
- There are several short but noticeable silent gaps, or
- A single gap lasts over 200ms (which sounds like audio cutting in/out).

### Why it matters:
This is the most direct way to catch streamer mic cutouts or packet loss in audio.

## Envelope Discontinuity
Purpose: Detects abrupt breaks in how audio ramps up and down — a kind of "snap" or "jump" in audio levels that shouldn't happen.

### How it works:

- Breaks audio into small time windows (like a moving average of volume).
- Watches for a pattern like:
    - Normal level → Sudden drop to silence → Back to normal level
- If this happens quickly, it suggests a "glitchy blip" in the audio stream.

### Why it matters:

Great for catching tiny drops or robotic audio glitches that don’t register as pure silence but still sound broken.

## Amplitude Jumps (Disabled)
Purpose: Detects sudden big volume changes, like a loud pop or sudden drop.

### Why it’s disabled:
- It tends to false flag normal things like:
    - Pauses between words
    - Starting/stopping music
    - Excited yelling or gasps

So it’s not reliable for streaming glitch detection without more refinement.

## Temporal Consistency (Disabled)
Purpose: Tries to detect whether the volume changes over time are inconsistent with typical speech/music.

### Why it’s disabled:
Very sensitive — often flags natural speaking variations or expressive speech as "glitches". Too many false positives for regular streaming use.

## Energy Variance (Disabled)
Purpose: Measures how much the overall energy (loudness) changes, which can indicate inconsistency.

### Why it’s disabled:
Games and speech have naturally high variance, so it ends up detecting normal dynamics rather than real glitches.

## Zero Crossings (Disabled)
Purpose: Looks at how often the audio waveform crosses zero (changes from + to -). Glitchy noise sometimes causes weird patterns.

### Why it’s disabled:
Unreliable — doesn’t consistently correlate with audible streaming problems.

## Spectral Rolloff (Disabled)
Purpose: Measures the frequency at which most of the audio’s energy drops off — in simple terms, how “bright” or “dull” the sound is.

### Why it’s disabled:
Speech and gameplay have a wide range of frequencies, so this varies a lot naturally, making it bad for glitch detection.

## Spectral Centroid (Disabled)
Purpose: Tracks the "center of mass" of the audio’s frequencies — like where most of the tone is concentrated (low vs high).

### Why it’s disabled:
Again, too variable during normal gameplay or talking, so it doesn't help reliably detect glitches.

# Support
## Buy Me a Coffee
If this has helped you in any way, consider buying me a coffee! While I essentially built this with AI and played the role of architect and tester, I still had to put in time and effort, recording and analyzing audio samples, dealing with AI's hallucinations, and hopefully making this approachable by streamers from all walks of life.

Like other streamers I use StreamElements for my tipping platform.

[![Tip Me via StreamElements](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://streamelements.com/heroharmony/tip)

You can also show your support by following me on Twitch: [twitch.tv/heroharmony](https://www.twitch.tv/heroharmony)

## Shoutouts 
- Thank you to the Moblin folks for making IRL streaming much more approachable on iPhone.
- Cheers to the many IRL streamers who have inspired me, most whom don't know me:
    - [kiridane](https://www.twitch.tv/kiridane)
    - [Toronto Tech & Transit (Henry)](https://www.youtube.com/@TorontoTechTransitTV)
    - [Toxickbunny](https://www.twitch.tv/toxickbunny)
    - [Johnny Strides](https://www.youtube.com/@JohnnyStrides)
    - [CookSux](https://www.twitch.tv/cooksux)
    - [JinnyTTY](https://www.twitch.tv/jinnytty)
