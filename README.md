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
   pip install -r requirements.txt
   ```
3. **Configure Twitch Integration**: If you want to enable Twitch chat notifications, you'll need to set up the config.py file with your Twitch credentials. 
    ```python
    TWITCH_CHANNEL = "HeroHarmony"
    TWITCH_BOT_USERNAME = "HeroBot"
    TWITCH_OAUTH_TOKEN = "oauth:your_token_here"
    ALERT_COOLDOWN_MS = 60000  # Optional
    ```
   Replace the placeholders with your actual Twitch channel name, bot username, and the bot's OAuth token. You can generate an OAuth token from [Twitch Token Generator](https://twitchtokengenerator.com/). You want to use the client ID to formulate the OAuth token.

4. **Run the Script**: 
    Open a terminal and navigate the script directory. 
    Run the script using the following command:
   ```bash
   python live_analysis.py
   ```
   Follow the terminal prompts to enable Twitch alerts and select your audio input device.

   Skip the interactive prompts by using command line arguments. See the [Technical Reference](docs/technical-reference.md) for details.
5. **That's it!** The script should now be running and monitoring your audio input for glitches.

## GUI

A desktop GUI entry point is available.

Quick start (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app_gui.py
```

The original `live_analysis.py` script remains supported. The GUI uses the same detector runtime and adds tabs for monitoring controls, response templates, settings, and console-style runtime output.

If your virtual environment is already created, you can just run:

```bash
source .venv/bin/activate
python app_gui.py
```

### One-Click Launchers

You can also use launcher scripts that auto-create `.venv`, install requirements, and run the GUI:

- Windows: double-click `run_gui.bat`
- macOS: double-click `run_gui.command`

From terminal:

```bash
./run_gui.command
```

Windows command prompt:

```bat
run_gui.bat
```

### Packaged Builds (No Terminal Window)

Prebuilt desktop packages are available from the repository Releases page.

These packaged builds are standalone for normal use (no `.py` files required beside them).  
Settings are stored in the OS user config location:
- Windows: `%APPDATA%\ChoppyAudioDetector\settings.json`
- macOS: `~/Library/Application Support/ChoppyAudioDetector/settings.json`

Need to build from source? See [Build and Packaging Guide](docs/build-from-source.md).

### GUI Command Line Options

You can launch the GUI with command line options similar to `live_analysis.py`:

```bash
python app_gui.py --list-devices
python app_gui.py --audio-device 2 --twitch
python app_gui.py --audio-device 2 --audio-channel 1
python app_gui.py --no-twitch
```

- `--list-devices`: Lists available audio input devices and exits.
- `--audio-device N`: Selects audio input device by GUI index.
- `--audio-channel N`: Selects input audio channel index and auto-starts monitoring.
- `--twitch`: Enables Twitch alerts.
- `--no-twitch`: Disables Twitch alerts.
- `--twitch-channel NAME`: Sets Twitch channel (without `#`).
- `--twitch-bot-username USER`: Sets Twitch bot username.
- `--twitch-oauth-token TOKEN`: Sets Twitch OAuth token.

Compared to `live_analysis.py`, the GUI supports all shared flags (`--list-devices`, `--audio-device`, `--twitch`, `--no-twitch`) plus:
- `--audio-channel N` (legacy alias: `--channel N`)
- `--twitch-channel NAME`
- `--twitch-bot-username USER`
- `--twitch-oauth-token TOKEN`

`live_analysis.py` now supports these same options for CLI/GUI parity.

# Audio Input Device
Audio is different for everyone, so I'll explain my setup. I'm using voicemeeter to manage my audio devices. I've configured Open Broadcaster Software (OBS) to use one of voicemeeter's virtual inputs, namely Cable Input, as the monitoring device. For this, on OBS go to:
`Settings > Audio > Advanced > Monitoring Device > Cable Input (VB-Audio Virtual Cable)`

When you run the live_analysis script, it will list available audio input/capture devices, not OBS render devices. That means the names can look inverted for virtual cables:

- OBS Monitoring Device: `CABLE Input (VB-Audio Virtual Cable)`
- Script capture device: `CABLE Output (VB-Audio Virtual Cable)`

Those are the two ends of the same virtual cable. If you monitor to a device that does not expose a paired capture endpoint, the script will not be able to read it directly. In that case, route OBS monitoring through a virtual cable or another device pair that provides an input/capture side.

### Not Only For The Streamer
I'd like to point out that since this script is designed to monitor audio input, it can be used by users other than the streamer/broadcaster. For example, you could be the moderator of a channel and run this script to monitor the streamer's audio quality. Just be sure to isolate the audio input to the streamer's audio channel. Similiarly, you can test this script with previously recorded audio.

# Advanced and Technical Reference

For deeper technical usage and tuning details, see:

- [Technical Reference](docs/technical-reference.md)
- [Build and Packaging Guide](docs/build-from-source.md)

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
