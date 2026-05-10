#!/usr/bin/env python3
"""
Balanced Choppy Audio Detector
Focused on detecting streaming-related audio glitches while ignoring normal audio dynamics
"""

import numpy as np
import sounddevice as sd
import threading
import time
import time as time_module
import argparse
from collections import deque
from datetime import datetime
from queue import Empty, Full, Queue
import warnings
import logging
import math
import traceback
warnings.filterwarnings('ignore')

from choppy_detector_gui.alert_templates import AlertTemplates, severity_for_detection_count
from choppy_detector_gui.audio_devices import (
    get_hostapi_name,
    infer_obs_monitoring_device,
    list_audio_devices as list_all_audio_devices,
    list_input_devices,
)

# Import Twitch bot (assuming it's in the same directory)
try:
    from twitch_chat import TwitchBot
    TWITCH_AVAILABLE = True
except ImportError:
    print("[WARN] twitch_chat.py not found - running without Twitch integration")
    TWITCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_ALERT_COOLDOWN_MS = 60000


def resolve_alert_cooldown_ms():
    """Load optional ALERT_COOLDOWN_MS from config.py with safe fallback."""
    try:
        from config import ALERT_COOLDOWN_MS  # Optional setting
    except ImportError:
        return DEFAULT_ALERT_COOLDOWN_MS

    try:
        cooldown_ms = int(ALERT_COOLDOWN_MS)
    except (TypeError, ValueError):
        print(
            f"[WARN] Invalid ALERT_COOLDOWN_MS value ({ALERT_COOLDOWN_MS!r}); "
            f"using default {DEFAULT_ALERT_COOLDOWN_MS}"
        )
        return DEFAULT_ALERT_COOLDOWN_MS

    if cooldown_ms <= 0:
        print(
            f"[WARN] ALERT_COOLDOWN_MS must be > 0 ({cooldown_ms}); "
            f"using default {DEFAULT_ALERT_COOLDOWN_MS}"
        )
        return DEFAULT_ALERT_COOLDOWN_MS

    return cooldown_ms

# Configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 4096  # Larger chunks for stability
DEFAULT_PRODUCTION_WINDOW_MS = 1000
DEFAULT_PRODUCTION_STEP_MS = 50
BUFFER_DURATION = float(DEFAULT_PRODUCTION_WINDOW_MS) / 1000.0
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
DETECTION_LOOP_INTERVAL_SEC = float(DEFAULT_PRODUCTION_STEP_MS) / 1000.0


def production_window_ms() -> int:
    """Canonical live detection window used by runtime analysis."""
    return max(100, int(round(float(BUFFER_DURATION) * 1000.0)))


def production_step_ms() -> int:
    """Canonical live detection step used by runtime analysis."""
    return max(10, int(round(float(DETECTION_LOOP_INTERVAL_SEC) * 1000.0)))


def configure_production_timing(window_ms: int | float, step_ms: int | float) -> None:
    """Apply runtime production timing in one place for live detector behavior."""
    global BUFFER_DURATION, BUFFER_SIZE, DETECTION_LOOP_INTERVAL_SEC

    normalized_window_ms = max(100, int(round(float(window_ms or DEFAULT_PRODUCTION_WINDOW_MS))))
    normalized_step_ms = max(10, int(round(float(step_ms or DEFAULT_PRODUCTION_STEP_MS))))
    BUFFER_DURATION = normalized_window_ms / 1000.0
    DETECTION_LOOP_INTERVAL_SEC = normalized_step_ms / 1000.0
    BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Detection approaches - focused on streaming glitches
APPROACHES = {
    'silence_gaps': True,          # Dropouts/gaps from streaming
    'amplitude_jumps': False,       # Sudden level changes
    'envelope_discontinuity': True, # Audio cuts/breaks
    'amplitude_modulation': True,   # Fast flutter/"helicopter" texture
    'temporal_consistency': False,  # Disabled - too sensitive for normal audio
    'energy_variance': False,      # Disabled - normal music has high variance
    'zero_crossings': False,       # Disabled - not reliable for glitch detection
    'spectral_rolloff': False,     # Disabled - varies too much in normal audio
    'spectral_centroid': False,    # Disabled - music naturally varies
}

# Alert configuration
ALERT_CONFIG = {
    'detections_for_alert': 6,      # Number of detections needed to trigger alert
    'alert_cooldown_ms': resolve_alert_cooldown_ms(),  # Minimum time between alerts (milliseconds)
    'detection_window_seconds': 120, # Time window to count detections in
    'confidence_threshold': 70,     # Minimum confidence for counting detection
    'clean_audio_reset_seconds': 60, # Seconds of clean audio to reset episode
    'event_dedup_seconds': 0.9,     # Suppress duplicate hits from one burst
    'fast_alert_burst_detections': 4,    # Fast path for obvious glitches
    'fast_alert_window_seconds': 15,     # Time window for fast path
    'fast_alert_min_confidence': 75,     # Confidence required for fast path
    'log_possible_glitches': True,       # Show occasional low-confidence hints
    'possible_log_min_confidence': 0.70, # Only log stronger low-confidence hits
    'possible_log_interval_seconds': 10.0,# Throttle repeated "possible" logs
    'max_alert_age_seconds': 15.0,       # Drop stale queued alerts
    'max_alert_send_window_seconds': 8.0,# Give up on sending if retries exceed this window
    'twitch_send_failures_for_pause': 3, # Consecutive send failures before pausing
    'twitch_send_pause_seconds': 60.0,   # Circuit-breaker pause window
}

STREAM_START_WARMUP_IGNORE_SECONDS = 3.0
BASELINE_MIN_RMS_SAMPLES = 20
BASELINE_CLEAN_LOCK_SECONDS = 20.0

# More reasonable thresholds for streaming glitch detection
THRESHOLDS = {
    'silence_ratio': 0.60,         # Only flag if >60% silence (major dropouts)
    'amplitude_jump': 2.5,         # Much higher - only flag dramatic jumps
    'envelope_discontinuity': 2.0, # Higher threshold
    'modulation_freq_min_hz': 15.0,  # Ignore normal speech cadence (<15Hz)
    'modulation_freq_max_hz': 36.0,  # Focus on rapid glitchy flutter
    'modulation_strength': 8.5,      # Peak-vs-floor ratio in modulation band
    'modulation_depth': 0.42,        # Required envelope flutter depth
    'modulation_peak_concentration': 0.20,  # Require narrow, strong modulation peak
    'gap_duration_ms': 100,        # Flag gaps longer than 100ms (significant dropouts)
    'min_audio_level': 0.005,      # Minimum RMS to even analyze
    'max_normal_gaps': 2,          # Max gaps allowed in normal audio
    'suspicious_gap_count': 4,     # Number of gaps that suggests real problems
}

def list_audio_devices():
    """List available audio input devices"""
    input_devices = list_input_devices()
    
    print("\nAvailable audio input devices:")
    print("-" * 50)
    print("OBS Monitoring Device is an output/render device.")
    print("This script records from input/capture devices, so virtual cables often look inverted:")
    print("  OBS 'CABLE Input' -> script 'CABLE Output'")
    print()
    for device in input_devices:
        status = " (default)" if device.is_default else ""
        print(f"  {device.selection_index}: {device.name}{status}")
        print(f"     PortAudio Index: {device.portaudio_index}, Host API: {device.hostapi_name}")
        print(f"     Channels: {device.max_input_channels}, Sample Rate: {device.default_samplerate}")
        if device.obs_monitoring_hint:
            print(f"     OBS Monitoring Device likely appears as: {device.obs_monitoring_hint}")

    output_devices = [device for device in list_all_audio_devices(include_output_only=True) if not device.is_monitorable]
    if output_devices:
        print()
        print("Output/render devices shown for routing reference only:")
        for device in output_devices:
            status = " (default)" if device.is_default else ""
            print(f"  - {device.name}{status}")
            print(f"     PortAudio Index: {device.portaudio_index}, Host API: {device.hostapi_name}")
            print(f"     Output Channels: {device.max_output_channels}, Sample Rate: {device.default_samplerate}")
    
    return [(device.portaudio_index, {
        'name': device.name,
        'hostapi': None,
        'max_input_channels': device.max_input_channels,
        'default_samplerate': device.default_samplerate,
    }) for device in input_devices]

def select_audio_device(device_id=None):
    """Select audio device interactively or by ID"""
    input_devices = list_audio_devices()
    
    if not input_devices:
        print("[ERROR] No audio input devices found!")
        return None
    
    if device_id is not None:
        if 0 <= device_id < len(input_devices):
            selected = input_devices[device_id]
            print(f"\nUsing device {device_id}: {selected[1]['name']}")
            return selected[0]  # Return the actual device index
        else:
            print(f"[ERROR] Invalid device ID {device_id}. Must be 0-{len(input_devices)-1}")
            return None
    
    # Interactive selection
    print(f"\nEnter device number (0-{len(input_devices)-1}, default=0): ", end="")
    try:
        choice = input().strip()
        if choice == "":
            choice = 0
        else:
            choice = int(choice)
            
        if 0 <= choice < len(input_devices):
            selected = input_devices[choice]
            print(f"Selected: {selected[1]['name']}")
            return selected[0]  # Return the actual device index
        else:
            print("Invalid choice. Using default device.")
            return input_devices[0][0]
            
    except (ValueError, KeyboardInterrupt):
        print("Invalid input. Using default device.")
        return input_devices[0][0]

class BalancedChoppyDetector:
    def __init__(
        self,
        enable_twitch=True,
        audio_device=None,
        input_channel_index=0,
        twitch_channel=None,
        twitch_bot_username=None,
        twitch_oauth_token=None,
        event_callback=None,
        alert_templates=None,
        file_logger=None,
    ):
        self.volume_report_interval_sec = 10
        self._volume_report_started = False
        self._last_volume_report_time = 0.0
        self._last_audio_level_event_time = 0.0
        self.sample_rate = int(SAMPLE_RATE)
        self.stream_channels = 1
        self.device_info = None
        self.audio_buffer = deque(maxlen=int(self.sample_rate * BUFFER_DURATION))
        self.running = False
        self.lock = threading.Lock()
        self.audio_device = audio_device
        self.input_channel_index = max(0, int(input_channel_index or 0))
        self.twitch_channel = twitch_channel
        self.twitch_bot_username = twitch_bot_username
        self.twitch_oauth_token = twitch_oauth_token
        self.event_callback = event_callback
        self.alert_templates = alert_templates or AlertTemplates()
        self.file_logger = file_logger
        self.baseline_stats = {
            'rms_history': deque(maxlen=50),
            'established_baseline': False,
            'learning_started_at': 0.0,
            'last_blocked_at': 0.0,
        }
        
        # Twitch integration
        self.twitch_enabled = enable_twitch and TWITCH_AVAILABLE
        self.twitch_bot = None
        if self.twitch_enabled:
            self.twitch_bot = TwitchBot(
                channel=self.twitch_channel,
                username=self.twitch_bot_username,
                token=self.twitch_oauth_token,
            )
        self._audio_callback_started = False
            
        # Alert tracking - simplified
        self.detection_history = deque(maxlen=200)  # Store recent detections
        self.last_alert_time = 0
        self.total_detections = 0
        self.total_alerts_sent = 0
        self.last_clean_time = time.time()  # Track when audio was last clean
        self.current_episode_started = False  # Track if we're in a glitch episode
        self.last_detection_event_time = 0.0  # De-dup rapid repeats from same burst
        self.last_detection_signature = ""
        self.last_dedup_log_time = 0.0
        self.last_possible_glitch_log_time = 0.0
        self.last_possible_glitch_log_reason = ""
        self.recent_analysis_windows = 0
        self.recent_low_conf_hits = 0
        self.last_audio_callback_time = 0.0
        self.last_audio_callback_status = ""
        self.audio_callback_errors = 0
        self.last_stale_audio_warning_time = 0.0
        self.audio_stale_active = False
        self.audio_stale_started_at = 0.0
        self.stream_restart_requested = False
        self.stream_restart_requested_at = 0.0
        self.stream_restart_stale_seconds = 0.0
        self.detection_loop_errors = 0
        self.last_detection_error_time = 0.0
        self.last_detection_traceback = ""
        self.detection_thread = None
        self.stream_warmup_ignore_seconds = float(STREAM_START_WARMUP_IGNORE_SECONDS)
        self.stream_warmup_until = 0.0
        self.alert_queue = Queue(maxsize=32)
        self.alert_sender_thread = None
        self.twitch_connection_thread = None
        self.alert_queue_drops = 0
        self._pending_alert_by_key = {}
        self._twitch_send_fail_streak = 0
        self._twitch_send_paused_until = 0.0

    def emit_event(self, event_type, **payload):
        if self.event_callback:
            try:
                self.event_callback(event_type, payload)
            except Exception:
                pass

    def log_file_event(self, level, event, message="", **fields):
        if self.file_logger:
            try:
                self.file_logger.log(level, event, message, **fields)
            except Exception:
                pass

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def configure_audio_stream(self):
        """Use the selected device's native defaults when possible."""
        self.sample_rate = int(SAMPLE_RATE)
        self.stream_channels = 1
        self.device_info = None

        if self.audio_device is None:
            self.audio_buffer = deque(maxlen=int(self.sample_rate * BUFFER_DURATION))
            return

        try:
            device_info = sd.query_devices(self.audio_device)
            self.device_info = device_info

            default_samplerate = device_info.get('default_samplerate')
            if default_samplerate:
                self.sample_rate = int(round(float(default_samplerate)))

            max_input_channels = int(device_info.get('max_input_channels', 0) or 0)
            if max_input_channels <= 0:
                raise ValueError("Selected device reports no input channels")

            self.input_channel_index = min(self.input_channel_index, max_input_channels - 1)
            self.stream_channels = max(1, self.input_channel_index + 1)
            self.audio_buffer = deque(maxlen=int(self.sample_rate * BUFFER_DURATION))
        except Exception as e:
            print(
                f"[{self._timestamp()}] [WARN] Could not load selected device defaults: {e}. "
                f"Falling back to {SAMPLE_RATE} Hz mono."
            )
            self.sample_rate = int(SAMPLE_RATE)
            self.input_channel_index = 0
            self.stream_channels = 1
            self.audio_buffer = deque(maxlen=int(self.sample_rate * BUFFER_DURATION))

    def _start_detection_thread(self):
        if self.detection_thread and self.detection_thread.is_alive():
            return

        self.detection_thread = threading.Thread(
            target=self.detection_loop,
            name="detection-loop",
            daemon=True,
        )
        self.detection_thread.start()

    def _start_alert_sender(self):
        if not self.twitch_enabled or not self.twitch_bot:
            return

        if self.alert_sender_thread and self.alert_sender_thread.is_alive():
            return

        self.alert_sender_thread = threading.Thread(
            target=self.alert_sender_loop,
            name="twitch-alert-sender",
            daemon=True,
        )
        self.alert_sender_thread.start()

    def _start_twitch_connection_thread(self):
        if not self.twitch_enabled or not self.twitch_bot:
            return

        if self.twitch_connection_thread and self.twitch_connection_thread.is_alive():
            return

        self.twitch_connection_thread = threading.Thread(
            target=self.connect_to_twitch,
            name="twitch-connection",
            daemon=True,
        )
        self.twitch_connection_thread.start()

    def queue_twitch_alert(self, detection_count, time_span_minutes, is_first_alert):
        """Queue alert delivery so network work never blocks detection."""
        if not self.twitch_enabled or not self.twitch_bot:
            return False

        key = "first" if is_first_alert else "ongoing"
        now_ts = time_module.time()
        existing = self._pending_alert_by_key.get(key)
        if existing is not None:
            # Coalesce while pending: keep the newest/highest-severity snapshot.
            existing['detection_count'] = max(int(existing.get('detection_count', 0)), int(detection_count))
            existing['time_span_minutes'] = max(float(existing.get('time_span_minutes', 0.0)), float(time_span_minutes))
            existing['queued_at'] = now_ts
            self.emit_event(
                "alert.coalesced",
                key=key,
                detection_count=existing['detection_count'],
                time_span_minutes=existing['time_span_minutes'],
            )
            self.log_file_event(
                "info",
                "alert.coalesced",
                key=key,
                detection_count=existing['detection_count'],
                time_span_minutes=round(existing['time_span_minutes'], 2),
            )
            return True

        try:
            payload = {
                'detection_count': detection_count,
                'time_span_minutes': time_span_minutes,
                'is_first_alert': is_first_alert,
                'queued_at': now_ts,
                'coalesce_key': key,
            }
            self.alert_queue.put_nowait(payload)
            self._pending_alert_by_key[key] = payload
            return True
        except Full:
            self.alert_queue_drops += 1
            print(
                f"[{self._timestamp()}] [WARN] Twitch alert queue full; "
                f"dropping alert #{self.alert_queue_drops}"
            )
            return False

    def alert_sender_loop(self):
        """Deliver queued Twitch alerts in the background."""
        while self.running or not self.alert_queue.empty():
            try:
                payload = self.alert_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                coalesce_key = payload.get('coalesce_key')
                if coalesce_key:
                    pending = self._pending_alert_by_key.get(coalesce_key)
                    if pending is payload:
                        self._pending_alert_by_key.pop(coalesce_key, None)
                queued_at = float(payload.get('queued_at', time_module.time()))
                age_seconds = max(0.0, time_module.time() - queued_at)
                max_alert_age_seconds = float(ALERT_CONFIG.get('max_alert_age_seconds', 15.0))
                if age_seconds > max_alert_age_seconds:
                    print(
                        f"[{self._timestamp()}] [WARN] Dropping stale Twitch alert "
                        f"(age={age_seconds:.1f}s > {max_alert_age_seconds:.1f}s)"
                    )
                    self.emit_event(
                        "alert.dropped_stale",
                        age_seconds=round(age_seconds, 2),
                        max_age_seconds=max_alert_age_seconds,
                    )
                    self.log_file_event(
                        "warn",
                        "alert.dropped_stale",
                        age_seconds=round(age_seconds, 2),
                        max_age_seconds=max_alert_age_seconds,
                    )
                    continue
                self.send_twitch_alert(
                    payload['detection_count'],
                    payload['time_span_minutes'],
                    payload['is_first_alert'],
                    queued_at=queued_at,
                )
            except Exception as e:
                print(f"[{self._timestamp()}] [ERROR] Alert sender failure: {e}")
                print(traceback.format_exc().rstrip())
            finally:
                self.alert_queue.task_done()
        
    def establish_baseline(self, audio):
        """Learn what normal audio levels look like"""
        if self.baseline_stats['established_baseline']:
            # Freeze baseline after initial lock to avoid long-session drift.
            return
        now = time_module.time()
        high_conf_recent_cutoff = now - BASELINE_CLEAN_LOCK_SECONDS
        recent_high_conf = any(
            det['timestamp'] > high_conf_recent_cutoff
            and float(det.get('confidence', 0.0) or 0.0) >= float(ALERT_CONFIG['confidence_threshold'])
            for det in self.detection_history
        )
        if recent_high_conf:
            if self.baseline_stats['rms_history']:
                self.baseline_stats['rms_history'].clear()
                self.baseline_stats['learning_started_at'] = 0.0
                if (now - float(self.baseline_stats.get('last_blocked_at') or 0.0)) >= 10.0:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        "Baseline learning reset due to recent high-confidence detection."
                    )
                    self.emit_event("baseline.learning_reset", reason="recent_high_conf_detection")
                    self.log_file_event("info", "baseline.learning_reset", reason="recent_high_conf_detection")
                    self.baseline_stats['last_blocked_at'] = now
            return
        rms = np.sqrt(np.mean(audio**2))
        if rms > THRESHOLDS['min_audio_level']:
            self.baseline_stats['rms_history'].append(rms)
            if not self.baseline_stats['learning_started_at']:
                self.baseline_stats['learning_started_at'] = now

        # Check if we now have enough samples to lock in a baseline
        learning_elapsed = (
            now - float(self.baseline_stats['learning_started_at'])
            if self.baseline_stats['learning_started_at']
            else 0.0
        )
        if (
            not self.baseline_stats['established_baseline']
            and len(self.baseline_stats['rms_history']) >= BASELINE_MIN_RMS_SAMPLES
            and learning_elapsed >= BASELINE_CLEAN_LOCK_SECONDS
        ):
            self.baseline_stats['established_baseline'] = True
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Baseline audio profile established "
                f"({BASELINE_CLEAN_LOCK_SECONDS:.0f}s clean window)."
            )
            self.emit_event(
                "baseline.established",
                sample_count=len(self.baseline_stats['rms_history']),
                clean_window_seconds=BASELINE_CLEAN_LOCK_SECONDS,
            )
            self.log_file_event(
                "info",
                "baseline.established",
                sample_count=len(self.baseline_stats['rms_history']),
                clean_window_seconds=BASELINE_CLEAN_LOCK_SECONDS,
            )
            
    def get_baseline_rms(self):
        """Get typical RMS level"""
        if len(self.baseline_stats['rms_history']) < 5:
            return 0.1  # Default assumption
        return np.median(list(self.baseline_stats['rms_history']))

    def rebuild_baseline(self, *, reason: str = "manual") -> None:
        """Reset baseline learning so a fresh clean-gated profile can be learned."""
        with self.lock:
            self.baseline_stats['rms_history'].clear()
            self.baseline_stats['established_baseline'] = False
            self.baseline_stats['learning_started_at'] = 0.0
            self.baseline_stats['last_blocked_at'] = 0.0
            self.stream_warmup_until = time_module.time() + self.stream_warmup_ignore_seconds
        print(
            f"[{self._timestamp()}] Baseline rebuild requested ({reason}). "
            f"Learning will relock after {BASELINE_CLEAN_LOCK_SECONDS:.0f}s clean audio."
        )
        self.emit_event(
            "baseline.rebuild_requested",
            reason=reason,
            clean_window_seconds=BASELINE_CLEAN_LOCK_SECONDS,
        )
        self.log_file_event(
            "info",
            "baseline.rebuild_requested",
            reason=reason,
            clean_window_seconds=BASELINE_CLEAN_LOCK_SECONDS,
        )
    
    def _format_volume(self, rms: float) -> str:
        """Format RMS as both raw value and dBFS (assuming audio is -1..1 float)."""
        dbfs = 20.0 * math.log10(rms + 1e-12)
        return f"RMS={rms:.6f} ({dbfs:.1f} dBFS)"

    def silence_gaps_detector(self, audio):
        """Detect actual dropouts/gaps in streaming audio (not normal speech pauses)"""
        if len(audio) == 0:
            return False, 0.0, []
            
        rms = np.sqrt(np.mean(audio**2))
        
        # Use adaptive threshold based on recent audio levels
        baseline_rms = self.get_baseline_rms()
        silence_threshold = max(0.002, baseline_rms * 0.03)  # 3% of typical level (stricter)
        
        silent_samples = np.abs(audio) < silence_threshold
        silence_ratio = np.sum(silent_samples) / len(audio)
        
        # Find actual gaps (consecutive silent samples)
        gaps = self.find_silence_gaps(audio, silence_threshold)
        significant_gaps = [gap for gap in gaps if gap['duration_ms'] > THRESHOLDS['gap_duration_ms']]
        
        # More sophisticated detection logic
        has_many_gaps = len(significant_gaps) >= THRESHOLDS['suspicious_gap_count']
        has_very_long_gaps = any(gap['duration_ms'] > 200 for gap in significant_gaps)
        extreme_silence = silence_ratio > THRESHOLDS['silence_ratio']
        
        # Only flag if we have clear signs of streaming problems
        choppy = (has_many_gaps or has_very_long_gaps or extreme_silence) and rms > THRESHOLDS['min_audio_level']
        
        return choppy, silence_ratio, significant_gaps

    def find_silence_gaps(self, audio, threshold):
        """Find gaps/dropouts in audio with duration info"""
        silent = np.abs(audio) < threshold
        
        # Find runs of silence
        diff = np.diff(np.concatenate(([False], silent, [False])).astype(int))
        gap_starts = np.where(diff == 1)[0]
        gap_ends = np.where(diff == -1)[0]
        
        gaps = []
        for start, end in zip(gap_starts, gap_ends):
            duration_samples = end - start
            duration_ms = (duration_samples / self.sample_rate) * 1000
            gaps.append({
                'start': start,
                'end': end,
                'duration_ms': duration_ms
            })
        
        return gaps

    def amplitude_jumps_detector(self, audio):
        """Detect sudden dramatic amplitude changes (not normal dynamics)"""
        if len(audio) < 2048:
            return False, 0.0, []
            
        # Use larger windows to avoid flagging normal audio dynamics
        window_size = 2048
        step = window_size // 2
        amplitudes = []
        
        for i in range(0, len(audio) - window_size, step):
            window = audio[i:i + window_size]
            amp = np.sqrt(np.mean(window**2))
            amplitudes.append(amp)
        
        if len(amplitudes) < 3:
            return False, 0.0, []
            
        amplitudes = np.array(amplitudes)
        
        # Filter out very quiet sections
        baseline_rms = self.get_baseline_rms()
        valid_amps = amplitudes[amplitudes > baseline_rms * 0.1]
        
        if len(valid_amps) < 2:
            return False, 0.0, []
        
        # Look for dramatic jumps relative to the audio's own levels
        median_amp = np.median(valid_amps)
        if median_amp == 0:
            return False, 0.0, []
            
        # Calculate relative jumps
        amp_ratios = amplitudes / (median_amp + 1e-10)
        
        # Find dramatic jumps up or down
        jumps = []
        for i in range(1, len(amp_ratios)):
            if amplitudes[i-1] > THRESHOLDS['min_audio_level'] or amplitudes[i] > THRESHOLDS['min_audio_level']:
                jump_ratio = abs(amp_ratios[i] - amp_ratios[i-1])
                if jump_ratio > THRESHOLDS['amplitude_jump']:
                    jumps.append({
                        'position': i,
                        'ratio': jump_ratio,
                        'from_amp': amplitudes[i-1],
                        'to_amp': amplitudes[i]
                    })
        
        max_jump = max([j['ratio'] for j in jumps], default=0.0)
        choppy = len(jumps) > 0
        
        return choppy, max_jump, jumps

    def envelope_discontinuity_detector(self, audio):
        """Detect sudden breaks/cuts in audio envelope"""
        if len(audio) < 2048:
            return False, 0.0
            
        # Calculate envelope using moving RMS
        window_size = 256
        envelope = []
        
        for i in range(0, len(audio) - window_size, window_size // 4):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            envelope.append(rms)
        
        if len(envelope) < 4:
            return False, 0.0
        
        envelope = np.array(envelope)
        
        # Only analyze if we have meaningful audio levels
        baseline_rms = self.get_baseline_rms()
        if np.max(envelope) < baseline_rms * 0.1:
            return False, 0.0
        
        # Look for sudden drops to near-silence followed by audio resuming
        discontinuities = []
        
        for i in range(1, len(envelope) - 1):
            prev_amp = envelope[i-1]
            curr_amp = envelope[i]
            next_amp = envelope[i+1]
            
            # Check for drop-to-silence-then-resume pattern
            if (prev_amp > baseline_rms * 0.2 and 
                curr_amp < baseline_rms * 0.05 and 
                next_amp > baseline_rms * 0.2):
                
                drop_ratio = prev_amp / (curr_amp + 1e-10)
                if drop_ratio > THRESHOLDS['envelope_discontinuity']:
                    discontinuities.append(drop_ratio)
        
        max_discontinuity = max(discontinuities, default=0.0)
        choppy = len(discontinuities) > 0
        
        return choppy, max_discontinuity

    def amplitude_modulation_detector(self, audio):
        """Detect rapid envelope flutter often heard as robotic/helicopter artifacts."""
        if len(audio) < 4096:
            return False, 0.0, 0.0, 0.0, 0.0

        baseline_rms = self.get_baseline_rms()
        rms = np.sqrt(np.mean(audio**2))
        if rms < max(THRESHOLDS['min_audio_level'], baseline_rms * 0.12):
            return False, 0.0, 0.0, 0.0, 0.0

        # Build a smooth amplitude envelope.
        window_size = 512
        hop = 128
        envelope = []
        for i in range(0, len(audio) - window_size, hop):
            window = audio[i:i + window_size]
            envelope.append(np.sqrt(np.mean(window**2)))

        if len(envelope) < 16:
            return False, 0.0, 0.0, 0.0, 0.0

        envelope = np.array(envelope, dtype=np.float64)
        env_p10 = float(np.percentile(envelope, 10))
        env_p90 = float(np.percentile(envelope, 90))
        modulation_depth = (env_p90 - env_p10) / (env_p90 + 1e-10)

        centered = envelope - np.mean(envelope)
        if np.max(np.abs(centered)) < 1e-8:
            return False, 0.0, 0.0, modulation_depth, 0.0

        env_sr = self.sample_rate / hop
        spectrum = np.abs(np.fft.rfft(centered))
        freqs = np.fft.rfftfreq(len(centered), d=1.0 / env_sr)

        band_mask = (
            (freqs >= THRESHOLDS['modulation_freq_min_hz']) &
            (freqs <= THRESHOLDS['modulation_freq_max_hz'])
        )
        if not np.any(band_mask):
            return False, 0.0, 0.0, modulation_depth, 0.0

        band_power = spectrum[band_mask]
        band_freqs = freqs[band_mask]
        peak_idx = int(np.argmax(band_power))
        peak_power = float(band_power[peak_idx])
        peak_freq_hz = float(band_freqs[peak_idx])
        floor_power = float(np.median(band_power) + 1e-10)
        modulation_strength = peak_power / floor_power
        band_total_power = float(np.sum(band_power) + 1e-10)
        peak_concentration = peak_power / band_total_power

        choppy = (
            modulation_strength >= THRESHOLDS['modulation_strength'] and
            modulation_depth >= THRESHOLDS['modulation_depth'] and
            peak_concentration >= THRESHOLDS['modulation_peak_concentration']
        )
        return choppy, modulation_strength, peak_freq_hz, modulation_depth, peak_concentration

    def analyze_audio(self, audio):
        """Run enabled detection methods with focus on streaming glitches"""
        results = {}
        
        # Skip analysis if audio is too quiet
        rms = np.sqrt(np.mean(audio**2))
        if rms < THRESHOLDS['min_audio_level']:
            return {}
        
        # Update baseline stats
        self.establish_baseline(audio)
        
        # Run enabled methods
        if APPROACHES['silence_gaps']:
            choppy, score, gaps = self.silence_gaps_detector(audio)
            # In longer windows, bursty glitches can be diluted. Also test
            # half-windows and keep the strongest silence-gap signal.
            if len(audio) >= int(self.sample_rate * 1.5):
                half = len(audio) // 2
                for chunk in (audio[:half], audio[half:]):
                    if len(chunk) < max(256, int(self.sample_rate * 0.5)):
                        continue
                    sub_choppy, sub_score, sub_gaps = self.silence_gaps_detector(chunk)
                    current_max_gap = max((g["duration_ms"] for g in gaps), default=0.0)
                    sub_max_gap = max((g["duration_ms"] for g in sub_gaps), default=0.0)
                    if (
                        (sub_choppy and not choppy)
                        or (sub_score > score + 0.03)
                        or (sub_max_gap > current_max_gap + 40.0)
                        or (len(sub_gaps) > len(gaps) and sub_score >= score)
                    ):
                        choppy, score, gaps = sub_choppy, sub_score, sub_gaps
            results['silence_gaps'] = {
                'choppy': choppy, 
                'score': score, 
                'gaps': gaps,
                'description': f"Silence ratio: {score:.1%}, {len(gaps)} gaps detected"
            }
            
        if APPROACHES['amplitude_jumps']:
            choppy, score, jumps = self.amplitude_jumps_detector(audio)
            results['amplitude_jumps'] = {
                'choppy': choppy, 
                'score': score, 
                'jumps': jumps,
                'description': f"Max jump: {score:.1f}x, {len(jumps)} jumps detected"
            }
            
        if APPROACHES['envelope_discontinuity']:
            choppy, score = self.envelope_discontinuity_detector(audio)
            results['envelope_discontinuity'] = {
                'choppy': choppy, 
                'score': score,
                'description': f"Discontinuity score: {score:.1f}"
            }

        if APPROACHES['amplitude_modulation']:
            choppy, score, peak_freq_hz, depth, peak_concentration = self.amplitude_modulation_detector(audio)
            results['amplitude_modulation'] = {
                'choppy': choppy,
                'score': score,
                'peak_freq_hz': peak_freq_hz,
                'depth': depth,
                'peak_concentration': peak_concentration,
                'description': (
                    f"Envelope modulation: {score:.1f}x at {peak_freq_hz:.1f}Hz, "
                    f"depth={depth:.2f}, conc={peak_concentration:.2f}"
                )
            }
        
        return results

    def assess_glitch_confidence(self, results):
        """Assess confidence that this is a real streaming glitch"""
        if not results:
            return 0.0, "No analysis data"
        
        confidence = 0.0
        reasons = []
        
        # Check each detection type
        for method, result in results.items():
            if result['choppy']:
                if method == 'silence_gaps':
                    # More strict evaluation for silence gaps
                    gap_count = len(result.get('gaps', []))
                    silence_ratio = result['score']
                    window_ms = int(getattr(self, "_analysis_window_ms", production_window_ms()) or production_window_ms())
                    max_gap_ms = max(
                        (float(gap.get('duration_ms', 0.0)) for gap in result.get('gaps', [])),
                        default=0.0,
                    )
                    mod = results.get('amplitude_modulation', {}) if isinstance(results, dict) else {}
                    mod_strength = float(mod.get('score', 0.0) or 0.0)
                    mod_depth = float(mod.get('depth', 0.0) or 0.0)
                    mod_peak_concentration = float(mod.get('peak_concentration', 0.0) or 0.0)
                    mod_corroborated_strong = (
                        mod_strength >= 5.0 and
                        mod_depth >= 0.55 and
                        mod_peak_concentration >= 0.20
                    )
                    mod_corroborated_medium = (
                        mod_strength >= 4.6 and
                        mod_depth >= 0.50 and
                        mod_peak_concentration >= 0.095
                    )
                    
                    if silence_ratio > 0.8:  # Extreme silence
                        if gap_count >= THRESHOLDS['suspicious_gap_count'] or mod_corroborated_strong:
                            if (
                                window_ms >= 1600
                                and not mod_corroborated_strong
                                and max_gap_ms < 500.0
                            ):
                                confidence += 0.72
                                reasons.append(
                                    f"High silence in long window ({silence_ratio:.1%}) without strong corroboration"
                                )
                            else:
                                confidence += 0.9
                                reasons.append(f"Extreme silence ({silence_ratio:.1%})")
                        else:
                            confidence += 0.70
                            reasons.append(f"High silence without corroboration ({silence_ratio:.1%})")
                    elif max_gap_ms >= 500.0:
                        if gap_count >= THRESHOLDS['suspicious_gap_count'] or mod_corroborated_strong:
                            confidence += 0.85
                            reasons.append(f"Severe long audio gap ({max_gap_ms:.0f}ms)")
                        elif mod_corroborated_medium and silence_ratio <= 0.80:
                            confidence += 0.76
                            reasons.append(
                                f"Long audio gap ({max_gap_ms:.0f}ms) corroborated by modulation texture"
                            )
                        else:
                            confidence += 0.70
                            reasons.append(f"Isolated long audio gap ({max_gap_ms:.0f}ms)")
                    elif gap_count >= THRESHOLDS['suspicious_gap_count']:
                        if (
                            window_ms >= 1600
                            and max_gap_ms < 350.0
                            and not mod_corroborated_strong
                        ):
                            confidence += 0.70
                            reasons.append(f"{gap_count} significant gaps in long window")
                        elif max_gap_ms >= 250.0 or mod_corroborated_strong:
                            confidence += 0.8
                            reasons.append(f"{gap_count} significant gaps detected")
                        else:
                            confidence += 0.72
                            reasons.append(f"{gap_count} short-to-medium gaps detected")
                    elif max_gap_ms >= 250.0:
                        confidence += 0.62
                        reasons.append(f"Very long audio gap ({max_gap_ms:.0f}ms)")
                    elif max_gap_ms > 200.0:
                        confidence += 0.55
                        reasons.append("Long audio gap detected")
                    elif silence_ratio >= (THRESHOLDS['silence_ratio'] + 0.03):
                        if mod_corroborated_strong:
                            confidence += 0.75
                            reasons.append(
                                f"Sustained high silence ({silence_ratio:.1%}) with modulation stress"
                            )
                        elif mod_corroborated_medium and silence_ratio <= 0.75:
                            confidence += 0.75
                            reasons.append(
                                f"Sustained high silence ({silence_ratio:.1%}) with mild modulation corroboration"
                            )
                        else:
                            confidence += 0.58
                            reasons.append(f"Sustained high silence ({silence_ratio:.1%})")
                    elif gap_count > 0 and silence_ratio > 0.4:
                        confidence += 0.5
                        reasons.append(f"{gap_count} gaps with high silence")
                    else:
                        confidence += 0.3
                        reasons.append("Possible audio gaps")
                        
                elif method == 'amplitude_jumps':
                    # Medium confidence - could be normal dynamics
                    jump_count = len(result.get('jumps', []))
                    if result['score'] > 5.0:  # Very dramatic jump
                        confidence += 0.6
                        reasons.append(f"Dramatic amplitude jump ({result['score']:.1f}x)")
                    elif jump_count > 2:
                        confidence += 0.4
                        reasons.append(f"Multiple amplitude jumps")
                    else:
                        confidence += 0.2
                        reasons.append("Amplitude instability")
                        
                elif method == 'envelope_discontinuity':
                    # Envelope breaks are an important glitch signal in your samples.
                    # Keep them meaningful, while still below the old aggressive levels.
                    if result['score'] > 12.0:
                        confidence += 0.78
                        reasons.append("Strong audio envelope break detected")
                    elif result['score'] > 3.0:
                        confidence += 0.68
                        reasons.append("Audio envelope break detected")
                    else:
                        confidence += 0.58
                        reasons.append("Envelope discontinuity")
                elif method == 'amplitude_modulation':
                    # Very conservative: supporting evidence only.
                    if (
                        result['score'] > 11.0 and
                        result.get('depth', 0.0) > 0.50 and
                        result.get('peak_concentration', 0.0) > 0.24
                    ):
                        confidence += 0.2
                        reasons.append(
                            f"Rapid modulation texture ({result.get('peak_freq_hz', 0.0):.1f}Hz)"
                        )
                    elif (
                        result['score'] > 9.5 and
                        result.get('depth', 0.0) > 0.45 and
                        result.get('peak_concentration', 0.0) > 0.22
                    ):
                        confidence += 0.1
                        reasons.append("Possible modulation texture")
        
        # Boost confidence if multiple methods agree
        strong_detections = sum(
            1
            for name in ('silence_gaps', 'envelope_discontinuity')
            if results.get(name, {}).get('choppy', False)
        )
        if strong_detections >= 2:
            confidence = min(confidence * 1.3, 1.0)
            reasons.append("Multiple primary detection methods agree")

        strong_method_hit = any(
            results.get(name, {}).get('choppy', False)
            for name in ('silence_gaps', 'envelope_discontinuity')
        )
        if results.get('amplitude_modulation', {}).get('choppy', False) and strong_method_hit:
            confidence = min(confidence + 0.10, 1.0)
            reasons.append("Modulation pattern corroborates primary detector")

        # Long-window guardrail: prevent silence-only scoring in 2000ms windows
        # from reaching high-confidence without corroboration.
        silence_result = results.get('silence_gaps', {}) if isinstance(results, dict) else {}
        if isinstance(silence_result, dict) and silence_result.get('choppy', False):
            window_ms = int(getattr(self, "_analysis_window_ms", production_window_ms()) or production_window_ms())
            envelope_hit = bool(results.get('envelope_discontinuity', {}).get('choppy', False))
            mod = results.get('amplitude_modulation', {}) if isinstance(results, dict) else {}
            mod_strength = float(mod.get('score', 0.0) or 0.0)
            mod_depth = float(mod.get('depth', 0.0) or 0.0)
            mod_peak_concentration = float(mod.get('peak_concentration', 0.0) or 0.0)
            mod_medium = (
                mod_strength >= 4.6 and
                mod_depth >= 0.50 and
                mod_peak_concentration >= 0.095
            )
            if window_ms >= 1600 and not envelope_hit and not mod_medium:
                if confidence > 0.72:
                    confidence = 0.72
                    reasons.append("Long-window silence-only guardrail applied")

        return min(confidence, 1.0), "; ".join(reasons)
        
    def connect_to_twitch(self):
        """Initialize Twitch connection"""
        if not self.twitch_enabled or not self.twitch_bot:
            return False
            
        try:
            self.emit_event(
                "twitch.connecting",
                channel=getattr(self.twitch_bot, "channel", ""),
                username=getattr(self.twitch_bot, "username", ""),
            )
            self.log_file_event(
                "info",
                "twitch.connecting",
                channel=getattr(self.twitch_bot, "channel", ""),
                username=getattr(self.twitch_bot, "username", ""),
            )
            if self.twitch_bot.connect(deadline_monotonic=time_module.monotonic() + 20.0):
                print("Connected to Twitch chat")
                self.emit_event("twitch.connected")
                self.log_file_event("info", "twitch.connected")
                return True
            else:
                print("[ERROR] Failed to connect to Twitch chat")
                error = getattr(self.twitch_bot, "last_error", "") or "Unknown Twitch connection failure"
                response = getattr(self.twitch_bot, "last_response", "")
                self.emit_event("twitch.connection_failed", error=error, response=response)
                self.log_file_event("error", "twitch.connection_failed", error=error, response=response)
                return False
        except Exception as e:
            print(f"[ERROR] Twitch connection error: {e}")
            self.emit_event("twitch.connection_error", error=str(e))
            self.log_file_event("error", "twitch.connection_error", error=str(e))
            return False

    def send_twitch_alert(self, detection_count, time_span_minutes, is_first_alert, queued_at=None):
        """Send alert to Twitch chat"""
        if not self.twitch_enabled or not self.twitch_bot:
            return False

        try:
            now_ts = time_module.time()
            if now_ts < self._twitch_send_paused_until:
                remaining = self._twitch_send_paused_until - now_ts
                self.emit_event("twitch.send_paused", seconds_remaining=round(remaining, 2))
                self.log_file_event(
                    "warn",
                    "twitch.send_paused",
                    seconds_remaining=round(remaining, 2),
                )
                return False

            device_name = (
                self.device_info.get('name')
                if isinstance(self.device_info, dict)
                else str(self.audio_device)
            )
            message = self.alert_templates.render(
                detection_count=detection_count,
                time_span_minutes=time_span_minutes,
                is_first_alert=is_first_alert,
                confidence_threshold=ALERT_CONFIG['confidence_threshold'],
                device_name=device_name,
            )

            if queued_at is not None:
                age_seconds = max(0.0, time_module.time() - float(queued_at))
                max_alert_age_seconds = float(ALERT_CONFIG.get('max_alert_age_seconds', 15.0))
                if age_seconds > max_alert_age_seconds:
                    print(
                        f"[{self._timestamp()}] [WARN] Skipping stale Twitch alert before send "
                        f"(age={age_seconds:.1f}s > {max_alert_age_seconds:.1f}s)"
                    )
                    self.emit_event(
                        "alert.dropped_stale",
                        age_seconds=round(age_seconds, 2),
                        max_age_seconds=max_alert_age_seconds,
                    )
                    self.log_file_event(
                        "warn",
                        "alert.dropped_stale",
                        age_seconds=round(age_seconds, 2),
                        max_age_seconds=max_alert_age_seconds,
                    )
                    return False

            # Send to chat with a hard cutoff so stale alerts do not linger in retries.
            max_alert_send_window_seconds = float(ALERT_CONFIG.get('max_alert_send_window_seconds', 8.0))
            success = self.twitch_bot.send_message(
                message,
                max_total_seconds=max_alert_send_window_seconds,
            )
            if success:
                self._twitch_send_fail_streak = 0
                if self._twitch_send_paused_until > 0:
                    self._twitch_send_paused_until = 0.0
                    self.emit_event("twitch.send_resumed")
                    self.log_file_event("info", "twitch.send_resumed")
                self.total_alerts_sent += 1
                print(f"Twitch alert sent: {message}")
                self.emit_event("alert.sent", message=message)
                self.log_file_event("info", "alert.sent", message=message)
            else:
                self._twitch_send_fail_streak += 1
                print("[ERROR] Failed to send Twitch alert")
                self.emit_event("alert.failed", message=message)
                self.log_file_event("error", "alert.failed", message=message)
                fail_limit = int(ALERT_CONFIG.get('twitch_send_failures_for_pause', 3))
                pause_seconds = float(ALERT_CONFIG.get('twitch_send_pause_seconds', 60.0))
                if self._twitch_send_fail_streak >= fail_limit:
                    self._twitch_send_paused_until = time_module.time() + pause_seconds
                    self.emit_event(
                        "twitch.send_circuit_open",
                        fail_streak=self._twitch_send_fail_streak,
                        pause_seconds=pause_seconds,
                    )
                    self.log_file_event(
                        "warn",
                        "twitch.send_circuit_open",
                        fail_streak=self._twitch_send_fail_streak,
                        pause_seconds=pause_seconds,
                    )
                    self._twitch_send_fail_streak = 0

            return success

        except Exception as e:
            self._twitch_send_fail_streak += 1
            print(f"[ERROR] Error sending Twitch alert: {e}")
            self.emit_event("alert.error", error=str(e))
            self.log_file_event("error", "alert.error", error=str(e))
            return False

    def should_send_alert(self):
        """Check if we should send a Twitch alert based on recent detections"""
        current_time = time.time()

        # Check cooldown
        time_since_last_alert = current_time - self.last_alert_time
        cooldown_seconds = ALERT_CONFIG['alert_cooldown_ms'] / 1000.0
        if time_since_last_alert < cooldown_seconds:
            cooldown_remaining = cooldown_seconds - time_since_last_alert
            return False, 0, 0, cooldown_remaining, "cooldown", 0

        # Count recent high-confidence detections
        cutoff_time = current_time - ALERT_CONFIG['detection_window_seconds']
        recent_detections = [
            det for det in self.detection_history
            if det['timestamp'] > cutoff_time and det['confidence'] >= ALERT_CONFIG['confidence_threshold']
        ]

        detection_count = len(recent_detections)
        time_span_minutes = ALERT_CONFIG['detection_window_seconds'] / 60

        if detection_count >= ALERT_CONFIG['detections_for_alert']:
            return True, detection_count, time_span_minutes, 0, "standard_window", 0

        # Fast-burst path: trigger quickly when several strong primary detections
        # happen in a short period.
        burst_cutoff_time = current_time - ALERT_CONFIG['fast_alert_window_seconds']
        burst_detections = [
            det for det in self.detection_history
            if (
                det['timestamp'] > burst_cutoff_time and
                det['confidence'] >= ALERT_CONFIG['fast_alert_min_confidence'] and
                det.get('primary_hit', False)
            )
        ]
        burst_count = len(burst_detections)
        burst_time_span_minutes = ALERT_CONFIG['fast_alert_window_seconds'] / 60
        if burst_count >= ALERT_CONFIG['fast_alert_burst_detections']:
            return True, burst_count, burst_time_span_minutes, 0, "fast_burst", burst_count

        return False, detection_count, time_span_minutes, 0, "threshold_pending", burst_count

    def record_detection(self, confidence, reasons, primary_hit):
        """Record a detection for alert tracking"""
        self.detection_history.append({
            'timestamp': time.time(),
            'confidence': confidence,
            'reasons': reasons,
            'primary_hit': primary_hit,
        })
        self.total_detections += 1

    def check_episode_reset(self, current_time):
        """Check if we should reset the current episode - use same logic as detection counter"""
        if not self.current_episode_started:
            return
            
        # Use the EXACT same logic as the detection counter
        cutoff_time = current_time - ALERT_CONFIG['clean_audio_reset_seconds']
        recent_detections = [
            det for det in self.detection_history
            if det['timestamp'] > cutoff_time and det['confidence'] >= ALERT_CONFIG['confidence_threshold']
        ]
        
        # If no recent high-confidence detections, reset episode
        if len(recent_detections) == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No high-confidence detections in last {ALERT_CONFIG['clean_audio_reset_seconds']}s - episode reset")
            self.current_episode_started = False

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        try:
            if status:
                self.last_audio_callback_status = str(status)
                print(f"[{self._timestamp()}] Audio callback status: {status}")

            with self.lock:
                if indata.ndim > 1:
                    channel_idx = min(self.input_channel_index, indata.shape[1] - 1)
                    audio_data = indata[:, channel_idx]
                else:
                    audio_data = indata
                self.audio_buffer.extend(audio_data)

            current_time = time_module.time()
            self.last_audio_callback_time = current_time
            if self.audio_stale_active:
                stale_seconds = max(0.0, current_time - self.audio_stale_started_at)
                self.audio_stale_active = False
                self.audio_stale_started_at = 0.0
                self.last_stale_audio_warning_time = current_time
                self.stream_restart_requested = False
                self.stream_restart_requested_at = 0.0
                self.stream_restart_stale_seconds = 0.0
                print(f"[{self._timestamp()}] Audio callback recovered after {stale_seconds:.1f}s")
                self.emit_event("audio.recovered", seconds=stale_seconds)
                self.log_file_event("info", "audio.recovered", seconds=f"{stale_seconds:.1f}", device=self.audio_device)
            if not self._audio_callback_started:
                self._audio_callback_started = True
                self.emit_event("audio.callback_started")
                self.log_file_event("info", "audio.callback_started", device=self.audio_device)
            if current_time - self._last_audio_level_event_time >= 0.05:
                rms_level = float(np.sqrt(np.mean(audio_data**2)))
                peak_level = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
                rms_dbfs = 20.0 * math.log10(rms_level + 1e-12)
                peak_dbfs = 20.0 * math.log10(peak_level + 1e-12)
                self.emit_event(
                    "audio.level",
                    rms=rms_level,
                    peak=peak_level,
                    rms_dbfs=rms_dbfs,
                    peak_dbfs=peak_dbfs,
                    dbfs=rms_dbfs,
                )
                self._last_audio_level_event_time = current_time
        except Exception as e:
            self.audio_callback_errors += 1
            print(f"[{self._timestamp()}] [ERROR] Audio callback failure #{self.audio_callback_errors}: {e}")
            print(traceback.format_exc().rstrip())
            self.emit_event("audio.callback_error", error=str(e), count=self.audio_callback_errors)
            self.log_file_event("error", "audio.callback_error", error=str(e), count=self.audio_callback_errors)

    def detection_loop(self):
        """Main detection loop with reasonable timing"""
        last_detection_time = 0
        detection_cooldown = 0.5  # Don't spam detections
        silence_persistence_hits = deque(maxlen=32)
        burst_cluster_hits = deque(maxlen=64)
        long_window_sparse_burst_hits = deque(maxlen=64)
        
        print("Listening for streaming audio glitches...")
        print("Building baseline audio profile...")
        
        while self.running:
            try:
                time.sleep(DETECTION_LOOP_INTERVAL_SEC)

                current_time = time_module.time()
                if current_time - last_detection_time < detection_cooldown:
                    continue

                with self.lock:
                    if len(self.audio_buffer) < self.sample_rate:  # Need at least 1 second
                        continue

                    audio = np.array(list(self.audio_buffer), dtype=np.float32)

                time_since_callback = (
                    current_time - self.last_audio_callback_time
                    if self.last_audio_callback_time > 0
                    else None
                )

                # --- Volume heartbeat ---
                rms_level = float(np.sqrt(np.mean(audio**2)))

                if not self._volume_report_started:
                    print(
                        f"[{self._timestamp()}] Detection running. "
                        f"{self._format_volume(rms_level)} | min_analyze={THRESHOLDS['min_audio_level']} "
                        f"| high_conf_total={self.total_detections}"
                    )
                    self._volume_report_started = True
                    self._last_volume_report_time = current_time

                elif (current_time - self._last_volume_report_time) >= self.volume_report_interval_sec:
                    status = "OK" if rms_level >= THRESHOLDS['min_audio_level'] else "quiet"
                    freshness = (
                        f"live={time_since_callback:.1f}s"
                        if time_since_callback is not None
                        else "live=unknown"
                    )
                    print(
                        f"[{self._timestamp()}] Audio level: "
                        f"{self._format_volume(rms_level)} [{status}] "
                        f"| {freshness} "
                        f"| windows={self.recent_analysis_windows} "
                        f"| low_conf={self.recent_low_conf_hits} "
                        f"| high_conf_total={self.total_detections}"
                    )
                    self._last_volume_report_time = current_time
                    self.recent_analysis_windows = 0
                    self.recent_low_conf_hits = 0

                stale_warn_threshold_sec = 5.0
                stale_restart_threshold_sec = 12.0
                stale_warn_repeat_sec = 10.0
                if (
                    time_since_callback is not None and
                    time_since_callback >= stale_warn_threshold_sec and
                    (current_time - self.last_stale_audio_warning_time) >= stale_warn_repeat_sec
                ):
                    device_label = (
                        self.device_info.get('name')
                        if isinstance(self.device_info, dict)
                        else self.audio_device
                    )
                    if not self.audio_stale_active:
                        self.audio_stale_active = True
                        self.audio_stale_started_at = current_time - time_since_callback
                    print(
                        f"[{self._timestamp()}] [WARN] No fresh audio callback data for "
                        f"{time_since_callback:.1f}s; stream may be stalled. "
                        f"Device={device_label!r}, format={self.sample_rate}Hz/{self.stream_channels}ch"
                    )
                    self.emit_event(
                        "audio.stale",
                        seconds=time_since_callback,
                        device=str(device_label),
                    )
                    self.log_file_event(
                        "warn",
                        "audio.stale",
                        seconds=f"{time_since_callback:.1f}",
                        device=device_label,
                    )
                    self.last_stale_audio_warning_time = current_time
                if (
                    time_since_callback is not None
                    and time_since_callback >= stale_restart_threshold_sec
                    and not self.stream_restart_requested
                ):
                    self.stream_restart_requested = True
                    self.stream_restart_requested_at = current_time
                    self.stream_restart_stale_seconds = float(time_since_callback)
                    print(
                        f"[{self._timestamp()}] [WARN] Audio callback stale for {time_since_callback:.1f}s; "
                        "requesting input stream restart."
                    )
                    self.emit_event(
                        "audio.stream_restarting",
                        reason="stale_callback",
                        seconds=time_since_callback,
                    )
                    self.log_file_event(
                        "warn",
                        "audio.stream_restarting",
                        reason="stale_callback",
                        seconds=f"{time_since_callback:.1f}",
                        device=self.audio_device,
                    )

                if self.stream_restart_requested:
                    # Wait for stream recycle instead of scoring stale/frozen audio.
                    continue
                if current_time < self.stream_warmup_until:
                    # Ignore startup/reopen transients for a short period.
                    continue

                # Analyze audio
                results = self.analyze_audio(audio)

                # Assess confidence
                self._analysis_window_ms = int(round((len(audio) / float(self.sample_rate or 1)) * 1000.0))
                confidence, reasons = self.assess_glitch_confidence(results)
                silence_result = results.get('silence_gaps', {}) if isinstance(results, dict) else {}
                silence_choppy = bool(silence_result.get('choppy', False))
                silence_ratio = float(silence_result.get('score', 0.0) or 0.0)
                silence_gaps = silence_result.get('gaps', []) if isinstance(silence_result, dict) else []
                silence_gap_count = len(silence_gaps)
                silence_max_gap_ms = max(
                    (float(gap.get('duration_ms', 0.0) or 0.0) for gap in silence_gaps),
                    default=0.0,
                )
                mod_result = results.get('amplitude_modulation', {}) if isinstance(results, dict) else {}
                mod_strength = float(mod_result.get('score', 0.0) or 0.0)
                mod_depth = float(mod_result.get('depth', 0.0) or 0.0)
                mod_peak_concentration = float(mod_result.get('peak_concentration', 0.0) or 0.0)
                envelope_hit = bool(results.get('envelope_discontinuity', {}).get('choppy', False))
                modulation_hit = bool(results.get('amplitude_modulation', {}).get('choppy', False))
                persistence_promoted = False
                severe_silence_ratio = max(float(THRESHOLDS['silence_ratio']) + 0.08, 0.68)
                severe_gap_pattern = (
                    silence_ratio >= severe_silence_ratio
                    and silence_max_gap_ms >= 250.0
                    and silence_gap_count >= 2
                )
                extreme_gap_pattern = silence_ratio >= 0.85 and silence_max_gap_ms >= 500.0
                modulation_gap_pattern = (
                    silence_max_gap_ms >= 500.0
                    and float(results.get('amplitude_modulation', {}).get('score', 0.0) or 0.0) >= 4.8
                    and float(results.get('amplitude_modulation', {}).get('depth', 0.0) or 0.0) >= 0.50
                    and float(results.get('amplitude_modulation', {}).get('peak_concentration', 0.0) or 0.0) >= 0.16
                )
                persistence_block_long_window = (
                    int(getattr(self, "_analysis_window_ms", production_window_ms()) or production_window_ms()) >= 1600
                    and not modulation_hit
                    and (
                        silence_ratio >= 0.88 and silence_gap_count >= 2
                        or confidence >= 0.70
                    )
                )
                persistence_candidate = (
                    silence_choppy
                    and not envelope_hit
                    and 0.55 <= confidence < 0.75
                    and not persistence_block_long_window
                    and (severe_gap_pattern or extreme_gap_pattern or modulation_gap_pattern)
                )
                while silence_persistence_hits and (current_time - silence_persistence_hits[0]) > 2.5:
                    silence_persistence_hits.popleft()
                if persistence_candidate:
                    if not silence_persistence_hits or (current_time - silence_persistence_hits[-1]) >= 0.45:
                        silence_persistence_hits.append(current_time)
                if len(silence_persistence_hits) >= 3:
                    if modulation_hit:
                        confidence = max(confidence, 0.80)
                        reasons = f"{reasons}; Persistent severe silence-gap pattern corroborated by modulation"
                    else:
                        confidence = max(confidence, 0.78)
                        reasons = f"{reasons}; Persistent severe silence-gap pattern"
                    persistence_promoted = True

                # Cluster lift: repeated near-threshold burst windows (common in real glitch bursts).
                burst_candidate = (
                    silence_choppy
                    and not envelope_hit
                    and 0.64 <= confidence < 0.75
                    and 0.52 <= silence_ratio <= 0.85
                    and silence_gap_count <= 2
                    and mod_strength >= 4.3
                    and mod_depth >= 0.45
                )
                while burst_cluster_hits and (current_time - burst_cluster_hits[0]) > 2.2:
                    burst_cluster_hits.popleft()
                if burst_candidate:
                    if not burst_cluster_hits or (current_time - burst_cluster_hits[-1]) >= 0.12:
                        burst_cluster_hits.append(current_time)
                if len(burst_cluster_hits) >= 3:
                    confidence = max(confidence, 0.77)
                    reasons = f"{reasons}; Repeated near-threshold burst cluster"

                # 2000ms micro-lift for sparse-gap burst clusters.
                # Target: recover long-window bursty glitches (Sample 7 style) without
                # reintroducing long-window speech false positives.
                long_window_sparse_candidate = (
                    int(getattr(self, "_analysis_window_ms", production_window_ms()) or production_window_ms()) >= 1600
                    and silence_choppy
                    and not envelope_hit
                    and 0.66 <= confidence < 0.75
                    and 0.52 <= silence_ratio <= 0.85
                    and silence_max_gap_ms >= 600.0
                    and mod_strength >= 4.0
                    and mod_depth >= 0.45
                    and mod_peak_concentration >= 0.05
                )
                while long_window_sparse_burst_hits and (current_time - long_window_sparse_burst_hits[0]) > 2.8:
                    long_window_sparse_burst_hits.popleft()
                if long_window_sparse_candidate:
                    if (
                        not long_window_sparse_burst_hits
                        or (current_time - long_window_sparse_burst_hits[-1]) >= 0.15
                    ):
                        long_window_sparse_burst_hits.append(current_time)
                if len(long_window_sparse_burst_hits) >= 2:
                    confidence = max(confidence, 0.76)
                    reasons = f"{reasons}; Long-window sparse-gap burst cluster"

                self.recent_analysis_windows += 1
                if results and 0 < confidence < 0.75:
                    self.recent_low_conf_hits += 1

                if not results or confidence < 0.75:
                    if (
                        results and
                        confidence > 0 and
                        ALERT_CONFIG['log_possible_glitches'] and
                        confidence >= ALERT_CONFIG['possible_log_min_confidence']
                    ):
                        reason_key = reasons
                        should_log = (
                            reason_key != self.last_possible_glitch_log_reason or
                            (current_time - self.last_possible_glitch_log_time) >= ALERT_CONFIG['possible_log_interval_seconds']
                        )
                        if should_log:
                            print(f"[{self._timestamp()}] Possible glitch (confidence: {confidence:.1%}) - {reasons}")
                            self.last_possible_glitch_log_time = current_time
                            self.last_possible_glitch_log_reason = reason_key
                    continue

                # De-duplicate repeated windows from the same glitch burst.
                active_methods = tuple(sorted(method for method, result in results.items() if result.get('choppy')))
                if persistence_promoted:
                    active_methods = tuple(sorted(set(active_methods) | {"silence_gaps_persistent"}))
                current_signature = "|".join(active_methods)
                time_since_event = current_time - self.last_detection_event_time
                if (
                    time_since_event < ALERT_CONFIG['event_dedup_seconds'] and
                    current_signature == self.last_detection_signature
                ):
                    if current_time - self.last_dedup_log_time >= 5:
                        remaining = ALERT_CONFIG['event_dedup_seconds'] - time_since_event
                        print(f"    Duplicate glitch window suppressed ({remaining:.1f}s dedup remaining)")
                        self.last_dedup_log_time = current_time
                    last_detection_time = current_time
                    continue
                self.last_detection_event_time = current_time
                self.last_detection_signature = current_signature

                # Check for episode reset before recording the new detection
                self.check_episode_reset(current_time)

                primary_hit = any(
                    results.get(name, {}).get('choppy', False)
                    for name in ('silence_gaps', 'envelope_discontinuity')
                )
                self.record_detection(confidence * 100, reasons, primary_hit)

                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n[{timestamp}] STREAMING GLITCH DETECTED!")
                print(f"  Confidence: {confidence:.1%}")
                print(f"  Reasons: {reasons}")
                device_name = (
                    self.device_info.get('name')
                    if isinstance(self.device_info, dict)
                    else str(self.audio_device)
                )
                active_methods = [
                    method for method, result in results.items() if result.get('choppy')
                ]
                if persistence_promoted:
                    active_methods = sorted(set(active_methods) | {"silence_gaps_persistent"})
                should_alert, detection_count, time_span, cooldown_remaining, alert_path, burst_count = self.should_send_alert()
                self.emit_event(
                    "glitch.detected",
                    confidence=round(confidence * 100, 1),
                    reasons=reasons,
                    methods=active_methods,
                    device=device_name,
                    detection_count=detection_count,
                    severity=severity_for_detection_count(int(detection_count)).strip("[]").lower(),
                )
                self.log_file_event(
                    "warn",
                    "glitch.detected",
                    confidence=round(confidence * 100, 1),
                    reasons=reasons,
                    methods=",".join(active_methods),
                    device=device_name,
                )

                for method, result in results.items():
                    if result['choppy']:
                        print(f"    {method}: {result.get('description', result['score'])}")

                if should_alert:
                    if alert_path == "standard_window":
                        print(
                            f"    Alert path: standard window "
                            f"({detection_count}/{ALERT_CONFIG['detections_for_alert']} in "
                            f"{ALERT_CONFIG['detection_window_seconds']}s)"
                        )
                    elif alert_path == "fast_burst":
                        print(
                            f"    Alert path: fast burst "
                            f"({burst_count}/{ALERT_CONFIG['fast_alert_burst_detections']} in "
                            f"{ALERT_CONFIG['fast_alert_window_seconds']}s, "
                            f">= {ALERT_CONFIG['fast_alert_min_confidence']}%)"
                        )
                    is_first_alert = not self.current_episode_started
                    if self.queue_twitch_alert(detection_count, time_span, is_first_alert):
                        self.last_alert_time = current_time
                        self.current_episode_started = True
                        print("    Twitch alert queued")
                        self.emit_event(
                            "alert.queued",
                            detection_count=detection_count,
                            time_span_minutes=time_span,
                            is_first_alert=is_first_alert,
                        )
                        self.log_file_event(
                            "info",
                            "alert.queued",
                            detection_count=detection_count,
                            time_span_minutes=f"{time_span:.1f}",
                            is_first_alert=is_first_alert,
                        )
                else:
                    if cooldown_remaining > 0:
                        cooldown_minutes = cooldown_remaining / 60.0
                        print(f"    Alert cooldown active (~{cooldown_minutes:.1f} min remaining)")
                    else:
                        print(
                            f"    Recent detections: {detection_count}/{ALERT_CONFIG['detections_for_alert']} "
                            f"(need {ALERT_CONFIG['detections_for_alert']} for alert)"
                        )
                        print(
                            f"    Fast-burst detections: {burst_count}/{ALERT_CONFIG['fast_alert_burst_detections']} "
                            f"(window {ALERT_CONFIG['fast_alert_window_seconds']}s)"
                        )

                last_detection_time = current_time
            except Exception as e:
                self.detection_loop_errors += 1
                self.last_detection_error_time = time_module.time()
                self.last_detection_traceback = traceback.format_exc()
                print(f"[{self._timestamp()}] [ERROR] Detection loop failure #{self.detection_loop_errors}: {e}")
                print(self.last_detection_traceback.rstrip())
                self.emit_event(
                    "detector.error",
                    error=str(e),
                    count=self.detection_loop_errors,
                )
                self.log_file_event(
                    "error",
                    "detector.error",
                    error=str(e),
                    count=self.detection_loop_errors,
                )
                time.sleep(1.0)

    def start_detection(self):
        """Start balanced detection with optional Twitch integration"""
        print("Balanced Choppy Audio Detector with Twitch Integration")
        print("Focused on detecting streaming-related audio glitches")
        self.emit_event("monitoring.starting")
        self.log_file_event("info", "monitoring.starting", device=self.audio_device)
        
        if self.twitch_enabled:
            print("Attempting to connect to Twitch in background...")
            self._start_twitch_connection_thread()
        else:
            print("Running without Twitch integration")
            
        print("Press Ctrl+C to stop")
        print()
        
        print("Active detection methods:")
        for method, active in APPROACHES.items():
            print(f"  [{'ON' if active else 'OFF'}] {method}")
        print()
        
        print("Alert Configuration:")
        print(f"  Detections needed for alert: {ALERT_CONFIG['detections_for_alert']}")
        print(f"  Detection window: {ALERT_CONFIG['detection_window_seconds']}s")
        print(
            f"  Fast-burst alert: {ALERT_CONFIG['fast_alert_burst_detections']} detections in "
            f"{ALERT_CONFIG['fast_alert_window_seconds']}s at >= {ALERT_CONFIG['fast_alert_min_confidence']}%"
        )
        print(f"  Alert cooldown: ~{ALERT_CONFIG['alert_cooldown_ms'] / 60000.0:.1f} min")
        print(f"  Confidence threshold: {ALERT_CONFIG['confidence_threshold']}%")
        print(f"  De-dup window: {ALERT_CONFIG['event_dedup_seconds']}s")
        print(f"  Episode reset after: {ALERT_CONFIG['clean_audio_reset_seconds']}s clean audio")
        print(f"  Stream-start warm-up ignore: {self.stream_warmup_ignore_seconds:.1f}s")
        print(f"  Log low-confidence possible glitches: {ALERT_CONFIG['log_possible_glitches']}")
        print()

        self.configure_audio_stream()
        if self.device_info is not None:
            hostapi_name = get_hostapi_name(self.device_info.get('hostapi'))
            print("Selected Audio Device:")
            print(f"  Name: {self.device_info.get('name')}")
            print(f"  Host API: {hostapi_name}")
            print(f"  Stream Format: {self.sample_rate} Hz, {self.stream_channels} channel")
            print(
                f"  Device Capabilities: {self.device_info.get('max_input_channels')} input channels, "
                f"default {self.device_info.get('default_samplerate')} Hz"
            )
            print()
        
        self.running = True
        self.last_audio_callback_time = time_module.time()

        self._start_detection_thread()
        self._start_alert_sender()

        try:
            while self.running:
                try:
                    with sd.InputStream(
                        device=self.audio_device,
                        samplerate=self.sample_rate,
                        channels=self.stream_channels,
                        blocksize=CHUNK_SIZE,
                        callback=self.audio_callback
                    ):
                        print(f"[{self._timestamp()}] Audio input stream opened")
                        self.stream_warmup_until = time_module.time() + self.stream_warmup_ignore_seconds
                        print(
                            f"[{self._timestamp()}] Ignoring detections for "
                            f"{self.stream_warmup_ignore_seconds:.1f}s after stream open."
                        )
                        self.emit_event(
                            "audio.stream_opened",
                            device=self.device_info.get('name') if isinstance(self.device_info, dict) else self.audio_device,
                            sample_rate=self.sample_rate,
                            channels=self.stream_channels,
                            channel_index=self.input_channel_index,
                        )
                        self.log_file_event("info", "audio.stream_opened", device=self.audio_device)
                        while self.running:
                            if self.stream_restart_requested:
                                stale_seconds = float(self.stream_restart_stale_seconds or 0.0)
                                print(
                                    f"[{self._timestamp()}] Restarting audio input stream after "
                                    f"{stale_seconds:.1f}s stale callback data."
                                )
                                self.log_file_event(
                                    "warn",
                                    "audio.stream_restart_requested",
                                    reason="stale_callback",
                                    stale_seconds=f"{stale_seconds:.1f}",
                                    device=self.audio_device,
                                )
                                with self.lock:
                                    self.audio_buffer = deque(maxlen=int(self.sample_rate * BUFFER_DURATION))
                                self.stream_restart_requested = False
                                self.stream_restart_requested_at = 0.0
                                self.stream_restart_stale_seconds = 0.0
                                break
                            if self.detection_thread and not self.detection_thread.is_alive():
                                print(f"[{self._timestamp()}] [WARN] Detection loop thread stopped; restarting")
                                self._start_detection_thread()
                            time.sleep(0.5)
                except KeyboardInterrupt:
                    print("\n\nStopping detection...")
                    self.running = False
                except Exception as e:
                    print(f"[{self._timestamp()}] [ERROR] Audio input stream failure: {e}")
                    print(traceback.format_exc().rstrip())
                    self.emit_event("audio.stream_error", error=str(e))
                    self.log_file_event("error", "audio.stream_error", error=str(e), device=self.audio_device)
                    if not self.running:
                        break
                    print(f"[{self._timestamp()}] Restarting audio input stream in 2 seconds...")
                    time.sleep(2.0)
        finally:
            self.running = False
            if self.alert_sender_thread and self.alert_sender_thread.is_alive():
                self.alert_sender_thread.join(timeout=3.0)
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=3.0)

            print("\nFinal Statistics:")
            print(f"  Total detections: {self.total_detections}")
            print(f"  Alerts sent: {self.total_alerts_sent}")
            print(f"  Detection loop errors: {self.detection_loop_errors}")
            print(f"  Audio callback errors: {self.audio_callback_errors}")

            if self.twitch_bot:
                self.twitch_bot.disconnect()
            self.emit_event("monitoring.stopped")
            self.log_file_event("info", "monitoring.stopped")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Balanced Choppy Audio Detector with Twitch Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detector.py                           # Interactive mode
  python detector.py --no-twitch              # Disable Twitch alerts
  python detector.py --audio-device 2         # Use audio device #2
  python detector.py --twitch --audio-device 1 # Enable Twitch, use device #1
        """
    )
    
    parser.add_argument(
        '--twitch', 
        action='store_true', 
        help='Enable Twitch chat alerts'
    )
    
    parser.add_argument(
        '--no-twitch', 
        action='store_true', 
        help='Disable Twitch chat alerts'
    )
    
    parser.add_argument(
        '--audio-device', 
        type=int, 
        metavar='N',
        help='Audio device number to use (see --list-devices)'
    )

    parser.add_argument(
        '--audio-channel',
        '--channel',
        dest='audio_channel',
        type=int,
        metavar='N',
        help='Input audio channel index to analyze (default: 0)'
    )
    
    parser.add_argument(
        '--list-devices', 
        action='store_true', 
        help='List available audio devices and exit'
    )

    parser.add_argument(
        '--twitch-channel',
        type=str,
        help='Twitch channel name (without #)'
    )

    parser.add_argument(
        '--twitch-bot-username',
        type=str,
        help='Twitch bot username override'
    )

    parser.add_argument(
        '--twitch-oauth-token',
        type=str,
        help='Twitch OAuth token override'
    )
    
    return parser.parse_args()

def main():
    print("Balanced Choppy Audio Detector with Twitch Integration")
    print("=" * 60)
    print("Designed to detect streaming audio glitches and alert Twitch chat")
    print()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle --list-devices
    if args.list_devices:
        list_audio_devices()
        return
    
    # Validate conflicting arguments
    if args.twitch and args.no_twitch:
        print("[ERROR] Cannot specify both --twitch and --no-twitch")
        return
    
    # Determine Twitch setting
    enable_twitch = None
    if args.twitch:
        enable_twitch = True
    elif args.no_twitch:
        enable_twitch = False
    else:
        # Interactive mode - ask user
        if TWITCH_AVAILABLE:
            twitch_choice = input("Enable Twitch chat alerts? (y/n, default=y): ").strip().lower()
            enable_twitch = twitch_choice != 'n'
        else:
            enable_twitch = False
            print("[WARN] Twitch integration not available (missing twitch_chat.py or config)")
    
    # Determine audio device
    audio_device = None
    if args.audio_device is not None:
        audio_device = select_audio_device(args.audio_device)
        if audio_device is None:
            return  # Error already printed
    else:
        # Interactive mode - ask user
        audio_device = select_audio_device()
        if audio_device is None:
            return
    
    detector = BalancedChoppyDetector(
        enable_twitch=enable_twitch,
        audio_device=audio_device,
        input_channel_index=max(0, int(args.audio_channel or 0)),
        twitch_channel=(str(args.twitch_channel).strip().lstrip("#") if args.twitch_channel else None),
        twitch_bot_username=(str(args.twitch_bot_username).strip() if args.twitch_bot_username else None),
        twitch_oauth_token=(str(args.twitch_oauth_token).strip() if args.twitch_oauth_token else None),
    )
    
    try:
        detector.start_detection()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Detection stopped.")

if __name__ == "__main__":
    main()
