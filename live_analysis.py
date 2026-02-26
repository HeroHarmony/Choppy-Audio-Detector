#!/usr/bin/env python3
"""
Balanced Choppy Audio Detector
Focused on detecting streaming-related audio glitches while ignoring normal audio dynamics
"""

import numpy as np
import sounddevice as sd
import threading
import time
import argparse
from collections import deque
from datetime import datetime
import warnings
import logging
import math
warnings.filterwarnings('ignore')

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
BUFFER_DURATION = 2.0  # Longer buffer for better analysis
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
    'detection_window_seconds': 90, # Time window to count detections in
    'confidence_threshold': 70,     # Minimum confidence for counting detection
    'clean_audio_reset_seconds': 60, # Seconds of clean audio to reset episode
    'event_dedup_seconds': 0.9,     # Suppress duplicate hits from one burst
    'fast_alert_burst_detections': 3,    # Fast path for obvious glitches
    'fast_alert_window_seconds': 15,     # Time window for fast path
    'fast_alert_min_confidence': 75,     # Confidence required for fast path
    'log_possible_glitches': True,       # Show occasional low-confidence hints
    'possible_log_min_confidence': 0.70, # Only log stronger low-confidence hits
    'possible_log_interval_seconds': 10.0,# Throttle repeated "possible" logs
}

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
    devices = sd.query_devices()
    input_devices = []
    
    print("\nAvailable audio input devices:")
    print("-" * 50)
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            status = " (default)" if i == sd.default.device[0] else ""
            print(f"  {len(input_devices)-1}: {device['name']}{status}")
            print(f"     Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
    
    return input_devices

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
    def __init__(self, enable_twitch=True, audio_device=None):
        self.volume_report_interval_sec = 10
        self._volume_report_started = False
        self._last_volume_report_time = 0.0
        self.audio_buffer = deque(maxlen=BUFFER_SIZE)
        self.running = False
        self.lock = threading.Lock()
        self.audio_device = audio_device
        self.baseline_stats = {
            'rms_history': deque(maxlen=50),
            'established_baseline': False
        }
        
        # Twitch integration
        self.twitch_enabled = enable_twitch and TWITCH_AVAILABLE
        self.twitch_bot = None
        if self.twitch_enabled:
            self.twitch_bot = TwitchBot()
            
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
        
    def establish_baseline(self, audio):
        """Learn what normal audio levels look like"""
        rms = np.sqrt(np.mean(audio**2))
        if rms > THRESHOLDS['min_audio_level']:
            self.baseline_stats['rms_history'].append(rms)

        # Check if we now have enough samples to lock in a baseline
        if (not self.baseline_stats['established_baseline'] 
            and len(self.baseline_stats['rms_history']) >= 20):
            self.baseline_stats['established_baseline'] = True
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Baseline audio profile established.")
            
    def get_baseline_rms(self):
        """Get typical RMS level"""
        if len(self.baseline_stats['rms_history']) < 5:
            return 0.1  # Default assumption
        return np.median(list(self.baseline_stats['rms_history']))
    
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
            duration_ms = (duration_samples / SAMPLE_RATE) * 1000
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

        env_sr = SAMPLE_RATE / hop
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
                    
                    if silence_ratio > 0.8:  # Extreme silence
                        confidence += 0.9
                        reasons.append(f"Extreme silence ({silence_ratio:.1%})")
                    elif gap_count >= THRESHOLDS['suspicious_gap_count']:
                        confidence += 0.8
                        reasons.append(f"{gap_count} significant gaps detected")
                    elif any(gap['duration_ms'] > 200 for gap in result.get('gaps', [])):
                        confidence += 0.6
                        reasons.append("Long audio gap detected")
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
                    # High confidence - envelope breaks are suspicious
                    if result['score'] > 3.0:
                        confidence += 0.8
                        reasons.append("Audio envelope break detected")
                    else:
                        confidence += 0.6
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

        return min(confidence, 1.0), "; ".join(reasons)
        
    def connect_to_twitch(self):
        """Initialize Twitch connection"""
        if not self.twitch_enabled or not self.twitch_bot:
            return False
            
        try:
            if self.twitch_bot.connect():
                print("Connected to Twitch chat")
                return True
            else:
                print("[ERROR] Failed to connect to Twitch chat")
                return False
        except Exception as e:
            print(f"[ERROR] Twitch connection error: {e}")
            return False

    def send_twitch_alert(self, detection_count, time_span_minutes, is_first_alert):
        """Send alert to Twitch chat"""
        if not self.twitch_enabled or not self.twitch_bot:
            return False

        try:
            # Determine severity
            if detection_count >= 12:
                severity = "[SEVERE]"
            elif detection_count >= 8:
                severity = "[MODERATE]"
            else:
                severity = "[MINOR]"

            # Create message based on whether this is first alert or followup
            if is_first_alert:
                message = f"{severity} Audio issues detected! {detection_count} glitches in {time_span_minutes:.1f} minutes. Stream audio may be choppy! modCheck"
            else:
                message = f"{severity} Ongoing audio issue: {detection_count} glitches in last {time_span_minutes:.1f} minutes. Still unstable... modCheck"

            # Send to chat
            success = self.twitch_bot.send_message(message)
            if success:
                self.total_alerts_sent += 1
                print(f"Twitch alert sent: {message}")
            else:
                print("[ERROR] Failed to send Twitch alert")

            return success

        except Exception as e:
            print(f"[ERROR] Error sending Twitch alert: {e}")
            return False

    def should_send_alert(self):
        """Check if we should send a Twitch alert based on recent detections"""
        current_time = time.time()

        # Check cooldown
        time_since_last_alert = current_time - self.last_alert_time
        cooldown_seconds = ALERT_CONFIG['alert_cooldown_ms'] / 1000.0
        if time_since_last_alert < cooldown_seconds:
            cooldown_remaining = cooldown_seconds - time_since_last_alert
            return False, 0, 0, cooldown_remaining

        # Count recent high-confidence detections
        cutoff_time = current_time - ALERT_CONFIG['detection_window_seconds']
        recent_detections = [
            det for det in self.detection_history
            if det['timestamp'] > cutoff_time and det['confidence'] >= ALERT_CONFIG['confidence_threshold']
        ]

        detection_count = len(recent_detections)
        time_span_minutes = ALERT_CONFIG['detection_window_seconds'] / 60

        if detection_count >= ALERT_CONFIG['detections_for_alert']:
            return True, detection_count, time_span_minutes, 0

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
            return True, burst_count, burst_time_span_minutes, 0

        return False, detection_count, time_span_minutes, 0

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
        if status:
            print(f"Audio callback status: {status}")
            
        with self.lock:
            audio_data = indata[:, 0] if indata.ndim > 1 else indata
            self.audio_buffer.extend(audio_data)

    def detection_loop(self):
        """Main detection loop with reasonable timing"""
        last_detection_time = 0
        detection_cooldown = 0.5  # Don't spam detections
        
        print("Listening for streaming audio glitches...")
        print("Building baseline audio profile...")
        
        while self.running:
            time.sleep(0.2)  # 5Hz analysis rate
            
            current_time = time.time()
            if current_time - last_detection_time < detection_cooldown:
                continue
            
            with self.lock:
                if len(self.audio_buffer) < SAMPLE_RATE:  # Need at least 1 second
                    continue
                    
                audio = np.array(list(self.audio_buffer))

            # --- Volume heartbeat (prints once at start, then every 30 seconds) ---
            rms_level = float(np.sqrt(np.mean(audio**2)))

            if not self._volume_report_started:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Detection running. "
                      f"{self._format_volume(rms_level)} | min_analyze={THRESHOLDS['min_audio_level']} "
                      f"| high_conf_total={self.total_detections}")
                self._volume_report_started = True
                self._last_volume_report_time = current_time

            elif (current_time - self._last_volume_report_time) >= self.volume_report_interval_sec:
                status = "OK" if rms_level >= THRESHOLDS['min_audio_level'] else "quiet"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Audio level: "
                      f"{self._format_volume(rms_level)} [{status}] "
                      f"| windows={self.recent_analysis_windows} "
                      f"| low_conf={self.recent_low_conf_hits} "
                      f"| high_conf_total={self.total_detections}")
                self._last_volume_report_time = current_time
                self.recent_analysis_windows = 0
                self.recent_low_conf_hits = 0
                
            # Analyze audio
            results = self.analyze_audio(audio)
            
            # Assess confidence
            confidence, reasons = self.assess_glitch_confidence(results)
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
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] Possible glitch (confidence: {confidence:.1%}) - {reasons}")
                        self.last_possible_glitch_log_time = current_time
                        self.last_possible_glitch_log_reason = reason_key
                continue

            # De-duplicate repeated windows from the same glitch burst.
            active_methods = tuple(sorted(method for method, result in results.items() if result.get('choppy')))
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

            # FIXED: Check for episode reset BEFORE recording the new detection
            self.check_episode_reset(current_time)
            
            # High confidence detection found
            primary_hit = any(
                results.get(name, {}).get('choppy', False)
                for name in ('silence_gaps', 'envelope_discontinuity')
            )
            self.record_detection(confidence * 100, reasons, primary_hit)
            
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"\n[{timestamp}] STREAMING GLITCH DETECTED!")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Reasons: {reasons}")
            
            # Show details for debugging
            for method, result in results.items():
                if result['choppy']:
                    print(f"    {method}: {result.get('description', result['score'])}")
            
            # Check if we should send Twitch alert
            should_alert, detection_count, time_span, cooldown_remaining = self.should_send_alert()
            
            if should_alert:
                # Check if this is a new episode or continuation
                is_first_alert = not self.current_episode_started
                
                # Send alert
                success = self.send_twitch_alert(detection_count, time_span, is_first_alert)
                if success:
                    self.last_alert_time = current_time
                    self.current_episode_started = True
                
            else:
                # Show why alert wasn't sent
                if cooldown_remaining > 0:
                    cooldown_minutes = cooldown_remaining / 60.0
                    print(f"    Alert cooldown active (~{cooldown_minutes:.1f} min remaining)")
                else:
                    print(f"    Recent detections: {detection_count}/{ALERT_CONFIG['detections_for_alert']} (need {ALERT_CONFIG['detections_for_alert']} for alert)")
            
            last_detection_time = current_time

    def start_detection(self):
        """Start balanced detection with optional Twitch integration"""
        print("Balanced Choppy Audio Detector with Twitch Integration")
        print("Focused on detecting streaming-related audio glitches")
        
        if self.twitch_enabled:
            print("Attempting to connect to Twitch...")
            self.connect_to_twitch()
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
        print(f"  Log low-confidence possible glitches: {ALERT_CONFIG['log_possible_glitches']}")
        print()
        
        self.running = True
        
        detection_thread = threading.Thread(target=self.detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        try:
            with sd.InputStream(
                device=self.audio_device,
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback
            ):
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\nStopping detection...")
            self.running = False
            
            # Show final stats
            print("\nFinal Statistics:")
            print(f"  Total detections: {self.total_detections}")
            print(f"  Alerts sent: {self.total_alerts_sent}")
            
            # Disconnect from Twitch
            if self.twitch_bot:
                self.twitch_bot.disconnect()

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
        '--list-devices', 
        action='store_true', 
        help='List available audio devices and exit'
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
    
    detector = BalancedChoppyDetector(enable_twitch=enable_twitch, audio_device=audio_device)
    
    try:
        detector.start_detection()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Detection stopped.")

if __name__ == "__main__":
    main()
