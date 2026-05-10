"""Offline WAV analysis helpers for the Playground tab."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import json
import struct
import wave

import numpy as np

from .settings import AppSettings


@dataclass
class LoadedWavFile:
    path: str
    sample_rate: int
    channel_count: int
    frame_count: int
    duration_ms: int
    samples: np.ndarray


@dataclass
class PlaygroundTelemetryRow:
    index: int
    start_ms: int
    end_ms: int
    rms_dbfs: float
    confidence_pct: float
    high_confidence: bool
    primary_hit: bool
    deduped_detection: bool
    suppressed_by_warmup: bool
    methods: str
    reasons: str
    silence_ratio: float
    gap_count: int
    max_gap_ms: float
    envelope_score: float
    modulation_strength: float
    modulation_freq_hz: float
    modulation_depth: float
    modulation_peak_concentration: float


@dataclass
class PlaygroundAnalysisResult:
    file: LoadedWavFile
    channel_index: int
    window_ms: int
    step_ms: int
    warmup_ms: int
    rows: list[PlaygroundTelemetryRow]
    deduped_detection_count: int
    high_confidence_count: int
    warmup_suppressed_count: int
    max_confidence_pct: float
    average_confidence_pct: float
    baseline_rms: float
    baseline_sample_count: int
    baseline_established: bool


@dataclass
class MarkerAlignmentSummary:
    marker_count: int
    marker_window_ms: int
    marker_hits: int
    marker_misses: int
    outside_marker_hits: int


def write_compact_report(
    result: PlaygroundAnalysisResult,
    settings: AppSettings,
    *,
    reports_dir: Path,
    expected_glitch: bool,
    report_stem: str | None = None,
    extended_report: bool = False,
    markers_ms: list[int] | None = None,
    marker_window_ms: int = 450,
    marker_latency_ms: int = 270,
) -> Path:
    report_text = build_compact_report(
        result,
        settings,
        expected_glitch=expected_glitch,
        extended_report=extended_report,
        markers_ms=markers_ms,
        marker_window_ms=marker_window_ms,
        marker_latency_ms=marker_latency_ms,
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = report_stem.strip() if isinstance(report_stem, str) else ""
    if not stem:
        stem = Path(result.file.path).stem
    report_prefix = f"w{int(result.window_ms)}_s{int(result.step_ms)}"
    report_path = reports_dir / f"{report_prefix}_{stem}.report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def build_compact_report(
    result: PlaygroundAnalysisResult,
    settings: AppSettings,
    *,
    expected_glitch: bool,
    extended_report: bool = False,
    markers_ms: list[int] | None = None,
    marker_window_ms: int = 450,
    marker_latency_ms: int = 270,
) -> str:
    rows = result.rows
    high_conf_rows = [r for r in rows if r.high_confidence]
    dedup_rows = [r for r in rows if r.deduped_detection]
    method_counts_high: dict[str, int] = {}
    method_counts_all: dict[str, int] = {}
    for row in rows:
        if not row.methods:
            continue
        for method in row.methods.split(","):
            key = method.strip()
            if not key:
                continue
            method_counts_all[key] = method_counts_all.get(key, 0) + 1
            if row.high_confidence:
                method_counts_high[key] = method_counts_high.get(key, 0) + 1
    method_counts_text = ",".join(f"{k}:{method_counts_high[k]}" for k in sorted(method_counts_high)) or "none"
    method_windows_all_text = ",".join(f"{k}:{method_counts_all[k]}" for k in sorted(method_counts_all)) or "none"
    dedup_window_text = ";".join(
        f"{r.start_ms}-{r.end_ms}@{r.confidence_pct:.1f}%|{r.methods or '-'}" for r in dedup_rows
    ) or "none"
    top_rows = sorted(rows, key=lambda r: r.confidence_pct, reverse=True)[:15]
    top_windows_text = ";".join(
        f"{r.start_ms}@{r.confidence_pct:.1f}%|{r.methods or '-'}" for r in top_rows if r.confidence_pct > 0.0
    ) or "none"
    low_rows = sorted(rows, key=lambda r: r.confidence_pct)[:10]
    low_windows_text = ";".join(
        f"{r.start_ms}@{r.confidence_pct:.1f}%|{r.methods or '-'}" for r in low_rows
    ) or "none"

    thresholds = settings.advanced_thresholds
    methods = settings.detection_methods
    active_methods = ",".join(key for key in sorted(methods) if methods.get(key)) or "none"
    detected_glitch = result.deduped_detection_count > 0
    if expected_glitch and detected_glitch:
        outcome = "TP"
    elif (not expected_glitch) and (not detected_glitch):
        outcome = "TN"
    elif (not expected_glitch) and detected_glitch:
        outcome = "FP"
    else:
        outcome = "FN"

    confidence_values = np.array([float(r.confidence_pct) for r in rows], dtype=np.float64)
    nonzero_conf = int(np.sum(confidence_values > 0.0))
    conf_p50 = float(np.percentile(confidence_values, 50)) if len(confidence_values) else 0.0
    conf_p75 = float(np.percentile(confidence_values, 75)) if len(confidence_values) else 0.0
    conf_p90 = float(np.percentile(confidence_values, 90)) if len(confidence_values) else 0.0
    conf_p95 = float(np.percentile(confidence_values, 95)) if len(confidence_values) else 0.0
    conf_p99 = float(np.percentile(confidence_values, 99)) if len(confidence_values) else 0.0

    duration_sec = max(0.001, result.file.duration_ms / 1000.0)
    high_conf_per_min = (result.high_confidence_count / duration_sec) * 60.0
    dedup_per_min = (result.deduped_detection_count / duration_sec) * 60.0

    silence_values = np.array([float(r.silence_ratio) for r in rows], dtype=np.float64)
    gap_counts = np.array([float(r.gap_count) for r in rows], dtype=np.float64)
    gap_max_values = np.array([float(r.max_gap_ms) for r in rows], dtype=np.float64)
    envelope_values = np.array([float(r.envelope_score) for r in rows], dtype=np.float64)
    modulation_strength_values = np.array([float(r.modulation_strength) for r in rows], dtype=np.float64)
    modulation_depth_values = np.array([float(r.modulation_depth) for r in rows], dtype=np.float64)
    modulation_conc_values = np.array([float(r.modulation_peak_concentration) for r in rows], dtype=np.float64)

    def pct(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, q))

    def mean(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return float(np.mean(values))

    primary_hit_windows = sum(1 for r in rows if r.primary_hit)
    warmup_hit_windows = sum(1 for r in rows if r.suppressed_by_warmup)

    dedup_gap_text = "none"
    if len(dedup_rows) >= 2:
        starts = [r.start_ms for r in dedup_rows]
        gaps = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
        dedup_gap_text = ",".join(str(g) for g in gaps[:20])

    lines = [
        "CHOPPY_PLAYGROUND_REPORT v2",
        f"sample={Path(result.file.path).name}",
        (
            f"audio=duration_ms:{result.file.duration_ms},sample_rate:{result.file.sample_rate},"
            f"channels:{result.file.channel_count},analysis_channel:{result.channel_index + 1}"
        ),
        (
            f"windowing=window_ms:{result.window_ms},step_ms:{result.step_ms},"
            f"warmup_ignore_ms:{result.warmup_ms}"
        ),
        (
            f"summary=windows:{len(rows)},high_conf:{result.high_confidence_count},"
            f"dedup_hits:{result.deduped_detection_count},warmup_suppressed:{result.warmup_suppressed_count},"
            f"max_conf_pct:{result.max_confidence_pct:.1f},avg_conf_pct:{result.average_confidence_pct:.1f}"
        ),
        (
            f"baseline=rms:{result.baseline_rms:.6f},sample_count:{result.baseline_sample_count},"
            f"established:{1 if result.baseline_established else 0}"
        ),
        (
            f"rates=high_conf_per_min:{high_conf_per_min:.2f},dedup_per_min:{dedup_per_min:.2f},"
            f"primary_hit_windows:{primary_hit_windows},warmup_hit_windows:{warmup_hit_windows}"
        ),
        (
            f"confidence_dist=nonzero:{nonzero_conf},p50:{conf_p50:.1f},p75:{conf_p75:.1f},"
            f"p90:{conf_p90:.1f},p95:{conf_p95:.1f},p99:{conf_p99:.1f},max:{result.max_confidence_pct:.1f}"
        ),
        (
            f"expectation=expected_glitch:{1 if expected_glitch else 0},"
            f"detected_glitch:{1 if detected_glitch else 0},outcome:{outcome}"
        ),
        f"active_methods={active_methods}",
        (
            "thresholds="
            f"min_audio_level:{thresholds.get('min_audio_level')},"
            f"silence_ratio:{thresholds.get('silence_ratio')},"
            f"gap_duration_ms:{thresholds.get('gap_duration_ms')},"
            f"suspicious_gap_count:{thresholds.get('suspicious_gap_count')},"
            f"envelope_discontinuity:{thresholds.get('envelope_discontinuity')},"
            f"modulation_strength:{thresholds.get('modulation_strength')},"
            f"modulation_depth:{thresholds.get('modulation_depth')},"
            f"modulation_peak_concentration:{thresholds.get('modulation_peak_concentration')}"
        ),
        f"method_hits_high_conf={method_counts_text}",
        f"method_hits_all_windows={method_windows_all_text}",
        (
            f"feature_silence=silence_avg:{mean(silence_values):.4f},silence_p95:{pct(silence_values, 95):.4f},"
            f"silence_max:{float(np.max(silence_values)) if silence_values.size else 0.0:.4f},"
            f"gap_count_avg:{mean(gap_counts):.3f},gap_count_p95:{pct(gap_counts, 95):.3f},"
            f"gap_max_ms_p95:{pct(gap_max_values, 95):.1f},gap_max_ms_max:{float(np.max(gap_max_values)) if gap_max_values.size else 0.0:.1f}"
        ),
        (
            f"feature_envelope_mod=envelope_avg:{mean(envelope_values):.3f},envelope_p95:{pct(envelope_values, 95):.3f},"
            f"envelope_max:{float(np.max(envelope_values)) if envelope_values.size else 0.0:.3f},"
            f"mod_strength_p95:{pct(modulation_strength_values, 95):.3f},mod_strength_max:{float(np.max(modulation_strength_values)) if modulation_strength_values.size else 0.0:.3f},"
            f"mod_depth_p95:{pct(modulation_depth_values, 95):.3f},mod_depth_max:{float(np.max(modulation_depth_values)) if modulation_depth_values.size else 0.0:.3f},"
            f"mod_conc_p95:{pct(modulation_conc_values, 95):.3f},mod_conc_max:{float(np.max(modulation_conc_values)) if modulation_conc_values.size else 0.0:.3f}"
        ),
        f"dedup_gap_ms={dedup_gap_text}",
        f"dedup_windows={dedup_window_text}",
        f"top_windows={top_windows_text}",
        f"low_windows={low_windows_text}",
    ]

    normalized_markers = sorted({max(0, int(round(v))) for v in (markers_ms or [])})
    marker_points_text = "none"
    if normalized_markers:
        max_points = 120
        shown = normalized_markers[:max_points]
        marker_points_text = ",".join(str(v) for v in shown)
        if len(normalized_markers) > max_points:
            marker_points_text = f"{marker_points_text},...(+{len(normalized_markers) - max_points})"
    lines.append(
        "markers="
        f"provided:{1 if normalized_markers else 0},"
        f"count:{len(normalized_markers)},"
        f"latency_ms:{max(0, int(marker_latency_ms))},"
        f"match_window_ms:{max(0, int(marker_window_ms))}"
    )
    lines.append(f"marker_points_ms={marker_points_text}")

    marker_summary = summarize_marker_alignment(
        rows=rows,
        markers_ms=normalized_markers,
        marker_window_ms=marker_window_ms,
    )
    if marker_summary is None:
        lines.append("marker_alignment=not_provided")
    else:
        hit_rate = (
            (marker_summary.marker_hits / marker_summary.marker_count) * 100.0
            if marker_summary.marker_count > 0
            else 0.0
        )
        lines.append(
            "marker_alignment="
            f"marker_count:{marker_summary.marker_count},"
            f"marker_window_ms:{marker_summary.marker_window_ms},"
            f"marker_hits:{marker_summary.marker_hits},"
            f"marker_misses:{marker_summary.marker_misses},"
            f"outside_marker_hits:{marker_summary.outside_marker_hits},"
            f"hit_rate_pct:{hit_rate:.1f}"
        )

    if extended_report:
        near_threshold_rows = [
            r for r in rows if (65.0 <= float(r.confidence_pct) < 75.0)
        ]
        near_threshold_text = ";".join(
            f"{r.start_ms}-{r.end_ms}@{r.confidence_pct:.1f}%|{r.methods or '-'}"
            for r in near_threshold_rows[:300]
        ) or "none"
        high_conf_windows_text = ";".join(
            f"{r.start_ms}-{r.end_ms}@{r.confidence_pct:.1f}%|{r.methods or '-'}"
            for r in high_conf_rows[:500]
        ) or "none"

        # Group adjacent near/high windows into coarse burst clusters.
        cluster_source = sorted(
            [r for r in rows if float(r.confidence_pct) >= 65.0],
            key=lambda r: r.start_ms,
        )
        cluster_items: list[str] = []
        if cluster_source:
            cluster_start = cluster_source[0].start_ms
            cluster_end = cluster_source[0].end_ms
            cluster_count = 1
            cluster_peak = float(cluster_source[0].confidence_pct)
            cluster_silence_sum = float(cluster_source[0].silence_ratio)
            for row in cluster_source[1:]:
                if row.start_ms - cluster_end <= 1000:
                    cluster_end = max(cluster_end, row.end_ms)
                    cluster_count += 1
                    cluster_peak = max(cluster_peak, float(row.confidence_pct))
                    cluster_silence_sum += float(row.silence_ratio)
                    continue
                cluster_items.append(
                    f"{cluster_start}-{cluster_end}|n:{cluster_count}|peak:{cluster_peak:.1f}|sil_avg:{(cluster_silence_sum / max(1, cluster_count)):.3f}"
                )
                cluster_start = row.start_ms
                cluster_end = row.end_ms
                cluster_count = 1
                cluster_peak = float(row.confidence_pct)
                cluster_silence_sum = float(row.silence_ratio)
            cluster_items.append(
                f"{cluster_start}-{cluster_end}|n:{cluster_count}|peak:{cluster_peak:.1f}|sil_avg:{(cluster_silence_sum / max(1, cluster_count)):.3f}"
            )
        cluster_summary_text = ";".join(cluster_items[:200]) or "none"

        method_transitions: dict[str, int] = {}
        prev_method_key: str | None = None
        for row in rows:
            method_key = row.methods or "-"
            if prev_method_key is not None:
                key = f"{prev_method_key}->{method_key}"
                method_transitions[key] = method_transitions.get(key, 0) + 1
            prev_method_key = method_key
        transition_text = ",".join(
            f"{k}:{method_transitions[k]}"
            for k in sorted(method_transitions, key=lambda x: method_transitions[x], reverse=True)[:25]
        ) or "none"

        top_negative_rows = sorted(
            [r for r in rows if not r.high_confidence and float(r.confidence_pct) > 0.0],
            key=lambda r: r.confidence_pct,
            reverse=True,
        )[:25]
        top_negative_text = ";".join(
            f"{r.start_ms}-{r.end_ms}@{r.confidence_pct:.1f}%|{r.methods or '-'}"
            for r in top_negative_rows
        ) or "none"

        def range_text(values: np.ndarray) -> str:
            if values.size == 0:
                return "min:0,p50:0,p90:0,p99:0,max:0"
            return (
                f"min:{float(np.min(values)):.4f},"
                f"p50:{float(np.percentile(values, 50)):.4f},"
                f"p90:{float(np.percentile(values, 90)):.4f},"
                f"p99:{float(np.percentile(values, 99)):.4f},"
                f"max:{float(np.max(values)):.4f}"
            )

        method_presence = {
            "silence_gaps": sum(1 for r in rows if "silence_gaps" in (r.methods or "")),
            "silence_gaps_persistent": sum(1 for r in rows if "silence_gaps_persistent" in (r.methods or "")),
            "envelope_discontinuity": sum(1 for r in rows if "envelope_discontinuity" in (r.methods or "")),
            "amplitude_modulation": sum(1 for r in rows if "amplitude_modulation" in (r.methods or "")),
        }
        promotion_audit = (
            f"persistent_rows:{method_presence['silence_gaps_persistent']},"
            f"silence_rows:{method_presence['silence_gaps']},"
            f"envelope_rows:{method_presence['envelope_discontinuity']},"
            f"mod_rows:{method_presence['amplitude_modulation']}"
        )

        lines.extend(
            [
                "extended=1",
                f"high_conf_windows={high_conf_windows_text}",
                f"near_threshold_windows={near_threshold_text}",
                f"cluster_summary={cluster_summary_text}",
                f"method_transition_counts={transition_text}",
                f"top_negative_windows={top_negative_text}",
                (
                    "raw_feature_ranges="
                    f"silence[{range_text(silence_values)}],"
                    f"gap_count[{range_text(gap_counts)}],"
                    f"gap_max_ms[{range_text(gap_max_values)}],"
                    f"envelope[{range_text(envelope_values)}],"
                    f"mod_strength[{range_text(modulation_strength_values)}],"
                    f"mod_depth[{range_text(modulation_depth_values)}],"
                    f"mod_conc[{range_text(modulation_conc_values)}]"
                ),
                f"promotion_audit={promotion_audit}",
            ]
        )
    else:
        lines.append("extended=0")

    return "\n".join(lines) + "\n"


def _decode_pcm_frames(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        return data / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        return (signed.astype(np.float32) / 8388608.0).astype(np.float32)
    if sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        return data / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def _decode_float_frames(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
    if sample_width == 8:
        return np.frombuffer(raw, dtype="<f8").astype(np.float32, copy=False)
    raise ValueError(f"Unsupported float WAV sample width: {sample_width} bytes")


def _parse_fmt_chunk(fmt_data: bytes) -> dict[str, int]:
    if len(fmt_data) < 16:
        raise ValueError("Invalid WAV fmt chunk")
    (
        format_tag,
        channel_count,
        sample_rate,
        _byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack("<HHIIHH", fmt_data[:16])
    subformat_tag = 0
    if format_tag == 0xFFFE and len(fmt_data) >= 40:
        # WAVE_FORMAT_EXTENSIBLE: decode first DWORD of GUID when it follows
        # the canonical WAVEFORMATEX GUID tail.
        subformat_guid = fmt_data[24:40]
        guid_tail = b"\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71"
        if len(subformat_guid) == 16 and subformat_guid.endswith(guid_tail):
            subformat_tag = int(struct.unpack("<I", subformat_guid[:4])[0] & 0xFFFF)
    return {
        "format_tag": int(format_tag),
        "subformat_tag": int(subformat_tag),
        "channel_count": int(channel_count),
        "sample_rate": int(sample_rate),
        "block_align": int(block_align),
        "bits_per_sample": int(bits_per_sample),
    }


def _read_wav_riff_fallback(path: Path) -> tuple[int, int, int, int, bytes, int]:
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] not in {b"RIFF", b"RF64"} or header[8:12] != b"WAVE":
            raise ValueError("Not a valid RIFF/WAVE file")

        fmt_info: dict[str, int] | None = None
        data_chunk: bytes | None = None
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_data = f.read(chunk_size)
            if chunk_size % 2:
                f.read(1)
            if chunk_id == b"fmt ":
                fmt_info = _parse_fmt_chunk(chunk_data)
            elif chunk_id == b"data":
                data_chunk = chunk_data
                if fmt_info is not None:
                    break

    if fmt_info is None:
        raise ValueError("WAV fmt chunk not found")
    if data_chunk is None:
        raise ValueError("WAV data chunk not found")

    format_tag = int(fmt_info["format_tag"])
    subformat_tag = int(fmt_info["subformat_tag"])
    channel_count = int(fmt_info["channel_count"])
    sample_rate = int(fmt_info["sample_rate"])
    block_align = max(1, int(fmt_info["block_align"]))
    bits_per_sample = int(fmt_info["bits_per_sample"])
    sample_width = max(1, bits_per_sample // 8)
    frame_count = int(len(data_chunk) // block_align)

    # 1 = PCM int, 3 = IEEE float. Extensible maps through subformat_tag.
    normalized_tag = format_tag
    if normalized_tag == 0xFFFE:
        normalized_tag = subformat_tag
    if normalized_tag not in {1, 3}:
        raise ValueError(f"Unsupported WAV format tag: {format_tag} (subformat: {subformat_tag})")
    return sample_rate, channel_count, frame_count, sample_width, data_chunk, normalized_tag


def load_wav_file(path: str) -> LoadedWavFile:
    wav_path = Path(path).expanduser()
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError(
                    f"Unsupported WAV compression '{wav_file.getcomptype()}'. Use PCM WAV files."
                )
            sample_rate = int(wav_file.getframerate())
            channel_count = int(wav_file.getnchannels())
            frame_count = int(wav_file.getnframes())
            sample_width = int(wav_file.getsampwidth())
            raw = wav_file.readframes(frame_count)
            decoded = _decode_pcm_frames(raw, sample_width)
    except wave.Error as exc:
        sample_rate, channel_count, frame_count, sample_width, raw, format_tag = _read_wav_riff_fallback(wav_path)
        if format_tag == 3:
            decoded = _decode_float_frames(raw, sample_width)
        else:
            decoded = _decode_pcm_frames(raw, sample_width)
        if decoded.size == 0:
            raise ValueError(str(exc)) from exc

    if channel_count > 1:
        decoded = decoded.reshape(-1, channel_count)
    else:
        decoded = decoded.reshape(-1, 1)

    duration_ms = int(round((frame_count / float(sample_rate or 1)) * 1000.0))
    return LoadedWavFile(
        path=str(wav_path),
        sample_rate=sample_rate,
        channel_count=channel_count,
        frame_count=frame_count,
        duration_ms=duration_ms,
        samples=decoded.astype(np.float32, copy=False),
    )


def _apply_runtime_settings(settings: AppSettings) -> None:
    import live_analysis

    live_analysis.APPROACHES.update(settings.detection_methods)
    live_analysis.THRESHOLDS.update(settings.advanced_thresholds)
    live_analysis.ALERT_CONFIG.update(settings.advanced_alert_config)
    live_analysis.ALERT_CONFIG["alert_cooldown_ms"] = settings.alert_cooldown_ms


def analyze_wav_file(
    loaded: LoadedWavFile,
    settings: AppSettings,
    *,
    channel_index: int,
    window_ms: int,
    step_ms: int,
    warmup_ms: int,
) -> PlaygroundAnalysisResult:
    import live_analysis

    _apply_runtime_settings(settings)
    detector = live_analysis.BalancedChoppyDetector(enable_twitch=False)
    detector.sample_rate = int(loaded.sample_rate)

    channel_idx = max(0, min(int(channel_index), loaded.channel_count - 1))
    mono = loaded.samples[:, channel_idx]
    total_samples = int(len(mono))
    if total_samples <= 0:
        raise ValueError("WAV file has no audio samples")

    window_samples = max(256, int(round(loaded.sample_rate * (window_ms / 1000.0))))
    step_samples = max(64, int(round(loaded.sample_rate * (step_ms / 1000.0))))
    if window_samples > total_samples:
        window_samples = total_samples
    if step_samples > window_samples:
        step_samples = window_samples

    rows: list[PlaygroundTelemetryRow] = []
    high_confidence_count = 0
    warmup_suppressed_count = 0
    deduped_detection_count = 0
    confidences: list[float] = []
    silence_persistence_hits_ms: list[int] = []
    burst_cluster_hits_ms: list[int] = []
    long_window_sparse_burst_hits_ms: list[int] = []
    last_detection_ms = -10_000_000
    last_detection_signature = ""
    dedup_window_ms = float(live_analysis.ALERT_CONFIG.get("event_dedup_seconds", 0.9)) * 1000.0

    row_index = 1
    max_start = max(0, total_samples - window_samples)
    for start in range(0, max_start + 1, step_samples):
        end = start + window_samples
        window = mono[start:end]
        if len(window) == 0:
            continue

        results = detector.analyze_audio(window)
        detector._analysis_window_ms = int(window_ms)
        confidence, reasons = detector.assess_glitch_confidence(results)
        silence_result = results.get("silence_gaps", {}) if isinstance(results, dict) else {}
        silence_choppy = bool((silence_result or {}).get("choppy", False))
        silence_ratio = float((silence_result or {}).get("score", 0.0) or 0.0)
        silence_gaps = silence_result.get("gaps", []) if isinstance(silence_result, dict) else []
        silence_gap_count = len(silence_gaps)
        silence_max_gap_ms = max(
            (float(gap.get("duration_ms", 0.0) or 0.0) for gap in silence_gaps),
            default=0.0,
        )
        envelope_hit = bool((results.get("envelope_discontinuity", {}) or {}).get("choppy", False))
        modulation_hit = bool((results.get("amplitude_modulation", {}) or {}).get("choppy", False))
        persistence_promoted = False
        start_ms = int(round((start / loaded.sample_rate) * 1000.0))
        end_ms = int(round((end / loaded.sample_rate) * 1000.0))
        severe_silence_ratio = max(float(live_analysis.THRESHOLDS.get("silence_ratio", 0.60)) + 0.08, 0.68)
        severe_gap_pattern = (
            silence_ratio >= severe_silence_ratio
            and silence_max_gap_ms >= 250.0
            and silence_gap_count >= 2
        )
        extreme_gap_pattern = silence_ratio >= 0.85 and silence_max_gap_ms >= 500.0
        modulation_gap_pattern = (
            silence_max_gap_ms >= 500.0
            and float((results.get("amplitude_modulation", {}) or {}).get("score", 0.0) or 0.0) >= 4.8
            and float((results.get("amplitude_modulation", {}) or {}).get("depth", 0.0) or 0.0) >= 0.50
            and float((results.get("amplitude_modulation", {}) or {}).get("peak_concentration", 0.0) or 0.0) >= 0.16
        )
        persistence_block_long_window = (
            int(window_ms) >= 1600
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
        silence_persistence_hits_ms = [t for t in silence_persistence_hits_ms if (start_ms - t) <= 2500]
        if persistence_candidate:
            if not silence_persistence_hits_ms or (start_ms - silence_persistence_hits_ms[-1]) >= 450:
                silence_persistence_hits_ms.append(start_ms)
        if len(silence_persistence_hits_ms) >= 3:
            if modulation_hit:
                confidence = max(confidence, 0.80)
                reasons = f"{reasons}; Persistent severe silence-gap pattern corroborated by modulation"
            else:
                confidence = max(confidence, 0.78)
                reasons = f"{reasons}; Persistent severe silence-gap pattern"
            persistence_promoted = True

        mod_result = results.get("amplitude_modulation", {}) if isinstance(results, dict) else {}
        mod_strength = float((mod_result or {}).get("score", 0.0) or 0.0)
        mod_depth = float((mod_result or {}).get("depth", 0.0) or 0.0)
        mod_peak_concentration = float((mod_result or {}).get("peak_concentration", 0.0) or 0.0)
        burst_candidate = (
            silence_choppy
            and not envelope_hit
            and 0.66 <= confidence < 0.75
            and 0.52 <= silence_ratio <= 0.85
            and silence_gap_count <= 2
            and mod_strength >= 4.6
            and mod_depth >= 0.45
        )
        burst_cluster_hits_ms = [t for t in burst_cluster_hits_ms if (start_ms - t) <= 2200]
        if burst_candidate:
            if not burst_cluster_hits_ms or (start_ms - burst_cluster_hits_ms[-1]) >= 120:
                burst_cluster_hits_ms.append(start_ms)
        if len(burst_cluster_hits_ms) >= 3:
            confidence = max(confidence, 0.77)
            reasons = f"{reasons}; Repeated near-threshold burst cluster"

        long_window_sparse_candidate = (
            int(window_ms) >= 1600
            and silence_choppy
            and not envelope_hit
            and 0.68 <= confidence < 0.75
            and 0.52 <= silence_ratio <= 0.85
            and mod_strength >= 5.0
            and mod_depth >= 0.45
            and mod_peak_concentration >= 0.07
        )
        long_window_sparse_burst_hits_ms = [
            t for t in long_window_sparse_burst_hits_ms if (start_ms - t) <= 2800
        ]
        if long_window_sparse_candidate:
            if (
                not long_window_sparse_burst_hits_ms
                or (start_ms - long_window_sparse_burst_hits_ms[-1]) >= 150
            ):
                long_window_sparse_burst_hits_ms.append(start_ms)
        if len(long_window_sparse_burst_hits_ms) >= 4:
            confidence = max(confidence, 0.76)
            reasons = f"{reasons}; Long-window sparse-gap burst cluster"

        confidence_pct = float(round(confidence * 100.0, 1))
        confidences.append(confidence_pct)
        high_confidence = bool(results) and confidence >= 0.75
        if high_confidence:
            high_confidence_count += 1

        suppressed_by_warmup = high_confidence and start_ms < int(warmup_ms)
        if suppressed_by_warmup:
            warmup_suppressed_count += 1

        active_methods = tuple(
            sorted(
                method
                for method, result in results.items()
                if isinstance(result, dict) and bool(result.get("choppy"))
            )
        )
        if persistence_promoted:
            active_methods = tuple(sorted(set(active_methods) | {"silence_gaps_persistent"}))
        signature = "|".join(active_methods)

        deduped_detection = False
        if high_confidence and not suppressed_by_warmup:
            is_duplicate = (
                signature == last_detection_signature
                and (start_ms - last_detection_ms) < dedup_window_ms
            )
            if not is_duplicate:
                deduped_detection = True
                deduped_detection_count += 1
                last_detection_ms = start_ms
                last_detection_signature = signature

        rms = float(np.sqrt(np.mean(window**2)))
        rms_dbfs = 20.0 * math.log10(rms + 1e-12)
        rms_dbfs = max(-120.0, min(0.0, rms_dbfs))

        silence = results.get("silence_gaps", {}) if isinstance(results, dict) else {}
        gaps = silence.get("gaps", []) if isinstance(silence, dict) else []
        max_gap_ms = 0.0
        for gap in gaps:
            duration_ms = float(gap.get("duration_ms", 0.0))
            if duration_ms > max_gap_ms:
                max_gap_ms = duration_ms

        envelope = results.get("envelope_discontinuity", {}) if isinstance(results, dict) else {}
        modulation = results.get("amplitude_modulation", {}) if isinstance(results, dict) else {}

        rows.append(
            PlaygroundTelemetryRow(
                index=row_index,
                start_ms=start_ms,
                end_ms=end_ms,
                rms_dbfs=float(round(rms_dbfs, 2)),
                confidence_pct=confidence_pct,
                high_confidence=high_confidence,
                primary_hit=bool(
                    (results.get("silence_gaps", {}) or {}).get("choppy")
                    or (results.get("envelope_discontinuity", {}) or {}).get("choppy")
                ),
                deduped_detection=deduped_detection,
                suppressed_by_warmup=suppressed_by_warmup,
                methods=", ".join(active_methods),
                reasons=str(reasons or ""),
                silence_ratio=float(round(float(silence.get("score", 0.0) or 0.0), 4)),
                gap_count=int(len(gaps)),
                max_gap_ms=float(round(max_gap_ms, 1)),
                envelope_score=float(round(float(envelope.get("score", 0.0) or 0.0), 3)),
                modulation_strength=float(round(float(modulation.get("score", 0.0) or 0.0), 3)),
                modulation_freq_hz=float(round(float(modulation.get("peak_freq_hz", 0.0) or 0.0), 2)),
                modulation_depth=float(round(float(modulation.get("depth", 0.0) or 0.0), 3)),
                modulation_peak_concentration=float(
                    round(float(modulation.get("peak_concentration", 0.0) or 0.0), 3)
                ),
            )
        )
        row_index += 1

    max_confidence_pct = max(confidences) if confidences else 0.0
    average_confidence_pct = sum(confidences) / len(confidences) if confidences else 0.0
    baseline_history = list((detector.baseline_stats or {}).get("rms_history", []))
    baseline_rms = float(np.median(baseline_history)) if baseline_history else float(detector.get_baseline_rms())
    baseline_sample_count = int(len(baseline_history))
    baseline_established = bool((detector.baseline_stats or {}).get("established_baseline"))
    return PlaygroundAnalysisResult(
        file=loaded,
        channel_index=channel_idx,
        window_ms=int(window_ms),
        step_ms=int(step_ms),
        warmup_ms=int(warmup_ms),
        rows=rows,
        deduped_detection_count=deduped_detection_count,
        high_confidence_count=high_confidence_count,
        warmup_suppressed_count=warmup_suppressed_count,
        max_confidence_pct=float(round(max_confidence_pct, 1)),
        average_confidence_pct=float(round(average_confidence_pct, 1)),
        baseline_rms=float(baseline_rms),
        baseline_sample_count=baseline_sample_count,
        baseline_established=baseline_established,
    )


def summarize_marker_alignment(
    *,
    rows: list[PlaygroundTelemetryRow],
    markers_ms: list[int] | None,
    marker_window_ms: int,
) -> MarkerAlignmentSummary | None:
    if not markers_ms:
        return None
    normalized = sorted({max(0, int(round(v))) for v in markers_ms})
    if not normalized:
        return None
    window = max(0, int(marker_window_ms))
    dedup_rows = [r for r in rows if r.deduped_detection]

    marker_hits = 0
    for marker in normalized:
        if any((row.start_ms - window) <= marker <= (row.end_ms + window) for row in dedup_rows):
            marker_hits += 1
    marker_misses = max(0, len(normalized) - marker_hits)

    outside_marker_hits = 0
    for row in dedup_rows:
        overlaps_any_marker = any(
            (row.start_ms - window) <= marker <= (row.end_ms + window)
            for marker in normalized
        )
        if not overlaps_any_marker:
            outside_marker_hits += 1
    return MarkerAlignmentSummary(
        marker_count=len(normalized),
        marker_window_ms=window,
        marker_hits=marker_hits,
        marker_misses=marker_misses,
        outside_marker_hits=outside_marker_hits,
    )


def marker_sidecar_path(wav_path: str | Path) -> Path:
    path = Path(wav_path).expanduser()
    return path.with_suffix(path.suffix + ".markers.json")


def load_marker_sidecar(wav_path: str | Path) -> list[int]:
    sidecar = marker_sidecar_path(wav_path)
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    markers = payload.get("markers_ms")
    if not isinstance(markers, list):
        return []
    normalized: list[int] = []
    for item in markers:
        try:
            value = int(round(float(item)))
        except Exception:
            continue
        if value >= 0:
            normalized.append(value)
    return sorted(set(normalized))


def save_marker_sidecar(
    loaded: LoadedWavFile,
    markers_ms: list[int],
) -> Path:
    sidecar = marker_sidecar_path(loaded.path)
    cleaned = sorted(set(max(0, int(round(v))) for v in markers_ms))
    payload = {
        "version": 1,
        "wav_file": Path(loaded.path).name,
        "duration_ms": int(loaded.duration_ms),
        "sample_rate": int(loaded.sample_rate),
        "channel_count": int(loaded.channel_count),
        "markers_ms": cleaned,
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar
