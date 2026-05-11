"""Offline WAV analysis helpers for the Playground tab."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import json
import struct
import threading
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
    baseline_source: str


@dataclass
class MarkerAlignmentSummary:
    marker_count: int
    marker_window_ms: int
    marker_hits: int
    marker_misses: int
    outside_marker_hits: int


@dataclass
class BurstInterval:
    start_ms: int
    end_ms: int


@dataclass
class BurstAlignmentSummary:
    human_burst_count: int
    detector_burst_count: int
    human_bursts_covered: int
    human_bursts_missed: int
    detector_bursts_overlapping_human: int
    detector_bursts_outside_human: int


@dataclass
class LoopAlertProjection:
    trigger: bool
    mode: str
    first_alert_ms: int
    loop_duration_ms: int
    qualifying_events_per_loop: int


@dataclass
class ProdAlertSimulation:
    trigger: bool
    mode: str
    first_counted_ms: int
    first_fast_alert_ms: int
    first_standard_alert_ms: int
    counted_detection_count: int
    dedup_suppressed_count: int
    max_fast_count: int
    max_standard_count: int
    blocker_summary: str


def _project_loop_alert(result: PlaygroundAnalysisResult, settings: AppSettings) -> LoopAlertProjection:
    events_template = sorted(
        (
            int(row.start_ms),
            float(row.confidence_pct),
        )
        for row in result.rows
        if row.deduped_detection and row.high_confidence and not row.suppressed_by_warmup
    )
    if not events_template:
        return LoopAlertProjection(
            trigger=False,
            mode="none",
            first_alert_ms=-1,
            loop_duration_ms=max(1, int(result.file.duration_ms)),
            qualifying_events_per_loop=0,
        )

    alert = settings.advanced_alert_config if isinstance(settings.advanced_alert_config, dict) else {}
    confidence_threshold = float(alert.get("confidence_threshold", 70.0))
    detection_window_ms = int(round(float(alert.get("detection_window_seconds", 120.0)) * 1000.0))
    detections_for_alert = max(1, int(alert.get("detections_for_alert", 6)))
    fast_min_conf = float(alert.get("fast_alert_min_confidence", 75.0))
    fast_window_ms = int(round(float(alert.get("fast_alert_window_seconds", 15.0)) * 1000.0))
    fast_burst_required = max(1, int(alert.get("fast_alert_burst_detections", 4)))
    clean_reset_ms = int(round(float(alert.get("clean_audio_reset_seconds", 60.0)) * 1000.0))

    loop_duration_ms = max(1, int(result.file.duration_ms))
    horizon_ms = min(
        3_600_000,
        max(loop_duration_ms * 20, detection_window_ms * 4, fast_window_ms * 10),
    )
    max_loops = max(1, int(math.ceil(horizon_ms / loop_duration_ms)) + 1)

    repeated_events: list[tuple[int, float]] = []
    for loop_idx in range(max_loops):
        loop_offset = loop_idx * loop_duration_ms
        for start_ms, conf_pct in events_template:
            event_time_ms = start_ms + loop_offset
            if event_time_ms > horizon_ms:
                break
            repeated_events.append((event_time_ms, conf_pct))
        if repeated_events and repeated_events[-1][0] > horizon_ms:
            break

    standard_hits: deque[int] = deque()
    fast_hits: deque[int] = deque()
    for event_time_ms, conf_pct in repeated_events:
        if conf_pct >= fast_min_conf:
            fast_hits.append(event_time_ms)
            while fast_hits and (event_time_ms - fast_hits[0]) > fast_window_ms:
                fast_hits.popleft()
            if len(fast_hits) >= fast_burst_required:
                return LoopAlertProjection(
                    trigger=True,
                    mode="fast",
                    first_alert_ms=event_time_ms,
                    loop_duration_ms=loop_duration_ms,
                    qualifying_events_per_loop=sum(1 for _, c in events_template if c >= confidence_threshold),
                )

        if conf_pct >= confidence_threshold:
            if standard_hits and clean_reset_ms > 0 and (event_time_ms - standard_hits[-1]) > clean_reset_ms:
                standard_hits.clear()
            standard_hits.append(event_time_ms)
            while standard_hits and (event_time_ms - standard_hits[0]) > detection_window_ms:
                standard_hits.popleft()
            if len(standard_hits) >= detections_for_alert:
                return LoopAlertProjection(
                    trigger=True,
                    mode="standard",
                    first_alert_ms=event_time_ms,
                    loop_duration_ms=loop_duration_ms,
                    qualifying_events_per_loop=sum(1 for _, c in events_template if c >= confidence_threshold),
                )

    return LoopAlertProjection(
        trigger=False,
        mode="none",
        first_alert_ms=-1,
        loop_duration_ms=loop_duration_ms,
        qualifying_events_per_loop=sum(1 for _, c in events_template if c >= confidence_threshold),
    )


def _simulate_prod_alert_run(
    result: PlaygroundAnalysisResult,
    settings: AppSettings,
) -> tuple[ProdAlertSimulation, list[PlaygroundTelemetryRow], list[PlaygroundTelemetryRow]]:
    alert = settings.advanced_alert_config if isinstance(settings.advanced_alert_config, dict) else {}
    confidence_threshold = float(alert.get("confidence_threshold", 70.0))
    detection_window_ms = int(round(float(alert.get("detection_window_seconds", 120.0)) * 1000.0))
    detections_for_alert = max(1, int(alert.get("detections_for_alert", 6)))
    fast_min_conf = float(alert.get("fast_alert_min_confidence", 75.0))
    fast_window_ms = int(round(float(alert.get("fast_alert_window_seconds", 15.0)) * 1000.0))
    fast_burst_required = max(1, int(alert.get("fast_alert_burst_detections", 4)))
    clean_reset_ms = int(round(float(alert.get("clean_audio_reset_seconds", 60.0)) * 1000.0))

    high_conf_rows = [
        row for row in result.rows if row.high_confidence and not row.suppressed_by_warmup
    ]
    counted_rows = [
        row for row in result.rows
        if row.deduped_detection and row.high_confidence and not row.suppressed_by_warmup
    ]
    dedup_suppressed_rows = [
        row for row in result.rows
        if row.high_confidence and not row.deduped_detection and not row.suppressed_by_warmup
    ]

    standard_hits: deque[int] = deque()
    fast_hits: deque[int] = deque()
    max_standard_count = 0
    max_fast_count = 0
    first_counted_ms = counted_rows[0].start_ms if counted_rows else -1
    first_fast_alert_ms = -1
    first_standard_alert_ms = -1

    for row in counted_rows:
        event_time_ms = int(row.start_ms)
        conf_pct = float(row.confidence_pct)
        if conf_pct >= fast_min_conf:
            fast_hits.append(event_time_ms)
            while fast_hits and (event_time_ms - fast_hits[0]) > fast_window_ms:
                fast_hits.popleft()
            max_fast_count = max(max_fast_count, len(fast_hits))
            if first_fast_alert_ms < 0 and len(fast_hits) >= fast_burst_required:
                first_fast_alert_ms = event_time_ms

        if conf_pct >= confidence_threshold:
            if standard_hits and clean_reset_ms > 0 and (event_time_ms - standard_hits[-1]) > clean_reset_ms:
                standard_hits.clear()
            standard_hits.append(event_time_ms)
            while standard_hits and (event_time_ms - standard_hits[0]) > detection_window_ms:
                standard_hits.popleft()
            max_standard_count = max(max_standard_count, len(standard_hits))
            if first_standard_alert_ms < 0 and len(standard_hits) >= detections_for_alert:
                first_standard_alert_ms = event_time_ms

    if first_fast_alert_ms >= 0 and (first_standard_alert_ms < 0 or first_fast_alert_ms <= first_standard_alert_ms):
        trigger = True
        mode = "fast"
    elif first_standard_alert_ms >= 0:
        trigger = True
        mode = "standard"
    else:
        trigger = False
        mode = "none"

    if trigger:
        blocker_summary = "none"
    elif not high_conf_rows:
        blocker_summary = "no_high_conf_windows"
    elif not counted_rows:
        if dedup_suppressed_rows:
            blocker_summary = "all_high_conf_windows_dedup_suppressed"
        else:
            blocker_summary = "all_high_conf_windows_warmup_suppressed"
    else:
        blocker_parts: list[str] = []
        if first_fast_alert_ms < 0:
            blocker_parts.append(f"fast_peak_{max_fast_count}_of_{fast_burst_required}")
        if first_standard_alert_ms < 0:
            blocker_parts.append(f"standard_peak_{max_standard_count}_of_{detections_for_alert}")
        if dedup_suppressed_rows:
            blocker_parts.append(f"dedup_suppressed_{len(dedup_suppressed_rows)}")
        blocker_summary = ",".join(blocker_parts) if blocker_parts else "below_alert_threshold"

    return (
        ProdAlertSimulation(
            trigger=trigger,
            mode=mode,
            first_counted_ms=first_counted_ms,
            first_fast_alert_ms=first_fast_alert_ms,
            first_standard_alert_ms=first_standard_alert_ms,
            counted_detection_count=len(counted_rows),
            dedup_suppressed_count=len(dedup_suppressed_rows),
            max_fast_count=max_fast_count,
            max_standard_count=max_standard_count,
            blocker_summary=blocker_summary,
        ),
        counted_rows,
        dedup_suppressed_rows,
    )


def _format_event_windows(rows: list[PlaygroundTelemetryRow], *, limit: int = 60) -> str:
    if not rows:
        return "none"
    return ";".join(
        f"{row.start_ms}-{row.end_ms}@{row.confidence_pct:.1f}%|{row.methods or '-'}"
        for row in rows[:limit]
    )


def _format_event_clusters(rows: list[PlaygroundTelemetryRow], *, gap_ms: int = 1000, limit: int = 60) -> str:
    if not rows:
        return "none"
    ordered = sorted(rows, key=lambda row: row.start_ms)
    cluster_items: list[str] = []
    cluster_start = ordered[0].start_ms
    cluster_end = ordered[0].end_ms
    cluster_count = 1
    cluster_peak = float(ordered[0].confidence_pct)
    cluster_methods: set[str] = {ordered[0].methods or "-"}
    for row in ordered[1:]:
        if row.start_ms - cluster_end <= gap_ms:
            cluster_end = max(cluster_end, row.end_ms)
            cluster_count += 1
            cluster_peak = max(cluster_peak, float(row.confidence_pct))
            cluster_methods.add(row.methods or "-")
            continue
        cluster_items.append(
            f"{cluster_start}-{cluster_end}|n:{cluster_count}|peak:{cluster_peak:.1f}|methods:{'+'.join(sorted(cluster_methods))}"
        )
        cluster_start = row.start_ms
        cluster_end = row.end_ms
        cluster_count = 1
        cluster_peak = float(row.confidence_pct)
        cluster_methods = {row.methods or "-"}
    cluster_items.append(
        f"{cluster_start}-{cluster_end}|n:{cluster_count}|peak:{cluster_peak:.1f}|methods:{'+'.join(sorted(cluster_methods))}"
    )
    return ";".join(cluster_items[:limit]) or "none"


def _summarize_reason_counts(rows: list[PlaygroundTelemetryRow], *, limit: int = 12) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        for raw_reason in str(row.reasons or "").split(";"):
            reason = raw_reason.strip()
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "none"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(f"{reason}:{count}" for reason, count in ranked[:limit])


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
    extended_report = True
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
    loop_projection = _project_loop_alert(result, settings)
    prod_simulation, counted_rows, dedup_suppressed_rows = _simulate_prod_alert_run(result, settings)
    loop_projection_pass = (
        loop_projection.trigger if expected_glitch else (not loop_projection.trigger)
    )
    alert_cfg = settings.advanced_alert_config if isinstance(settings.advanced_alert_config, dict) else {}
    confidence_threshold = float(alert_cfg.get("confidence_threshold", 70.0))
    detections_for_alert = max(1, int(alert_cfg.get("detections_for_alert", 6)))
    detection_window_ms = int(round(float(alert_cfg.get("detection_window_seconds", 120.0)) * 1000.0))
    fast_min_conf = float(alert_cfg.get("fast_alert_min_confidence", 75.0))
    fast_window_ms = int(round(float(alert_cfg.get("fast_alert_window_seconds", 15.0)) * 1000.0))
    fast_burst_required = max(1, int(alert_cfg.get("fast_alert_burst_detections", 4)))
    clean_reset_ms = int(round(float(alert_cfg.get("clean_audio_reset_seconds", 60.0)) * 1000.0))

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
        "CHOPPY_PLAYGROUND_REPORT v3",
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
        f"baseline_source={result.baseline_source}",
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
        (
            "loop_alert_projection="
            f"trigger:{1 if loop_projection.trigger else 0},"
            f"mode:{loop_projection.mode},"
            f"first_alert_ms:{loop_projection.first_alert_ms},"
            f"events_per_loop:{loop_projection.qualifying_events_per_loop},"
            f"loop_duration_ms:{loop_projection.loop_duration_ms},"
            f"judgement:{'pass' if loop_projection_pass else 'fail'}"
        ),
        (
            "prod_alert_simulation="
            f"trigger:{1 if prod_simulation.trigger else 0},"
            f"mode:{prod_simulation.mode},"
            f"first_counted_ms:{prod_simulation.first_counted_ms},"
            f"first_fast_alert_ms:{prod_simulation.first_fast_alert_ms},"
            f"first_standard_alert_ms:{prod_simulation.first_standard_alert_ms},"
            f"counted_detections:{prod_simulation.counted_detection_count},"
            f"dedup_suppressed:{prod_simulation.dedup_suppressed_count},"
            f"max_fast_count:{prod_simulation.max_fast_count},"
            f"max_standard_count:{prod_simulation.max_standard_count},"
            f"blockers:{prod_simulation.blocker_summary}"
        ),
        (
            "alert_config="
            f"confidence_threshold:{confidence_threshold:.1f},"
            f"detections_for_alert:{detections_for_alert},"
            f"detection_window_ms:{detection_window_ms},"
            f"fast_min_confidence:{fast_min_conf:.1f},"
            f"fast_burst_required:{fast_burst_required},"
            f"fast_window_ms:{fast_window_ms},"
            f"stream_start_warmup_ignore_ms:{int(round(float(alert_cfg.get('stream_start_warmup_ignore_seconds', 3.0)) * 1000.0))},"
            f"clean_audio_reset_ms:{clean_reset_ms}"
        ),
        f"active_methods={active_methods}",
        (
            "thresholds="
            f"min_audio_level:{thresholds.get('min_audio_level')},"
            f"silence_ratio:{thresholds.get('silence_ratio')},"
            f"gap_duration_ms:{thresholds.get('gap_duration_ms')},"
            f"suspicious_gap_count:{thresholds.get('suspicious_gap_count')},"
            f"silence_guardrail_cap:{thresholds.get('silence_guardrail_cap')},"
            f"silence_extreme_ratio:{thresholds.get('silence_extreme_ratio')},"
            f"silence_extreme_gap_ms:{thresholds.get('silence_extreme_gap_ms')},"
            f"silence_extreme_gap_count_offset:{thresholds.get('silence_extreme_gap_count_offset')},"
            f"silence_require_modulation_hit:{1 if bool(thresholds.get('silence_require_modulation_hit')) else 0},"
            f"silence_persistence_require_modulation_hit:{1 if bool(thresholds.get('silence_persistence_require_modulation_hit')) else 0},"
            f"burst_promotion_require_modulation_hit:{1 if bool(thresholds.get('burst_promotion_require_modulation_hit')) else 0},"
            f"long_window_sparse_promotion_require_modulation_hit:{1 if bool(thresholds.get('long_window_sparse_promotion_require_modulation_hit')) else 0},"
            f"burst_promotion_uncorroborated_cap:{thresholds.get('burst_promotion_uncorroborated_cap')},"
            f"long_window_sparse_uncorroborated_cap:{thresholds.get('long_window_sparse_uncorroborated_cap')},"
            f"burst_episode_window_seconds:{thresholds.get('burst_episode_window_seconds')},"
            f"burst_episode_hits_required:{thresholds.get('burst_episode_hits_required')},"
            f"burst_episode_min_conf:{thresholds.get('burst_episode_min_conf')},"
            f"burst_episode_max_conf:{thresholds.get('burst_episode_max_conf')},"
            f"burst_episode_min_gap_ms:{thresholds.get('burst_episode_min_gap_ms')},"
            f"burst_episode_max_density_per_second:{thresholds.get('burst_episode_max_density_per_second')},"
            f"burst_episode_promotion_conf:{thresholds.get('burst_episode_promotion_conf')},"
            f"burst_episode_max_span_seconds:{thresholds.get('burst_episode_max_span_seconds')},"
            f"burst_episode_guard_window_seconds:{thresholds.get('burst_episode_guard_window_seconds')},"
            f"burst_episode_guard_max_candidates:{thresholds.get('burst_episode_guard_max_candidates')},"
            f"compact_silence_cluster_min_conf:{thresholds.get('compact_silence_cluster_min_conf')},"
            f"compact_silence_cluster_max_conf:{thresholds.get('compact_silence_cluster_max_conf')},"
            f"compact_silence_cluster_min_silence_ratio:{thresholds.get('compact_silence_cluster_min_silence_ratio')},"
            f"compact_silence_cluster_max_silence_ratio:{thresholds.get('compact_silence_cluster_max_silence_ratio')},"
            f"compact_silence_cluster_min_gap_ms:{thresholds.get('compact_silence_cluster_min_gap_ms')},"
            f"compact_silence_cluster_max_gap_count:{thresholds.get('compact_silence_cluster_max_gap_count')},"
            f"compact_silence_cluster_min_mod_strength:{thresholds.get('compact_silence_cluster_min_mod_strength')},"
            f"compact_silence_cluster_min_mod_depth:{thresholds.get('compact_silence_cluster_min_mod_depth')},"
            f"compact_silence_cluster_min_mod_concentration:{thresholds.get('compact_silence_cluster_min_mod_concentration')},"
            f"compact_silence_cluster_mod_halo_seconds:{thresholds.get('compact_silence_cluster_mod_halo_seconds')},"
            f"compact_silence_cluster_hits_required:{thresholds.get('compact_silence_cluster_hits_required')},"
            f"compact_silence_cluster_max_span_seconds:{thresholds.get('compact_silence_cluster_max_span_seconds')},"
            f"compact_silence_cluster_guard_window_seconds:{thresholds.get('compact_silence_cluster_guard_window_seconds')},"
            f"compact_silence_cluster_guard_max_candidates:{thresholds.get('compact_silence_cluster_guard_max_candidates')},"
            f"compact_silence_cluster_promotion_conf:{thresholds.get('compact_silence_cluster_promotion_conf')},"
            f"subtle_sparse_min_conf:{thresholds.get('subtle_sparse_min_conf')},"
            f"subtle_sparse_max_conf:{thresholds.get('subtle_sparse_max_conf')},"
            f"subtle_sparse_min_silence_ratio:{thresholds.get('subtle_sparse_min_silence_ratio')},"
            f"subtle_sparse_max_silence_ratio:{thresholds.get('subtle_sparse_max_silence_ratio')},"
            f"subtle_sparse_min_gap_ms:{thresholds.get('subtle_sparse_min_gap_ms')},"
            f"subtle_sparse_recent_mod_window_seconds:{thresholds.get('subtle_sparse_recent_mod_window_seconds')},"
            f"subtle_sparse_hits_required:{thresholds.get('subtle_sparse_hits_required')},"
            f"subtle_sparse_max_span_seconds:{thresholds.get('subtle_sparse_max_span_seconds')},"
            f"subtle_sparse_guard_window_seconds:{thresholds.get('subtle_sparse_guard_window_seconds')},"
            f"subtle_sparse_guard_max_candidates:{thresholds.get('subtle_sparse_guard_max_candidates')},"
            f"subtle_sparse_promotion_conf:{thresholds.get('subtle_sparse_promotion_conf')},"
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
        burst_summary, human_bursts, detector_bursts = summarize_burst_alignment(
            rows=rows,
            markers_ms=normalized_markers,
            marker_window_ms=marker_window_ms,
        )
        human_recall_pct = (
            (burst_summary.human_bursts_covered / burst_summary.human_burst_count) * 100.0
            if burst_summary.human_burst_count > 0
            else 0.0
        )
        detector_precision_pct = (
            (burst_summary.detector_bursts_overlapping_human / burst_summary.detector_burst_count) * 100.0
            if burst_summary.detector_burst_count > 0
            else 0.0
        )
        lines.append(
            "burst_alignment="
            f"human_bursts:{burst_summary.human_burst_count},"
            f"detector_bursts:{burst_summary.detector_burst_count},"
            f"human_covered:{burst_summary.human_bursts_covered},"
            f"human_missed:{burst_summary.human_bursts_missed},"
            f"detector_overlap:{burst_summary.detector_bursts_overlapping_human},"
            f"detector_outside:{burst_summary.detector_bursts_outside_human},"
            f"human_recall_pct:{human_recall_pct:.1f},"
            f"detector_precision_pct:{detector_precision_pct:.1f}"
        )
        lines.append(f"human_bursts_ms={format_burst_intervals(human_bursts)}")
        lines.append(f"detector_bursts_ms={format_burst_intervals(detector_bursts)}")

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
            "compact_silence_cluster": sum(1 for r in rows if "compact_silence_cluster" in (r.methods or "")),
            "subtle_sparse_cluster": sum(1 for r in rows if "subtle_sparse_cluster" in (r.methods or "")),
            "subtle_modulation_cluster": sum(1 for r in rows if "subtle_modulation_cluster" in (r.methods or "")),
            "burst_episode_cluster": sum(1 for r in rows if "burst_episode_cluster" in (r.methods or "")),
        }
        promotion_audit = (
            f"persistent_rows:{method_presence['silence_gaps_persistent']},"
            f"silence_rows:{method_presence['silence_gaps']},"
            f"envelope_rows:{method_presence['envelope_discontinuity']},"
            f"mod_rows:{method_presence['amplitude_modulation']},"
            f"compact_silence_rows:{method_presence['compact_silence_cluster']},"
            f"subtle_sparse_rows:{method_presence['subtle_sparse_cluster']},"
            f"subtle_rows:{method_presence['subtle_modulation_cluster']},"
            f"burst_episode_rows:{method_presence['burst_episode_cluster']}"
        )
        alert_cfg = settings.advanced_alert_config if isinstance(settings.advanced_alert_config, dict) else {}

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
                (
                    "promotion_toggles="
                    f"subtle_modulation:{1 if bool(alert_cfg.get('enable_subtle_modulation_promotion', False)) else 0},"
                    f"burst_episode:{1 if bool(alert_cfg.get('enable_burst_episode_promotion', False)) else 0}"
                ),
                f"counted_detection_windows={_format_event_windows(counted_rows)}",
                f"dedup_suppressed_windows={_format_event_windows(dedup_suppressed_rows)}",
                f"counted_cluster_summary={_format_event_clusters(counted_rows)}",
                f"dedup_suppressed_cluster_summary={_format_event_clusters(dedup_suppressed_rows)}",
                f"counted_reason_summary={_summarize_reason_counts(counted_rows)}",
                f"dedup_suppressed_reason_summary={_summarize_reason_counts(dedup_suppressed_rows)}",
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
    baseline_profile: dict | None = None,
    cancel_event: threading.Event | None = None,
) -> PlaygroundAnalysisResult:
    import live_analysis

    _apply_runtime_settings(settings)
    detector = live_analysis.BalancedChoppyDetector(enable_twitch=False)
    detector.sample_rate = int(loaded.sample_rate)
    baseline_source = "learned"
    if apply_baseline_profile_to_detector(detector, baseline_profile):
        baseline_source = "imported"

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
    burst_episode_hits_ms: list[int] = []
    burst_episode_candidate_hits_ms: list[int] = []
    compact_silence_cluster_hits_ms: list[int] = []
    compact_silence_cluster_candidate_hits_ms: list[int] = []
    compact_silence_modulation_feature_hits_ms: list[int] = []
    subtle_sparse_hits_ms: list[int] = []
    subtle_sparse_candidate_hits_ms: list[int] = []
    recent_modulation_hits_ms: list[int] = []
    subtle_modulation_hits_ms: list[int] = []
    last_detection_ms = -10_000_000
    last_detection_signature = ""
    dedup_window_ms = float(live_analysis.ALERT_CONFIG.get("event_dedup_seconds", 0.9)) * 1000.0

    row_index = 1
    max_start = max(0, total_samples - window_samples)
    for start in range(0, max_start + 1, step_samples):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Playground analysis canceled.")
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
        subtle_sparse_promoted = False
        burst_episode_promoted = False
        compact_silence_cluster_promoted = False
        subtle_modulation_promoted = False
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
        mod_result = results.get("amplitude_modulation", {}) if isinstance(results, dict) else {}
        mod_strength = float((mod_result or {}).get("score", 0.0) or 0.0)
        mod_depth = float((mod_result or {}).get("depth", 0.0) or 0.0)
        mod_peak_concentration = float((mod_result or {}).get("peak_concentration", 0.0) or 0.0)
        if modulation_hit:
            recent_modulation_hits_ms.append(start_ms)
        recent_mod_window_ms = max(
            100,
            int(
                round(
                    float(live_analysis.THRESHOLDS.get("subtle_sparse_recent_mod_window_seconds", 1.5))
                    * 1000.0
                )
            ),
        )
        recent_modulation_hits_ms = [
            t for t in recent_modulation_hits_ms if (start_ms - t) <= recent_mod_window_ms
        ]
        persistence_block_long_window = (
            int(window_ms) >= 1600
            and not modulation_hit
            and (
                silence_ratio >= 0.88 and silence_gap_count >= 2
                or confidence >= 0.70
            )
        )
        mod_hit_required_for_persistence = bool(
            live_analysis.THRESHOLDS.get("silence_persistence_require_modulation_hit", True)
        )
        mod_corroborated_for_persistence = (
            (modulation_hit or not mod_hit_required_for_persistence)
            and
            mod_strength >= 4.8
            and mod_depth >= 0.50
            and mod_peak_concentration >= 0.12
        )
        persistence_candidate = (
            silence_choppy
            and not envelope_hit
            and 0.55 <= confidence < 0.75
            and not persistence_block_long_window
            and mod_corroborated_for_persistence
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

        burst_candidate = (
            silence_choppy
            and not envelope_hit
            and 0.64 <= confidence < 0.75
            and 0.52 <= silence_ratio <= 0.85
            and silence_gap_count <= 2
            and mod_strength >= 4.3
            and mod_depth >= 0.45
        )
        burst_require_mod_hit = bool(live_analysis.THRESHOLDS.get("burst_promotion_require_modulation_hit", True))
        burst_is_corroborated = modulation_hit or not burst_require_mod_hit
        burst_cluster_hits_ms = [t for t in burst_cluster_hits_ms if (start_ms - t) <= 2200]
        if burst_candidate and burst_is_corroborated:
            if not burst_cluster_hits_ms or (start_ms - burst_cluster_hits_ms[-1]) >= 120:
                burst_cluster_hits_ms.append(start_ms)
        if len(burst_cluster_hits_ms) >= 3:
            if burst_is_corroborated:
                confidence = max(confidence, 0.77)
                reasons = f"{reasons}; Repeated near-threshold burst cluster"
            else:
                confidence = min(
                    max(confidence, 0.74),
                    float(live_analysis.THRESHOLDS.get("burst_promotion_uncorroborated_cap", 0.74)),
                )
                reasons = f"{reasons}; Uncorroborated burst cluster capped"

        long_window_sparse_candidate = (
            int(window_ms) >= 1600
            and silence_choppy
            and not envelope_hit
            and 0.66 <= confidence < 0.75
            and 0.52 <= silence_ratio <= 0.85
            and silence_max_gap_ms >= 600.0
            and mod_strength >= 4.0
            and mod_depth >= 0.45
            and mod_peak_concentration >= 0.05
        )
        long_window_sparse_require_mod_hit = bool(
            live_analysis.THRESHOLDS.get("long_window_sparse_promotion_require_modulation_hit", True)
        )
        long_window_sparse_corroborated = modulation_hit or not long_window_sparse_require_mod_hit
        long_window_sparse_burst_hits_ms = [
            t for t in long_window_sparse_burst_hits_ms if (start_ms - t) <= 2800
        ]
        if long_window_sparse_candidate and long_window_sparse_corroborated:
            if (
                not long_window_sparse_burst_hits_ms
                or (start_ms - long_window_sparse_burst_hits_ms[-1]) >= 150
            ):
                long_window_sparse_burst_hits_ms.append(start_ms)
        if len(long_window_sparse_burst_hits_ms) >= 2:
            if long_window_sparse_corroborated:
                confidence = max(confidence, 0.76)
                reasons = f"{reasons}; Long-window sparse-gap burst cluster"
            else:
                confidence = min(
                    max(confidence, 0.74),
                    float(live_analysis.THRESHOLDS.get("long_window_sparse_uncorroborated_cap", 0.74)),
                )
                reasons = f"{reasons}; Uncorroborated long-window sparse cluster capped"

        compact_silence_cluster_min_conf = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_min_conf", 0.68)
        )
        compact_silence_cluster_max_conf = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_max_conf", 0.75)
        )
        compact_silence_cluster_min_silence_ratio = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_min_silence_ratio", 0.70)
        )
        compact_silence_cluster_max_silence_ratio = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_max_silence_ratio", 0.82)
        )
        compact_silence_cluster_min_gap_ms = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_min_gap_ms", 500.0)
        )
        compact_silence_cluster_max_gap_count = max(
            0, int(live_analysis.THRESHOLDS.get("compact_silence_cluster_max_gap_count", 1))
        )
        compact_silence_cluster_min_mod_strength = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_min_mod_strength", 3.8)
        )
        compact_silence_cluster_min_mod_depth = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_min_mod_depth", 0.90)
        )
        compact_silence_cluster_min_mod_concentration = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_min_mod_concentration", 0.10)
        )
        compact_silence_cluster_mod_halo_ms = max(
            100,
            int(round(float(live_analysis.THRESHOLDS.get("compact_silence_cluster_mod_halo_seconds", 1.0)) * 1000.0)),
        )
        compact_silence_cluster_hits_required = max(
            2, int(live_analysis.THRESHOLDS.get("compact_silence_cluster_hits_required", 4))
        )
        compact_silence_cluster_max_span_ms = max(
            100,
            int(round(float(live_analysis.THRESHOLDS.get("compact_silence_cluster_max_span_seconds", 2.0)) * 1000.0)),
        )
        compact_silence_cluster_guard_window_ms = max(
            5000,
            int(round(float(live_analysis.THRESHOLDS.get("compact_silence_cluster_guard_window_seconds", 20.0)) * 1000.0)),
        )
        compact_silence_cluster_guard_max_candidates = max(
            2, int(live_analysis.THRESHOLDS.get("compact_silence_cluster_guard_max_candidates", 12))
        )
        compact_silence_cluster_promotion_conf = float(
            live_analysis.THRESHOLDS.get("compact_silence_cluster_promotion_conf", 0.76)
        )
        compact_modulation_window_hit = (
            mod_strength >= compact_silence_cluster_min_mod_strength
            and mod_depth >= compact_silence_cluster_min_mod_depth
            and mod_peak_concentration >= compact_silence_cluster_min_mod_concentration
        )
        if compact_modulation_window_hit:
            compact_silence_modulation_feature_hits_ms.append(start_ms)
        compact_silence_modulation_feature_hits_ms = [
            t for t in compact_silence_modulation_feature_hits_ms if (start_ms - t) <= compact_silence_cluster_guard_window_ms
        ]
        compact_modulation_corroborated = any(
            abs(start_ms - mod_ms) <= compact_silence_cluster_mod_halo_ms
            for mod_ms in compact_silence_modulation_feature_hits_ms
        )
        compact_silence_cluster_candidate = (
            int(window_ms) < 1600
            and silence_choppy
            and not envelope_hit
            and compact_silence_cluster_min_conf <= confidence < compact_silence_cluster_max_conf
            and compact_silence_cluster_min_silence_ratio <= silence_ratio <= compact_silence_cluster_max_silence_ratio
            and silence_max_gap_ms >= compact_silence_cluster_min_gap_ms
            and silence_gap_count <= compact_silence_cluster_max_gap_count
            and compact_modulation_corroborated
        )
        compact_silence_cluster_hits_ms = [
            t for t in compact_silence_cluster_hits_ms if (start_ms - t) <= compact_silence_cluster_guard_window_ms
        ]
        compact_silence_cluster_candidate_hits_ms = [
            t for t in compact_silence_cluster_candidate_hits_ms if (start_ms - t) <= compact_silence_cluster_guard_window_ms
        ]
        if compact_silence_cluster_candidate:
            if (
                not compact_silence_cluster_candidate_hits_ms
                or (start_ms - compact_silence_cluster_candidate_hits_ms[-1]) >= 150
            ):
                compact_silence_cluster_candidate_hits_ms.append(start_ms)
            if (
                not compact_silence_cluster_hits_ms
                or (start_ms - compact_silence_cluster_hits_ms[-1]) >= 150
            ):
                compact_silence_cluster_hits_ms.append(start_ms)
        if len(compact_silence_cluster_candidate_hits_ms) > compact_silence_cluster_guard_max_candidates:
            compact_silence_cluster_hits_ms = []
        compact_silence_cluster_hit_count = len(compact_silence_cluster_hits_ms)
        compact_silence_cluster_span_ms = (
            (compact_silence_cluster_hits_ms[-1] - compact_silence_cluster_hits_ms[0])
            if compact_silence_cluster_hit_count >= 2 else 0
        )
        if (
            compact_silence_cluster_candidate
            and len(compact_silence_cluster_candidate_hits_ms) <= compact_silence_cluster_guard_max_candidates
            and compact_silence_cluster_hit_count >= compact_silence_cluster_hits_required
            and compact_silence_cluster_span_ms <= compact_silence_cluster_max_span_ms
        ):
            confidence = max(confidence, compact_silence_cluster_promotion_conf)
            reasons = f"{reasons}; Compact repeated silence-gap burst"
            compact_silence_cluster_promoted = True

        subtle_sparse_min_conf = float(live_analysis.THRESHOLDS.get("subtle_sparse_min_conf", 0.68))
        subtle_sparse_max_conf = float(live_analysis.THRESHOLDS.get("subtle_sparse_max_conf", 0.75))
        subtle_sparse_min_silence_ratio = float(
            live_analysis.THRESHOLDS.get("subtle_sparse_min_silence_ratio", 0.65)
        )
        subtle_sparse_max_silence_ratio = float(
            live_analysis.THRESHOLDS.get("subtle_sparse_max_silence_ratio", 0.82)
        )
        subtle_sparse_min_gap_ms = float(live_analysis.THRESHOLDS.get("subtle_sparse_min_gap_ms", 500.0))
        subtle_sparse_hits_required = max(1, int(live_analysis.THRESHOLDS.get("subtle_sparse_hits_required", 2)))
        subtle_sparse_max_span_ms = max(
            50,
            int(round(float(live_analysis.THRESHOLDS.get("subtle_sparse_max_span_seconds", 0.8)) * 1000.0)),
        )
        subtle_sparse_guard_window_ms = max(
            5000,
            int(
                round(
                    float(live_analysis.THRESHOLDS.get("subtle_sparse_guard_window_seconds", 45.0))
                    * 1000.0
                )
            ),
        )
        subtle_sparse_guard_max_candidates = max(
            1, int(live_analysis.THRESHOLDS.get("subtle_sparse_guard_max_candidates", 4))
        )
        subtle_sparse_promotion_conf = float(
            live_analysis.THRESHOLDS.get("subtle_sparse_promotion_conf", 0.76)
        )
        subtle_sparse_candidate = (
            int(window_ms) < 1600
            and silence_choppy
            and not envelope_hit
            and subtle_sparse_min_conf <= confidence < subtle_sparse_max_conf
            and subtle_sparse_min_silence_ratio <= silence_ratio <= subtle_sparse_max_silence_ratio
            and silence_max_gap_ms >= subtle_sparse_min_gap_ms
            and bool(recent_modulation_hits_ms)
        )
        subtle_sparse_hits_ms = [t for t in subtle_sparse_hits_ms if (start_ms - t) <= subtle_sparse_guard_window_ms]
        subtle_sparse_candidate_hits_ms = [
            t for t in subtle_sparse_candidate_hits_ms if (start_ms - t) <= subtle_sparse_guard_window_ms
        ]
        if subtle_sparse_candidate:
            if (
                not subtle_sparse_candidate_hits_ms
                or (start_ms - subtle_sparse_candidate_hits_ms[-1]) >= 180
            ):
                subtle_sparse_candidate_hits_ms.append(start_ms)
            if not subtle_sparse_hits_ms or (start_ms - subtle_sparse_hits_ms[-1]) >= 180:
                subtle_sparse_hits_ms.append(start_ms)
        if len(subtle_sparse_candidate_hits_ms) > subtle_sparse_guard_max_candidates:
            subtle_sparse_hits_ms = []
        subtle_sparse_hit_count = len(subtle_sparse_hits_ms)
        subtle_sparse_span_ms = (
            (subtle_sparse_hits_ms[-1] - subtle_sparse_hits_ms[0]) if subtle_sparse_hit_count >= 2 else 0
        )
        if (
            subtle_sparse_candidate
            and len(subtle_sparse_candidate_hits_ms) <= subtle_sparse_guard_max_candidates
            and subtle_sparse_hit_count >= subtle_sparse_hits_required
            and subtle_sparse_span_ms <= subtle_sparse_max_span_ms
        ):
            confidence = max(confidence, subtle_sparse_promotion_conf)
            reasons = f"{reasons}; Subtle sparse low-ambient pattern"
            subtle_sparse_promoted = True

        if bool(live_analysis.ALERT_CONFIG.get("enable_burst_episode_promotion", False)):
            current_window_ms = int(window_ms)
            episode_min_conf = float(live_analysis.THRESHOLDS.get("burst_episode_min_conf", 0.68))
            episode_max_conf = float(live_analysis.THRESHOLDS.get("burst_episode_max_conf", 0.75))
            episode_min_gap_ms = float(live_analysis.THRESHOLDS.get("burst_episode_min_gap_ms", 500.0))
            episode_window_ms = max(
                1000, int(round(float(live_analysis.THRESHOLDS.get("burst_episode_window_seconds", 8.0)) * 1000.0))
            )
            episode_hits_required = max(2, int(live_analysis.THRESHOLDS.get("burst_episode_hits_required", 5)))
            episode_max_density = max(
                0.05, float(live_analysis.THRESHOLDS.get("burst_episode_max_density_per_second", 0.55))
            )
            episode_promotion_conf = float(live_analysis.THRESHOLDS.get("burst_episode_promotion_conf", 0.76))
            episode_max_span_ms = max(
                100, int(round(float(live_analysis.THRESHOLDS.get("burst_episode_max_span_seconds", 3.0)) * 1000.0))
            )
            episode_guard_window_ms = max(
                5000,
                int(round(float(live_analysis.THRESHOLDS.get("burst_episode_guard_window_seconds", 20.0)) * 1000.0)),
            )
            episode_guard_max_candidates = max(
                4, int(live_analysis.THRESHOLDS.get("burst_episode_guard_max_candidates", 10))
            )
            episode_candidate = (
                current_window_ms >= 1600
                and
                silence_choppy
                and not envelope_hit
                and episode_min_conf <= confidence < episode_max_conf
                and silence_max_gap_ms >= episode_min_gap_ms
            )
            burst_episode_hits_ms = [t for t in burst_episode_hits_ms if (start_ms - t) <= episode_window_ms]
            burst_episode_candidate_hits_ms = [
                t for t in burst_episode_candidate_hits_ms if (start_ms - t) <= episode_guard_window_ms
            ]
            if episode_candidate:
                if (
                    not burst_episode_candidate_hits_ms
                    or (start_ms - burst_episode_candidate_hits_ms[-1]) >= 180
                ):
                    burst_episode_candidate_hits_ms.append(start_ms)
                if not burst_episode_hits_ms or (start_ms - burst_episode_hits_ms[-1]) >= 180:
                    burst_episode_hits_ms.append(start_ms)
            if len(burst_episode_candidate_hits_ms) > episode_guard_max_candidates:
                burst_episode_hits_ms = []
            hit_count = len(burst_episode_hits_ms)
            burst_span_ms = (
                (burst_episode_hits_ms[-1] - burst_episode_hits_ms[0]) if hit_count >= 2 else 0
            )
            density_per_second = hit_count / max((episode_window_ms / 1000.0), 0.001)
            if (
                episode_candidate
                and len(burst_episode_candidate_hits_ms) <= episode_guard_max_candidates
                and hit_count >= episode_hits_required
                and burst_span_ms <= episode_max_span_ms
                and density_per_second <= episode_max_density
            ):
                confidence = max(confidence, episode_promotion_conf)
                reasons = f"{reasons}; Sparse burst-episode pattern"
                burst_episode_promoted = True

        if bool(live_analysis.ALERT_CONFIG.get("enable_subtle_modulation_promotion", False)):
            # Subtle low-ambient path:
            # Recover gradual/ambient glitch texture where hard gap detectors stay below threshold.
            if int(window_ms) >= 1600:
                subtle_strength_min = 5.2
                subtle_depth_min = 0.50
                subtle_concentration_min = 0.10
                subtle_hits_required = 2
                subtle_window_ms = 3200
                subtle_spacing_ms = 180
            else:
                subtle_strength_min = 5.8
                subtle_depth_min = 0.50
                subtle_concentration_min = 0.16
                subtle_hits_required = 3
                subtle_window_ms = 3000
                subtle_spacing_ms = 120

            subtle_candidate = (
                not silence_choppy
                and not envelope_hit
                and 0.08 <= silence_ratio <= 0.55
                and mod_strength >= subtle_strength_min
                and mod_depth >= subtle_depth_min
                and mod_peak_concentration >= subtle_concentration_min
            )
            subtle_modulation_hits_ms = [t for t in subtle_modulation_hits_ms if (start_ms - t) <= subtle_window_ms]
            if subtle_candidate:
                if not subtle_modulation_hits_ms or (start_ms - subtle_modulation_hits_ms[-1]) >= subtle_spacing_ms:
                    subtle_modulation_hits_ms.append(start_ms)
            if len(subtle_modulation_hits_ms) >= subtle_hits_required:
                confidence = max(confidence, 0.76)
                if reasons:
                    reasons = f"{reasons}; Subtle low-ambient modulation cluster"
                else:
                    reasons = "Subtle low-ambient modulation cluster"
                subtle_modulation_promoted = True

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
        if subtle_sparse_promoted:
            active_methods = tuple(sorted(set(active_methods) | {"subtle_sparse_cluster"}))
        if compact_silence_cluster_promoted:
            active_methods = tuple(sorted(set(active_methods) | {"compact_silence_cluster"}))
        if burst_episode_promoted:
            active_methods = tuple(sorted(set(active_methods) | {"burst_episode_cluster"}))
        if subtle_modulation_promoted:
            active_methods = tuple(sorted(set(active_methods) | {"subtle_modulation_cluster"}))
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
        baseline_source=baseline_source,
    )


def apply_baseline_profile_to_detector(detector, baseline_profile: dict | None) -> bool:
    if not isinstance(baseline_profile, dict):
        return False
    baseline = baseline_profile.get("baseline")
    if not isinstance(baseline, dict):
        return False

    rms_history_raw = baseline.get("rms_history")
    normalized_history: list[float] = []
    if isinstance(rms_history_raw, list):
        for item in rms_history_raw:
            try:
                value = float(item)
            except Exception:
                continue
            if math.isfinite(value) and value > 0.0:
                normalized_history.append(value)
    normalized_history = normalized_history[-50:]

    try:
        established = bool(int(baseline.get("established", 0)))
    except Exception:
        established = False

    baseline_rms = baseline.get("rms")
    if not normalized_history and baseline_rms is not None:
        try:
            fallback_rms = float(baseline_rms)
        except Exception:
            fallback_rms = 0.0
        if math.isfinite(fallback_rms) and fallback_rms > 0.0:
            normalized_history = [fallback_rms] * 8

    if not normalized_history:
        return False

    with detector.lock:
        detector.baseline_stats["rms_history"].clear()
        detector.baseline_stats["rms_history"].extend(normalized_history)
        detector.baseline_stats["established_baseline"] = established
        detector.baseline_stats["learning_started_at"] = 0.0
        detector.baseline_stats["last_blocked_at"] = 0.0
    return True


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


def summarize_burst_alignment(
    *,
    rows: list[PlaygroundTelemetryRow],
    markers_ms: list[int],
    marker_window_ms: int,
) -> tuple[BurstAlignmentSummary, list[BurstInterval], list[BurstInterval]]:
    marker_intervals = [
        BurstInterval(
            start_ms=max(0, marker - marker_window_ms),
            end_ms=max(0, marker + marker_window_ms),
        )
        for marker in markers_ms
    ]
    human_bursts = merge_burst_intervals(marker_intervals, bridge_ms=max(0, int(marker_window_ms)))

    detector_intervals = [
        BurstInterval(start_ms=int(row.start_ms), end_ms=int(row.end_ms))
        for row in rows
        if row.deduped_detection
    ]
    detector_bursts = merge_burst_intervals(detector_intervals, bridge_ms=1000)

    human_bursts_covered = sum(
        1 for hb in human_bursts if any(intervals_overlap(hb, db) for db in detector_bursts)
    )
    human_bursts_missed = max(0, len(human_bursts) - human_bursts_covered)

    detector_bursts_overlapping_human = sum(
        1 for db in detector_bursts if any(intervals_overlap(db, hb) for hb in human_bursts)
    )
    detector_bursts_outside_human = max(0, len(detector_bursts) - detector_bursts_overlapping_human)

    return (
        BurstAlignmentSummary(
            human_burst_count=len(human_bursts),
            detector_burst_count=len(detector_bursts),
            human_bursts_covered=human_bursts_covered,
            human_bursts_missed=human_bursts_missed,
            detector_bursts_overlapping_human=detector_bursts_overlapping_human,
            detector_bursts_outside_human=detector_bursts_outside_human,
        ),
        human_bursts,
        detector_bursts,
    )


def merge_burst_intervals(
    intervals: list[BurstInterval],
    *,
    bridge_ms: int,
) -> list[BurstInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda it: (it.start_ms, it.end_ms))
    merged: list[BurstInterval] = [BurstInterval(start_ms=ordered[0].start_ms, end_ms=ordered[0].end_ms)]
    for current in ordered[1:]:
        tail = merged[-1]
        if current.start_ms <= (tail.end_ms + max(0, int(bridge_ms))):
            tail.end_ms = max(tail.end_ms, current.end_ms)
            continue
        merged.append(BurstInterval(start_ms=current.start_ms, end_ms=current.end_ms))
    return merged


def intervals_overlap(a: BurstInterval, b: BurstInterval) -> bool:
    return a.start_ms <= b.end_ms and b.start_ms <= a.end_ms


def format_burst_intervals(intervals: list[BurstInterval], *, max_items: int = 80) -> str:
    if not intervals:
        return "none"
    shown = intervals[:max_items]
    text = ";".join(f"{int(it.start_ms)}-{int(it.end_ms)}" for it in shown)
    if len(intervals) > max_items:
        text = f"{text};...(+{len(intervals) - max_items})"
    return text


def marker_sidecar_path(wav_path: str | Path) -> Path:
    path = Path(wav_path).expanduser()
    return path.with_suffix(path.suffix + ".markers.json")


def baseline_sidecar_path(wav_path: str | Path) -> Path:
    path = Path(wav_path).expanduser()
    return path.with_suffix(path.suffix + ".baseline.json")


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


def load_baseline_sidecar(wav_path: str | Path) -> dict | None:
    sidecar = baseline_sidecar_path(wav_path)
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        return None
    return payload


def save_baseline_sidecar(wav_path: str | Path, payload: dict) -> Path:
    sidecar = baseline_sidecar_path(wav_path)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def write_mono_wav_file(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    wav_path = Path(path).expanduser()
    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    if pcm.size == 0:
        raise ValueError("No audio samples to write")
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = np.round(pcm * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(max(1, sample_rate)))
        wav.writeframes(pcm_i16.tobytes())
    return wav_path
