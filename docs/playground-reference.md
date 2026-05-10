# Playground Reference

This page documents the Playground tab workflows for offline WAV testing and live-capture reporting.

## Purpose

Playground is for detector calibration and verification. It lets you:

- Analyze local WAV files with configurable `window/step/warm-up`.
- Run optional second-pass analysis using production timing.
- Preview audio before/after analysis.
- Create compact report files in `Reports/`.
- Capture live audio and export it as a report using the same detector pipeline.

## Controls

- `WAV path` + `Browse` + `Load`: Select and load a local WAV file.
- `Channel`: 1-based channel selector for multi-channel WAV input.
- `Window (ms)`: Analysis window length in milliseconds.
- `Step (ms)`: Sliding step interval in milliseconds.
- `Warm-up Ignore (ms)`: Initial analysis duration where detections are suppressed.
- `Preview on done`: If enabled, starts preview playback when analysis completes.
- `Also run prod timing`: If enabled and current timing differs from production timing, writes a second report using production timing.
- `Analyze File`: Runs offline analysis for the loaded file.
- `Preview Sound`: Toggle playback preview for loaded WAV.
- `Start Live Report` / `Stop & Save Live Report`: Capture current monitored audio and generate report(s).

## Analyze File Workflow

1. Load a WAV file.
2. Set `Channel`, `Window`, `Step`, and `Warm-up Ignore`.
3. Click `Analyze File`.
4. Expected outcome is resolved:
- Auto from filename tag if present (see below).
- Otherwise from popup buttons (`No glitch detected` / `Glitch detected`).
5. Report is finalized only after expected outcome is known.
6. If `Also run prod timing` is enabled and timing differs, a second report is generated.

## Filename Auto-Labeling

To skip the expected-outcome popup, include one of these tags in the WAV filename:

- `[no glitch]` -> expected clean
- `[glitchy]` -> expected glitch

Notes:

- Matching is case-insensitive.
- Minor separator variation inside the tag is accepted (space, `_`, or `-`), for example `[No_Glitch]`.
- If no recognized tag is present, the popup is shown.

## Live Report Workflow

1. Select a monitorable input/capture device on the `Main` tab.
2. In Playground, click `Start Live Report`.
3. When ready, click `Stop & Save Live Report`.
4. Captured audio is analyzed and expected outcome is requested via popup.
5. Report is written to `Reports/` (and optional prod-timing second report).

## Report Files

Reports are saved in:

- `Reports/`

Filename format:

- `w<window>_s<step>_<sample>.report.txt`

Examples:

- `w1000_s50_sample 7 - dane.report.txt`
- `w2000_s200_sample 7 - dane.report.txt`

Each report includes:

- Input metadata (source/file, sample rate, channels, duration, channel analyzed).
- Runtime settings (window, step, warm-up, expected label).
- Outcome evaluation (`TP`, `TN`, `FP`, `FN`).
- Confidence and method summaries.
- Per-window row data for deeper inspection.

## Production Timing Source of Truth

Playground production timing is not magic-numbered separately. It is sourced from runtime detection settings so future production timing changes apply to both:

- live detection
- Playground `Also run prod timing` reports

## Console and Events

Playground writes informational lines to the in-app Console for:

- analysis start/complete
- expectation check result
- report save path
- auto-tag expectation resolution

Playground analysis is local workflow instrumentation; it does not emit Twitch/OBS alert actions unless separately triggered by live runtime conditions outside Playground.
