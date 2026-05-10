# Playground Marker-Assisted Analysis Plan

## Goal
Improve Playground so human listening can be compared against detector output without treating human markers as perfect ground truth.

## Current constraints
- Playground analysis uses a fresh detector instance (`BalancedChoppyDetector(enable_twitch=False)`) each run.
- It does not reuse live/prod runtime baseline state.
- Reports are compact and currently do not include explicit baseline-profile summary.

## Design principles
- Human markers are **strong hints**, not absolute truth.
- Missing markers do **not** imply no glitches and do **not** imply continuous glitches.
- Comparison metrics should measure alignment quality, not pass/fail correctness.
- Marker data should be shareable across machines/users via sidecar file.

## Phase 1 (foundation)
1. Marker data model and persistence
- Marker model: timestamp in ms + optional note.
- Sidecar file per WAV in same folder: `<sample>.markers.json`.
- Auto-load markers when WAV is loaded.
- Save markers on add/remove/clear.

2. Playback-time marker workflow
- Add buttons: `Add Marker`, `Clear Markers`.
- Add latency compensation setting (default 150 ms) applied on marker creation.
- Marker timestamp clamped into `[0, duration_ms]`.

3. Report enrichment
- Always include baseline summary captured during analysis (e.g., median baseline RMS, sample count, lock status).
- If markers exist, include alignment metrics:
  - marker_count
  - marker_window_ms (matching radius)
  - marker_hits
  - marker_misses
  - outside_marker_hits
  - hit_rate / miss_rate style derived metrics
- Include a compact marker snapshot in each report for portability:
  - marker metadata (provided flag, marker count, latency setting, match window setting)
  - marker points list (timestamp ms), truncated if very long
- If no markers, report `marker_alignment=not_provided`.

## Phase 2 (visual UX)
1. Replace plain progress bar with marker-aware progress widget
- Draw marker dots/circles on bar.
- Flash/red highlight when playhead enters marker zone.
- Click marker dot to remove it.

2. Optional marker list panel
- Show sorted marker timestamps.
- Jump-to marker and delete actions.

## Phase 3 (advanced comparison)
1. Marker cluster comparison
- Group nearby markers into bursts.
- Compare detector burst coverage vs marker bursts.

2. Analyst confidence lanes
- Optional marker severity/confidence tags for richer tuning datasets.

## Open questions
- Marker sidecar + report embedding is the default approach (both, not either/or).
- Should latency compensation be global setting or Playground-only control?

## Initial implementation scope for this branch
- Phase 1 only, in incremental patches.
