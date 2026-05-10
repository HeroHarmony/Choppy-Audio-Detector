"""Playground tab UI for offline WAV detection tuning."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from choppy_detector_gui.gui.widgets.obs_level_meter import ObsLevelMeter
from choppy_detector_gui.gui.widgets.marker_progress_bar import MarkerProgressBar


def build_playground_tab(window) -> None:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    layout.addWidget(scroll)

    content = QWidget()
    scroll.setWidget(content)
    layout = QVBoxLayout(content)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)

    intro = QLabel(
        "Use this tab to replay local PCM WAV files and inspect detector telemetry per window."
    )
    intro.setWordWrap(True)
    intro.setStyleSheet("color: #bdbdbd;")
    layout.addWidget(intro)

    file_row = QHBoxLayout()
    window.playground_file_path = QLineEdit()
    window.playground_file_path.setPlaceholderText("Select a .wav file to test")
    window.playground_file_path.setToolTip(
        "Path to a local PCM WAV file for offline analysis.\n"
        "Use Browse for single-file mode, Browse Batch for multi-file mode,\n"
        "or paste one or more full paths separated by | or ,"
    )
    file_row.addWidget(window.playground_file_path, stretch=1)
    window.playground_browse_button = QPushButton("Browse")
    window.playground_browse_button.clicked.connect(window.browse_playground_file)
    window.playground_browse_button.setToolTip("Open a file picker to choose a WAV file.")
    file_row.addWidget(window.playground_browse_button)
    window.playground_load_button = QPushButton("Browse Batch...")
    window.playground_load_button.clicked.connect(window.browse_playground_files_batch)
    window.playground_load_button.setToolTip(
        "Open a file picker to choose multiple WAV files for batch analysis."
    )
    file_row.addWidget(window.playground_load_button)
    layout.addLayout(file_row)

    controls = QWidget()
    controls_layout = QGridLayout(controls)
    controls_layout.setContentsMargins(0, 0, 0, 0)
    controls_layout.setHorizontalSpacing(10)
    controls_layout.setVerticalSpacing(6)

    window.playground_channel_spin = QSpinBox()
    window.playground_channel_spin.setRange(1, 1)
    window.playground_channel_spin.setValue(1)
    window.playground_channel_spin.setToolTip(
        "Audio channel to analyze from the loaded WAV file.\n"
        "Use channel 1 for mono files."
    )
    controls_layout.addWidget(QLabel("Channel"), 0, 0)
    controls_layout.addWidget(window.playground_channel_spin, 0, 1)

    window.playground_window_ms_spin = QSpinBox()
    window.playground_window_ms_spin.setRange(100, 4000)
    window.playground_window_ms_spin.setSingleStep(50)
    prod_window_ms, prod_step_ms = window._playground_prod_timing()
    window.playground_window_ms_spin.setValue(prod_window_ms)
    window.playground_window_ms_spin.setToolTip(
        "Analysis window size in milliseconds.\n"
        "Larger windows smooth noise; smaller windows catch short bursts."
    )
    controls_layout.addWidget(QLabel("Window (ms)"), 0, 2)
    controls_layout.addWidget(window.playground_window_ms_spin, 0, 3)

    window.playground_step_ms_spin = QSpinBox()
    window.playground_step_ms_spin.setRange(10, 1000)
    window.playground_step_ms_spin.setSingleStep(10)
    window.playground_step_ms_spin.setValue(prod_step_ms)
    window.playground_step_ms_spin.setToolTip(
        "How far each analysis window moves forward (ms).\n"
        "Smaller step = finer timing detail, more rows."
    )
    controls_layout.addWidget(QLabel("Step (ms)"), 0, 4)
    controls_layout.addWidget(window.playground_step_ms_spin, 0, 5)

    window.playground_warmup_ms_spin = QSpinBox()
    window.playground_warmup_ms_spin.setRange(0, 10_000)
    window.playground_warmup_ms_spin.setSingleStep(50)
    window.playground_warmup_ms_spin.setValue(700)
    window.playground_warmup_ms_spin.setToolTip(
        "Ignore detections in the first N ms.\n"
        "Useful to avoid start-of-playback spikes and false positives."
    )
    controls_layout.addWidget(QLabel("Warm-up Ignore (ms)"), 0, 6)
    controls_layout.addWidget(window.playground_warmup_ms_spin, 0, 7)

    window.playground_preview_on_done = QCheckBox("Preview on done")
    window.playground_preview_on_done.setChecked(True)
    window.playground_preview_on_done.setToolTip(
        "When enabled, starts WAV preview after analysis finishes,\n"
        "while you choose expected outcome in the result popup."
    )
    controls_layout.addWidget(window.playground_preview_on_done, 1, 0, 1, 2)

    window.playground_also_prod_timing = QCheckBox("Also run 2000/200")
    window.playground_also_prod_timing.setChecked(True)
    window.playground_also_prod_timing.setToolTip(
        "When enabled, generates an additional report using\n"
        "legacy comparison timing (window 2000 ms, step 200 ms)."
    )
    controls_layout.addWidget(window.playground_also_prod_timing, 1, 2, 1, 2)

    window.playground_extended_report = QCheckBox("Extended report")
    window.playground_extended_report.setChecked(True)
    window.playground_extended_report.setToolTip(
        "When enabled, include deeper diagnostics in report output\n"
        "(near-threshold windows, clusters, feature ranges, and transition summaries)."
    )
    controls_layout.addWidget(window.playground_extended_report, 1, 4, 1, 2)

    window.playground_analyze_button = QPushButton("Analyze File")
    window.playground_analyze_button.clicked.connect(window.run_playground_analysis)
    window.playground_analyze_button.setToolTip(
        "Run offline detector analysis on the loaded WAV file and save a compact report."
    )
    controls_layout.addWidget(window.playground_analyze_button, 1, 6, 1, 2)

    layout.addWidget(controls)

    live_row = QHBoxLayout()
    live_row.setSpacing(8)
    window.playground_live_start_button = QPushButton("Start Live Report")
    window.playground_live_start_button.clicked.connect(window.start_live_playground_report)
    window.playground_live_start_button.setToolTip(
        "Start recording live audio from the currently selected input device/channel.\n"
        "Main tab monitoring is paused during recording and resumed after save."
    )
    live_row.addWidget(window.playground_live_start_button)

    window.playground_live_stop_button = QPushButton("Stop && Save Live WAV")
    window.playground_live_stop_button.clicked.connect(window.stop_live_playground_report)
    window.playground_live_stop_button.setToolTip(
        "Stop live recording and save a WAV file.\n"
        "A baseline profile sidecar is also saved next to the WAV when available."
    )
    live_row.addWidget(window.playground_live_stop_button)

    window.playground_live_status = QLabel("Live report idle")
    window.playground_live_status.setMinimumWidth(210)
    window.playground_live_status.setMaximumWidth(260)
    window.playground_live_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    window.playground_live_status.setStyleSheet("padding-left: 8px; padding-right: 8px;")
    live_row.addWidget(window.playground_live_status)
    live_row.addStretch(1)
    layout.addLayout(live_row)

    window.playground_peak_meter = ObsLevelMeter(show_ruler=False, overlay_label="")
    layout.addWidget(window.playground_peak_meter)

    window.playground_progress = MarkerProgressBar()
    window.playground_progress.set_duration_ms(1)
    window.playground_progress.set_position_ms(0)
    window.playground_progress.markerClicked.connect(window.remove_playground_marker_at)
    window.playground_progress.seekRequested.connect(window.seek_playground_preview)
    window.playground_progress.setToolTip(
        "Playback progress for WAV mode.\n"
        "Click the bar to seek preview.\n"
        "Click a marker to remove it."
    )
    layout.addWidget(window.playground_progress)

    marker_row = QHBoxLayout()
    marker_row.setSpacing(6)

    window.playground_preview_button = QPushButton("Preview Sound")
    window.playground_preview_button.clicked.connect(window.toggle_playground_preview)
    window.playground_preview_button.setToolTip(
        "Toggle WAV preview playback.\n"
        "Click once to play, click again to stop."
    )
    window.playground_preview_button.setMaximumWidth(130)
    marker_row.addWidget(window.playground_preview_button)

    window.playground_add_marker_button = QPushButton("Add Choppy Marker")
    window.playground_add_marker_button.clicked.connect(window.add_playground_marker)
    window.playground_add_marker_button.setToolTip(
        "While preview is playing, add a marker where you hear an obvious glitch burst."
    )
    window.playground_add_marker_button.setMaximumWidth(150)
    marker_row.addWidget(window.playground_add_marker_button)

    window.playground_clear_markers_button = QPushButton("Clear Markers")
    window.playground_clear_markers_button.clicked.connect(window.clear_playground_markers)
    window.playground_clear_markers_button.setToolTip("Remove all markers for the currently loaded WAV file.")
    window.playground_clear_markers_button.setMaximumWidth(115)
    marker_row.addWidget(window.playground_clear_markers_button)

    marker_row.addWidget(QLabel("Latency"))
    window.playground_marker_latency_ms_spin = QSpinBox()
    window.playground_marker_latency_ms_spin.setRange(0, 1000)
    window.playground_marker_latency_ms_spin.setValue(270)
    window.playground_marker_latency_ms_spin.setSingleStep(10)
    window.playground_marker_latency_ms_spin.setSuffix(" ms")
    window.playground_marker_latency_ms_spin.setToolTip(
        "Compensate human click reaction time when adding markers."
    )
    window.playground_marker_latency_ms_spin.setMaximumWidth(95)
    marker_row.addWidget(window.playground_marker_latency_ms_spin)

    marker_row.addWidget(QLabel("Match"))
    window.playground_marker_match_ms_spin = QSpinBox()
    window.playground_marker_match_ms_spin.setRange(0, 2000)
    window.playground_marker_match_ms_spin.setValue(450)
    window.playground_marker_match_ms_spin.setSingleStep(10)
    window.playground_marker_match_ms_spin.setSuffix(" ms")
    window.playground_marker_match_ms_spin.setToolTip(
        "Marker comparison radius used in report alignment metrics."
    )
    window.playground_marker_match_ms_spin.setMaximumWidth(95)
    marker_row.addWidget(window.playground_marker_match_ms_spin)

    marker_row.addStretch(1)
    window.playground_playback_status = QLabel("Not playing")
    marker_row.addWidget(window.playground_playback_status)
    window.playground_marker_status = QLabel("Markers: 0")
    window.playground_marker_status.setStyleSheet("color: #bdbdbd;")
    marker_row.addWidget(window.playground_marker_status)
    layout.addLayout(marker_row)

    file_label = QLabel("File")
    file_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(file_label)
    window.playground_file_info = QPlainTextEdit("No file loaded")
    window.playground_file_info.setReadOnly(True)
    window.playground_file_info.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    window.playground_file_info.setMaximumHeight(58)
    window.playground_file_info.setToolTip("Loaded file metadata.")
    layout.addWidget(window.playground_file_info)

    summary_label = QLabel("Summary")
    summary_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(summary_label)
    window.playground_analysis_summary = QPlainTextEdit("No analysis yet")
    window.playground_analysis_summary.setReadOnly(True)
    window.playground_analysis_summary.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    window.playground_analysis_summary.setMaximumHeight(76)
    window.playground_analysis_summary.setToolTip("Most recent analysis/result summary.")
    layout.addWidget(window.playground_analysis_summary)

    window.playground_table = QTableWidget()
    window.playground_table.setToolTip(
        "Per-window telemetry output.\n"
        "Each row is one analysis window with confidence, methods, and reason fields."
    )
    window.playground_table.setColumnCount(19)
    window.playground_table.setHorizontalHeaderLabels(
        [
            "#",
            "Start ms",
            "End ms",
            "RMS dBFS",
            "Conf %",
            "High Conf",
            "Primary",
            "Dedup Hit",
            "Warm-up",
            "Methods",
            "Silence",
            "Gaps",
            "Max Gap ms",
            "Env Score",
            "Mod Strength",
            "Mod Hz",
            "Mod Depth",
            "Mod Conc",
            "Reasons",
        ]
    )
    window.playground_table.verticalHeader().setVisible(False)
    window.playground_table.setAlternatingRowColors(True)
    window.playground_table.setSortingEnabled(False)
    window.playground_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    window.playground_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    layout.addWidget(window.playground_table, stretch=1)

    window.tabs.addTab(tab, "Playground")
