"""Playground tab UI for offline WAV detection tuning."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


def build_playground_tab(window) -> None:
    tab = QWidget()
    layout = QVBoxLayout(tab)
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
        "Use Browse or paste a full path, then click Load."
    )
    file_row.addWidget(window.playground_file_path, stretch=1)
    window.playground_browse_button = QPushButton("Browse")
    window.playground_browse_button.clicked.connect(window.browse_playground_file)
    window.playground_browse_button.setToolTip("Open a file picker to choose a WAV file.")
    file_row.addWidget(window.playground_browse_button)
    window.playground_load_button = QPushButton("Load")
    window.playground_load_button.clicked.connect(window.load_playground_file_clicked)
    window.playground_load_button.setToolTip(
        "Load the selected WAV file and show file details in the Playground."
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
    window.playground_window_ms_spin.setValue(1000)
    window.playground_window_ms_spin.setToolTip(
        "Analysis window size in milliseconds.\n"
        "Larger windows smooth noise; smaller windows catch short bursts."
    )
    controls_layout.addWidget(QLabel("Window (ms)"), 0, 2)
    controls_layout.addWidget(window.playground_window_ms_spin, 0, 3)

    window.playground_step_ms_spin = QSpinBox()
    window.playground_step_ms_spin.setRange(10, 1000)
    window.playground_step_ms_spin.setSingleStep(10)
    window.playground_step_ms_spin.setValue(50)
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

    window.playground_analyze_button = QPushButton("Analyze File")
    window.playground_analyze_button.clicked.connect(window.run_playground_analysis)
    window.playground_analyze_button.setToolTip(
        "Run offline detector analysis on the loaded WAV file and save a compact report."
    )
    controls_layout.addWidget(window.playground_analyze_button, 1, 2, 1, 2)

    window.playground_preview_button = QPushButton("Preview Sound")
    window.playground_preview_button.clicked.connect(window.toggle_playground_preview)
    window.playground_preview_button.setToolTip(
        "Toggle WAV preview playback.\n"
        "Click once to play, click again to stop."
    )
    controls_layout.addWidget(window.playground_preview_button, 1, 4, 1, 3)

    window.playground_playback_status = QLabel("Not playing")
    controls_layout.addWidget(window.playground_playback_status, 1, 7, 1, 1)

    window.playground_live_start_button = QPushButton("Start Live Report")
    window.playground_live_start_button.clicked.connect(window.start_live_playground_report)
    window.playground_live_start_button.setToolTip(
        "Start recording live audio from the currently selected input device/channel."
    )
    controls_layout.addWidget(window.playground_live_start_button, 2, 0, 1, 2)

    window.playground_live_stop_button = QPushButton("Stop & Save Live Report")
    window.playground_live_stop_button.clicked.connect(window.stop_live_playground_report)
    window.playground_live_stop_button.setToolTip(
        "Stop live recording, analyze captured audio, and save a compact report."
    )
    controls_layout.addWidget(window.playground_live_stop_button, 2, 2, 1, 3)

    window.playground_live_status = QLabel("Live report idle")
    controls_layout.addWidget(window.playground_live_status, 2, 5, 1, 3)

    layout.addWidget(controls)

    window.playground_progress = QProgressBar()
    window.playground_progress.setRange(0, 1000)
    window.playground_progress.setValue(0)
    window.playground_progress.setToolTip("Playback progress for WAV file mode.")
    layout.addWidget(window.playground_progress)

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
    layout.addWidget(window.playground_table, stretch=1)

    window.tabs.addTab(tab, "Playground")
