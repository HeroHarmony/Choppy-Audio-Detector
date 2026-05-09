"""Support tab UI builder."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def build_support_tab(window) -> None:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setSpacing(10)

    title = QLabel("Support and Links")
    title.setStyleSheet("font-size: 18px; font-weight: 600;")
    layout.addWidget(title)

    github_link = QLabel('<a href="https://github.com/HeroHarmony/Choppy-Audio-Detector">GitHub Repository</a>')
    github_link.setOpenExternalLinks(True)
    layout.addWidget(github_link)

    twitch_link = QLabel('<a href="https://www.twitch.tv/heroharmony">HeroHarmony on Twitch</a>')
    twitch_link.setOpenExternalLinks(True)
    layout.addWidget(twitch_link)

    support_link = QLabel('<a href="https://streamelements.com/heroharmony/tip">Buy Me a Coffee / Support</a>')
    support_link.setOpenExternalLinks(True)
    layout.addWidget(support_link)

    layout.addStretch(1)
    window.tabs.addTab(tab, "Support")
