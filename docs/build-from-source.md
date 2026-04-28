# Build and Packaging Guide

This guide is for maintainers or advanced users who want to build desktop packages locally.

## Prerequisites

- Python 3.x
- `pip`

## Windows

Run from repository root:

```bat
build_windows.bat
```

Output:

- `dist\ChoppyAudioDetector\ChoppyAudioDetector.exe`

## macOS

Run from repository root:

```bash
./build_macos.command
```

Output:

- `dist/ChoppyAudioDetector.app`

## Notes

- Builds use PyInstaller with windowed mode (no terminal window on launch).
- Packaged outputs are intended for end-user distribution.
