@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo Installing runtime dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install runtime dependencies.
  pause
  exit /b 1
)

echo Installing build dependencies...
".venv\Scripts\python.exe" -m pip install pyinstaller
if errorlevel 1 (
  echo Failed to install pyinstaller.
  pause
  exit /b 1
)

echo Building Windows executable...
".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --name ChoppyAudioDetector ^
  --hidden-import live_analysis ^
  app_gui.py
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo Build complete: dist\ChoppyAudioDetector\ChoppyAudioDetector.exe
endlocal
