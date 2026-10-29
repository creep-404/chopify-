@echo off
REM Chopify Web UI Launcher for Windows
REM Activates .venv and starts the Gradio web interface at http://127.0.0.1:7860

cd /d "%~dp0"

echo =============================================
echo   Chopify Web UI Launcher
echo   Local viral clip generator - 100% offline
echo =============================================
echo.

REM Check if .venv exists
if not exist ".venv" (
    echo [ERROR] Virtual environment not found.
    echo Please run the install steps first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Check if Ollama is running
echo Checking Ollama...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama doesn't seem to be running.
    echo The UI will still work with the built-in heuristic scorer.
    echo To use AI scoring, start Ollama first: ollama serve
    echo.
) else (
    echo [OK] Ollama is running.
    echo.
)

REM Check ffmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ffmpeg not found on PATH.
    echo Install with: winget install Gyan.FFmpeg
    echo.
) else (
    echo [OK] ffmpeg found.
    echo.
)

echo Starting Chopify Web UI...
echo URL: http://127.0.0.1:7860
echo Press Ctrl+C to stop.
echo =============================================
echo.

REM Activate venv and run UI
call .venv\Scripts\activate
python ui.py

echo.
echo UI stopped.
pause