@echo off
cd /d "%~dp0"
if defined PET_TRACER_PYTHON (
    "%PET_TRACER_PYTHON%" launch_gui.py
) else if exist "%USERPROFILE%\.venvs\pet-tracer\Scripts\python.exe" (
    "%USERPROFILE%\.venvs\pet-tracer\Scripts\python.exe" launch_gui.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch_gui.py
) else (
    python launch_gui.py
)
if errorlevel 1 pause
