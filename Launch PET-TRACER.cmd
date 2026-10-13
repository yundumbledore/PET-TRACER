@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch_gui.py
) else (
    python launch_gui.py
)
if errorlevel 1 pause
