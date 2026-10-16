@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
set "PET_SETUP_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %PET_SETUP_RESULT%
