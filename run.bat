@echo off
title Face Detection v2 - Biometric Verification Server (Port 5000)
cd /d "%~dp0"
echo ============================================================
echo   FACE DETECTION V2 - BIOMETRIC VERIFICATION (PORT 5000)
echo ============================================================
echo.
echo Local Live URL: http://localhost:5000
echo.
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe app.py
) else (
    python app.py
)
pause
