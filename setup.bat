@echo off
echo ============================================================
echo      FACE DETECTION SYSTEM - WINDOWS INSTALLER
echo ============================================================
echo.
echo [*] Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [-] CRITICAL: Python was not found in your Windows PATH environment.
    echo Please install Python 3.8+ (ensure 'Add Python to PATH' is checked) and retry.
    pause
    exit /b 1
)

echo [*] Step 2: Creating Python virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [-] Failed to initialize virtual environment.
        pause
        exit /b 1
    )
    echo [+] Virtual environment initialized.
) else (
    echo [~] Existing .venv detected. Skipping creation.
)

echo [*] Step 3: Upgrading Pip and installing packages...
.venv\Scripts\python -m pip install --upgrade pip >nul 2>&1
.venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [-] Dependency installation failed. Check internet connection and console logs.
    pause
    exit /b 1
)

echo [+] Setup complete! Launching server...
echo.
.venv\Scripts\python app.py
pause
