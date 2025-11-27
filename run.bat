@echo off
setlocal

rem -----------------------------------------------------------------------------
rem YomiToku OCR server launcher (Windows, venv)
rem - Creates/uses .venv next to this script
rem - Installs runtime deps if missing
rem - Starts FastAPI (LAN accessible) on configured host/port
rem -----------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\.venv"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [+] Creating virtual environment at "%VENV_DIR%" ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [!] Failed to create virtual environment. Ensure Python 3.10+ is installed and on PATH.
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [!] Failed to activate virtual environment.
    exit /b 1
)

echo [+] Installing/ensuring dependencies...
python -m pip install --upgrade pip >nul
python -m pip install -e "%SCRIPT_DIR%" fastapi uvicorn python-multipart >nul

rem Configure host/port. Override by setting HOST/PORT before running.
if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"

echo [+] Starting server at http://%HOST%:%PORT%
uvicorn app.main:app --host "%HOST%" --port "%PORT%" --reload

endlocal
