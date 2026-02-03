@echo off
REM ============================================================================
REM NKREPO Flask Backend - Windows Setup Script
REM ============================================================================

echo.
echo ========================================================================
echo NKREPO Flask Backend Setup for Windows
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [*] Python found:
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not installed
    echo Please install pip or reinstall Python with pip included
    pause
    exit /b 1
)

echo [*] pip found:
pip --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [+] Virtual environment created
) else (
    echo [*] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo [*] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
    echo You can continue, but some features may not work
    pause
)

REM Check for config.ini
if not exist "config.ini" (
    echo.
    echo [WARNING] config.ini not found!
    echo.
    if exist "config.ini.example" (
        echo [*] Creating config.ini from template...
        copy config.ini.example config.ini
        echo.
        echo [!] IMPORTANT: Please edit config.ini and configure:
        echo     - MySQL connection settings
        echo     - VirusTotal API key
        echo     - File paths for your system
        echo.
        echo Press any key to open config.ini in notepad...
        pause >nul
        notepad config.ini
    ) else (
        echo [ERROR] config.ini.example template not found
        echo Please create config.ini manually
        pause
        exit /b 1
    )
)

REM Create necessary directories
echo.
echo [*] Creating necessary directories...

if not exist "..\vue\uploads\" (
    mkdir "..\vue\uploads"
    echo [+] Created uploads directory
)

if not exist "training_data\" (
    mkdir "training_data"
    echo [+] Created training_data directory
)

if not exist "logs\" (
    mkdir "logs"
    echo [+] Created logs directory
)

echo.
echo ========================================================================
echo Setup Complete!
echo ========================================================================
echo.
echo Next steps:
echo   1. Configure config.ini with your settings
echo   2. Set up MySQL database (see database setup guide)
echo   3. Run the Flask server with: run_flask.bat
echo.
echo Virtual environment activated. To deactivate, type: deactivate
echo.
pause
