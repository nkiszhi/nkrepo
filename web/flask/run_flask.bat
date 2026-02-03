@echo off
REM ============================================================================
REM NKREPO Flask Backend - Run Server
REM ============================================================================

echo.
echo ========================================================================
echo Starting NKREPO Flask Backend Server
echo ========================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if config.ini exists
if not exist "config.ini" (
    echo [ERROR] config.ini not found!
    echo Please copy config.ini.example to config.ini and configure it
    pause
    exit /b 1
)

REM Set Flask environment variables
set FLASK_APP=app.py
set FLASK_ENV=development
set PYTHONUNBUFFERED=1

echo [*] Starting Flask server...
echo [*] Server will be available at: http://127.0.0.1:5005
echo [*] Press Ctrl+C to stop the server
echo.

REM Run Flask app
python app.py

REM If Flask exits with error
if errorlevel 1 (
    echo.
    echo [ERROR] Flask server stopped with errors
    echo Check the error messages above
    pause
    exit /b 1
)

pause
