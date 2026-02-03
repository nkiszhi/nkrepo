@echo off
REM ============================================================================
REM NKREPO Vue Frontend - Run Development Server
REM ============================================================================

echo.
echo ========================================================================
echo Starting NKREPO Vue Frontend Development Server
echo ========================================================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

REM Check if node_modules exists
if not exist "node_modules\" (
    echo [ERROR] node_modules directory not found!
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

REM Check if .env.development.local exists
if not exist ".env.development.local" (
    echo [WARNING] .env.development.local not found!
    echo Using default development configuration
    echo.
)

echo [*] Node.js version:
node --version

echo.
echo [*] Starting development server...
echo [*] Frontend will be available at: http://localhost:9528
echo [*] Make sure Flask backend is running at the configured URL
echo [*] Press Ctrl+C to stop the server
echo.

REM Run development server
npm run dev

REM If server exits with error
if errorlevel 1 (
    echo.
    echo [ERROR] Development server stopped with errors
    echo Check the error messages above
    pause
    exit /b 1
)

pause
