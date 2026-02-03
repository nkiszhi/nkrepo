@echo off
REM ============================================================================
REM NKREPO Vue Frontend - Windows Setup Script
REM ============================================================================

echo.
echo ========================================================================
echo NKREPO Vue Frontend Setup for Windows
echo ========================================================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 8.9+ from https://nodejs.org/
    echo Recommended: Node.js LTS version
    pause
    exit /b 1
)

echo [*] Node.js found:
node --version

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm is not installed
    echo Please reinstall Node.js with npm included
    pause
    exit /b 1
)

echo [*] npm found:
npm --version

REM Check Node.js version requirement (>=8.9)
echo.
echo [*] Checking Node.js version requirement (>=8.9)...
for /f "tokens=1 delims=v" %%i in ('node --version') do set NODE_VERSION=%%i
echo Node version: %NODE_VERSION%

REM Check if package.json exists
if not exist "package.json" (
    echo [ERROR] package.json not found!
    echo Please run this script from the web/vue directory
    pause
    exit /b 1
)

REM Check if node_modules exists
if exist "node_modules\" (
    echo.
    echo [WARNING] node_modules directory already exists
    choice /C YN /M "Do you want to delete and reinstall dependencies"
    if errorlevel 2 goto :skip_delete
    if errorlevel 1 (
        echo [*] Removing existing node_modules...
        rd /s /q node_modules
        if exist "package-lock.json" (
            del package-lock.json
        )
    )
)

:skip_delete

REM Set registry to npm (optional: use Taobao mirror for faster download in China)
echo.
choice /C YN /M "Use Taobao npm mirror for faster download (recommended in China)"
if errorlevel 2 goto :skip_mirror
if errorlevel 1 (
    echo [*] Setting npm registry to Taobao mirror...
    npm config set registry https://registry.npmmirror.com
    echo [+] Registry set to Taobao mirror
)

:skip_mirror

REM Install dependencies
echo.
echo [*] Installing dependencies from package.json...
echo [*] This may take several minutes...
npm install
if errorlevel 1 (
    echo.
    echo [ERROR] npm install failed!
    echo.
    echo Troubleshooting:
    echo   1. Try running: npm cache clean --force
    echo   2. Delete node_modules and package-lock.json
    echo   3. Run setup_windows.bat again
    echo   4. Check your internet connection
    pause
    exit /b 1
)

echo.
echo [+] Dependencies installed successfully!

REM Create .env.development.local if not exists
if not exist ".env.development.local" (
    echo.
    echo [*] Creating .env.development.local...
    (
        echo # Development Environment Configuration
        echo # This file overrides .env.development
        echo ENV = 'development'
        echo.
        echo # Flask backend API URL
        echo # Change this to match your Flask server configuration
        echo VUE_APP_BASE_API = 'http://127.0.0.1:5005'
        echo.
        echo # Alternative: Use proxy (see vue.config.js^)
        echo # VUE_APP_BASE_API = '/dev-api'
    ) > .env.development.local
    echo [+] Created .env.development.local
    echo.
    echo [!] IMPORTANT: Edit .env.development.local to configure Flask backend URL
)

REM Create .env.production.local if not exists
if not exist ".env.production.local" (
    echo.
    echo [*] Creating .env.production.local...
    (
        echo # Production Environment Configuration
        echo ENV = 'production'
        echo.
        echo # Flask backend API URL for production
        echo # Change this to your production server URL
        echo VUE_APP_BASE_API = 'http://your-production-server:5005'
    ) > .env.production.local
    echo [+] Created .env.production.local
    echo.
    echo [!] IMPORTANT: Edit .env.production.local before building for production
)

REM Check if uploads directory exists
if not exist "uploads\" (
    echo.
    echo [*] Creating uploads directory...
    mkdir uploads
    echo [+] Created uploads directory
)

echo.
echo ========================================================================
echo Setup Complete!
echo ========================================================================
echo.
echo Next steps:
echo   1. Start Flask backend server (see web/flask/run_flask.bat)
echo   2. Edit .env.development.local to configure backend URL
echo   3. Run development server: run_dev.bat
echo   4. Access application at: http://localhost:9528
echo.
echo Commands:
echo   run_dev.bat          - Start development server
echo   build_prod.bat       - Build for production
echo   npm run lint         - Check code style
echo   npm run test:unit    - Run unit tests
echo.
pause
