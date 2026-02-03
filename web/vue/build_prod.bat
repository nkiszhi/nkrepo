@echo off
REM ============================================================================
REM NKREPO Vue Frontend - Build for Production
REM ============================================================================

echo.
echo ========================================================================
echo Building NKREPO Vue Frontend for Production
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

REM Check if .env.production.local exists and warn user
if not exist ".env.production.local" (
    echo [WARNING] .env.production.local not found!
    echo Using default production configuration
    echo.
    echo [!] Make sure to configure VUE_APP_BASE_API in .env.production.local
    echo     to point to your production Flask backend URL
    echo.
    choice /C YN /M "Continue building anyway"
    if errorlevel 2 exit /b 0
)

echo [*] Node.js version:
node --version

echo.
echo [*] Building for production...
echo [*] This may take a few minutes...
echo.

REM Run production build
npm run build:prod

REM Check if build succeeded
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Build Complete!
echo ========================================================================
echo.
echo Production files are in: dist/
echo.
echo Deployment:
echo   1. Copy dist/ folder to your web server
echo   2. Configure web server (Nginx/Apache) to serve static files
echo   3. Set up proxy to Flask backend or configure CORS
echo.
echo Example Nginx configuration:
echo   location / {
echo     root /path/to/dist;
echo     try_files $uri $uri/ /index.html;
echo   }
echo   location /api {
echo     proxy_pass http://127.0.0.1:5005;
echo   }
echo.
pause
