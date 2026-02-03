@echo off
REM ============================================================================
REM NKREPO Database Setup for Windows
REM ============================================================================

echo.
echo ========================================================================
echo NKREPO Database Initialization
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
echo.

REM Check if MySQL is installed
mysql --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] MySQL client not found in PATH
    echo Please ensure MySQL Server is installed
    echo Download from: https://dev.mysql.com/downloads/mysql/
    echo.
)

REM Check if init_db.py exists
if not exist "init_db.py" (
    echo [ERROR] init_db.py not found!
    echo Please run this script from the db/ directory
    pause
    exit /b 1
)

REM Install mysql-connector-python if not installed
echo [*] Checking mysql-connector-python...
python -c "import mysql.connector" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing mysql-connector-python...
    pip install mysql-connector-python
    if errorlevel 1 (
        echo [ERROR] Failed to install mysql-connector-python
        pause
        exit /b 1
    )
)

echo.
echo ========================================================================
echo MySQL Configuration
echo ========================================================================
echo.
echo Please provide MySQL connection details:
echo.

REM Prompt for MySQL credentials
set /p MYSQL_HOST="MySQL Host (default: localhost): "
if "%MYSQL_HOST%"=="" set MYSQL_HOST=localhost

set /p MYSQL_USER="MySQL Username (default: root): "
if "%MYSQL_USER%"=="" set MYSQL_USER=root

set /p MYSQL_PASS="MySQL Password: "
if "%MYSQL_PASS%"=="" (
    echo [ERROR] Password cannot be empty
    pause
    exit /b 1
)

echo.
echo [*] Connection details:
echo     Host: %MYSQL_HOST%
echo     User: %MYSQL_USER%
echo     Password: ********
echo.

choice /C YN /M "Is this correct"
if errorlevel 2 goto :end

echo.
echo ========================================================================
echo Creating Database and Tables
echo ========================================================================
echo.
echo [*] This will create:
echo     1. Database: nkrepo
echo     2. 256 sample tables (sample_00 to sample_ff)
echo     3. Domain table (domain_YYYYMMDD)
echo     4. User table
echo.
echo [*] Running database initialization...
echo.

REM Run init_db.py
python init_db.py -u %MYSQL_USER% -p %MYSQL_PASS% -h %MYSQL_HOST%

if errorlevel 1 (
    echo.
    echo [ERROR] Database initialization failed!
    echo.
    echo Troubleshooting:
    echo   1. Verify MySQL Server is running
    echo   2. Check MySQL credentials are correct
    echo   3. Ensure user has CREATE DATABASE privilege
    echo   4. Check MySQL error messages above
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Additional Database Setup (Optional)
echo ========================================================================
echo.
echo The main NKREPO system also uses additional databases:
echo   - nkrepo_category (malware categories)
echo   - nkrepo_family (malware families)
echo   - nkrepo_platform (target platforms)
echo.
choice /C YN /M "Create additional databases now"
if errorlevel 2 goto :skip_additional

echo.
echo [*] Creating additional databases...

REM Create additional databases using MySQL command
mysql -h %MYSQL_HOST% -u %MYSQL_USER% -p%MYSQL_PASS% -e "CREATE DATABASE IF NOT EXISTS nkrepo_category DEFAULT CHARSET utf8mb4;" 2>nul
if not errorlevel 1 (
    echo [+] Database nkrepo_category created
) else (
    echo [!] Failed to create nkrepo_category
)

mysql -h %MYSQL_HOST% -u %MYSQL_USER% -p%MYSQL_PASS% -e "CREATE DATABASE IF NOT EXISTS nkrepo_family DEFAULT CHARSET utf8mb4;" 2>nul
if not errorlevel 1 (
    echo [+] Database nkrepo_family created
) else (
    echo [!] Failed to create nkrepo_family
)

mysql -h %MYSQL_HOST% -u %MYSQL_USER% -p%MYSQL_PASS% -e "CREATE DATABASE IF NOT EXISTS nkrepo_platform DEFAULT CHARSET utf8mb4;" 2>nul
if not errorlevel 1 (
    echo [+] Database nkrepo_platform created
) else (
    echo [!] Failed to create nkrepo_platform
)

:skip_additional

echo.
echo ========================================================================
echo Database Setup Complete!
echo ========================================================================
echo.
echo Next steps:
echo   1. Update web/flask/config.ini with MySQL credentials:
echo      - host = %MYSQL_HOST%
echo      - user = %MYSQL_USER%
echo      - passwd = ********
echo   2. Import sample data (if available)
echo   3. Start Flask backend server
echo.
echo Database summary:
echo   Main database:   nkrepo
echo   Sample tables:   256 sharded tables (sample_00 to sample_ff)
echo   Domain table:    domain_YYYYMMDD
echo   User table:      user
echo.

:end
pause
