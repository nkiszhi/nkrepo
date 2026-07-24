@echo off
REM ============================================================
REM  AV Detector Service — 一键启动脚本
REM  用法: run_detector_service.bat
REM  可选环境变量: AV_SERVICE_PORT=5006 (默认)
REM ============================================================

setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================
echo   AV Detector Service
echo   %DATE% %TIME%
echo ============================================================

REM 查找 Python
set PYTHON=
for %%p in (python.exe python3.exe) do (
    where %%p >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set PYTHON=%%p
        goto :found_python
    )
)

REM 尝试固定路径
if exist "C:\Python314\python.exe" set PYTHON=C:\Python314\python.exe
if exist "C:\Users\34701\AppData\Local\Programs\Python\Python314\python.exe" set PYTHON=C:\Users\34701\AppData\Local\Programs\Python\Python314\python.exe

:found_python
if "%PYTHON%"=="" (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo Python: %PYTHON%

REM 检查 vm_config.json
if not exist "vm_config.json" (
    if exist "..\共享\vm_config.json" (
        copy "..\共享\vm_config.json" "vm_config.json" >nul
        echo Copied vm_config.json from ..\共享\
    ) else (
        echo [WARN] vm_config.json not found — service may fail to init
    )
)

REM 创建 uploads 目录
if not exist "uploads" mkdir uploads

echo.
echo Starting AV Detector Service on port 5006...
echo Swagger: http://localhost:5006/docs
echo.

"%PYTHON%" av_detector_service.py

pause
