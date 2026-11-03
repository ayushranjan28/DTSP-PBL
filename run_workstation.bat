@echo off
setlocal
title SonicPure DSP Audio Workstation

echo ======================================================================
echo           SonicPure DSP Audio Workstation Launcher
echo      FastAPI REST Backend + HTML5/CSS3/JS Web Workstation
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/2] Verifying Python runtime...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in your PATH!
    echo Please ensure Python 3.11+ is installed and added to your system PATH.
    echo.
    pause
    exit /b 1
)

echo [2/2] Launching unified backend and frontend...
echo URL: http://localhost:5000
echo.
echo Opening default web browser in 2 seconds...
echo Press CTRL+C to stop the workstation.
echo ======================================================================
echo.

:: Automatically trigger browser open after server begins listening
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Launch the FastAPI server (hosts REST endpoints and static web workstation)
python server.py

if errorlevel 1 (
    echo.
    echo [ERROR] The server stopped unexpectedly.
    pause
)
