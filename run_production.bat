@echo off
cd /d "%~dp0"
title Aerial Change Detection Studio - DEPI Graduation Project
cls
echo =======================================================================
echo    AERIAL IMAGE CHANGE DETECTION STUDIO (LEVIR-CD+)
echo    DEPI AI Track Graduation Project - Production Web Studio
echo =======================================================================
echo.
echo Project Team:
echo   - Ahmed Refaat Abd-Elmotelb  (Lead ML Engineer ^& Pipeline Architect)
echo   - Ahmed Reda Mohamed         (Deep Learning Specialist ^& Loss Design)
echo   - Raeed Youssef Mahfouz      (Data Systems ^& Augmentation Specialist)
echo   - Ahmed Mohamed Qotb         (Evaluation Metrics ^& QA Lead)
echo.
echo GitHub Repository: https://github.com/nhahub/NHA-5-130
echo =======================================================================
echo.
echo [*] Initializing Deep Learning engine and loading weights...
echo [*] Browser will open automatically once the engine is ready.
echo.

set PYTHON_CMD=python
if exist "D:\envs\levir-cd\python.exe" set PYTHON_CMD="D:\envs\levir-cd\python.exe"

start /b "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

%PYTHON_CMD% app.py

pause
