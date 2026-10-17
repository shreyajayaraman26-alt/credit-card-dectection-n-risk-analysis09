@echo off
title Fraud Risk Analyser Dashboard
echo ===================================================================
echo   Starting Credit Card Fraud & Risk Analyser Dashboard
echo   Web Dashboard: http://localhost
echo   API Docs:      http://localhost/docs
echo ===================================================================
echo.
cd /d "%~dp0"
timeout /t 2 /nobreak >nul
start http://localhost
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] The application server encountered an issue or closed.
    pause
)
