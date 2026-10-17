@echo off
title SentinelPay AI - Fraud & Risk Decision Console
echo ===================================================================
echo   Starting SentinelPay AI - Fraud & Risk Decision Platform
echo   Local URL: http://127.0.0.1:8000
echo   API Docs:  http://127.0.0.1:8000/docs
echo ===================================================================
echo.
cd /d "%~dp0"
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8000
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] The application server encountered an issue or closed.
    pause
)
