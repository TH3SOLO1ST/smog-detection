@echo off
title Islamabad Smog Detection System
color 0A

echo ============================================================
echo 🌫️  Islamabad Smog Detection System
echo ============================================================
echo.
echo Starting web dashboard...
echo Please wait a moment...
echo ------------------------------------------------------------
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Error: Virtual environment not found
    echo Please run install.py first to set up the system
    echo.
    pause
    exit /b 1
)

REM Check if Flask app exists
if not exist "src\web_interface\app.py" (
    echo ❌ Error: Application files not found
    echo Please ensure you have the complete system
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment and start server
echo 🚀 Starting web server...
echo 📍 Dashboard will be available at: http://localhost:5000
echo 🔄 Press Ctrl+C to stop the system
echo ------------------------------------------------------------

call venv\Scripts\activate.bat

REM Set environment variables
set FLASK_APP=src\web_interface\app.py
set FLASK_ENV=production
set PYTHONPATH=%CD%

REM Start Flask server
python -m flask run --host=0.0.0.0 --port=5000

REM If we get here, the server has stopped
echo.
echo ✅ System stopped successfully
pause