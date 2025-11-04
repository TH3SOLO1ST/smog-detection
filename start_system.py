#!/usr/bin/env python3
"""
Islamabad Smog Detection System - Startup Script
Easy system launcher for non-technical users
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🌫️  Islamabad Smog Detection System")
    print("=" * 60)
    print("Starting web dashboard...")
    print("Please wait a moment...")
    print("-" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_virtual_environment():
    """Check if virtual environment exists"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Error: Virtual environment not found")
        print("Please run install.py first to set up the system")
        return False
    return True

def get_python_executable():
    """Get Python executable from virtual environment"""
    system = sys.platform
    venv_python = Path("venv")

    if system == "Windows":
        venv_python /= "Scripts" / "python.exe"
    else:
        venv_python /= "bin" / "python"

    if venv_python.exists():
        return str(venv_python)
    else:
        print("❌ Error: Python executable not found in virtual environment")
        return None

def start_web_server():
    """Start the Flask web server"""
    python_executable = get_python_executable()
    if not python_executable:
        return False

    try:
        # Change to script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)

        # Set environment variables
        env = os.environ.copy()
        env['FLASK_APP'] = 'src/web_interface/app.py'
        env['FLASK_ENV'] = 'production'
        env['PYTHONPATH'] = str(script_dir)

        print("🚀 Starting web server...")
        print("📍 Dashboard will be available at: http://localhost:5000")
        print("🔄 Press Ctrl+C to stop the system")
        print("-" * 60)

        # Start Flask server
        process = subprocess.Popen([
            python_executable,
            "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"
        ], env=env)

        return process

    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        return False

def open_browser():
    """Open web browser after server starts"""
    print("⏳ Waiting for server to start...")
    time.sleep(3)  # Give server time to start

    try:
        webbrowser.open("http://localhost:5000")
        print("🌐 Opening dashboard in your default browser...")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("Please manually open: http://localhost:5000")

def main():
    """Main startup function"""
    print_banner()

    # Check requirements
    if not check_python_version():
        input("Press Enter to exit...")
        return

    if not check_virtual_environment():
        input("Press Enter to exit...")
        return

    # Start web server
    server_process = start_web_server()
    if not server_process:
        input("Press Enter to exit...")
        return

    # Open browser after a short delay
    open_browser()

    try:
        # Wait for server process
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping web server...")
        server_process.terminate()
        server_process.wait()
        print("✅ System stopped successfully")

if __name__ == "__main__":
    main()