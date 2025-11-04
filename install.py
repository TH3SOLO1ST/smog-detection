#!/usr/bin/env python3
"""
Islamabad Smog Detection System - One-Click Installer
Automated installation for non-technical users
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
import urllib.request
import zipfile
import tempfile

class SmogSystemInstaller:
    def __init__(self):
        self.system_info = {
            'platform': platform.system(),
            'python_version': sys.version_info,
            'architecture': platform.machine()
        }
        self.install_dir = Path.cwd()
        self.log_file = self.install_dir / "install_log.txt"

    def log_message(self, message):
        """Log installation progress"""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")

    def check_system_requirements(self):
        """Check if system meets minimum requirements"""
        self.log_message("🔍 Checking system requirements...")

        # Check disk space (minimum 10GB)
        disk_usage = shutil.disk_usage(self.install_dir)
        available_gb = disk_usage.free / (1024**3)

        if available_gb < 10:
            raise Exception(f"Insufficient disk space. Need 10GB, only {available_gb:.1f}GB available.")

        self.log_message(f"✅ Disk space: {available_gb:.1f}GB available")

        # Check Python version
        if self.system_info['python_version'] < (3, 9):
            self.log_message("⚠️  Python 3.9+ required. Attempting to install...")
            self.install_python()
        else:
            self.log_message(f"✅ Python version: {self.system_info['python_version'].major}.{self.system_info['python_version'].minor}")

        # Check internet connection
        try:
            urllib.request.urlopen('https://www.google.com', timeout=10)
            self.log_message("✅ Internet connection: OK")
        except Exception as e:
            raise Exception("No internet connection. Required for API access and package installation.")

    def install_python(self):
        """Install Python 3.9+ if not present"""
        system = self.system_info['platform']

        if system == "Windows":
            self.log_message("📥 Downloading Python for Windows...")
            # Windows - download python installer
            python_url = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
            installer_path = self.install_dir / "python_installer.exe"

            urllib.request.urlretrieve(python_url, installer_path)

            self.log_message("🚀 Installing Python (this may take a few minutes)...")
            subprocess.run([str(installer_path), "/quiet", "InstallAllUsers=1", "PrependPath=1"],
                         check=True, shell=True)

            installer_path.unlink()  # Remove installer

        elif system == "Darwin":  # macOS
            self.log_message("📥 Installing Python via Homebrew...")
            subprocess.run(["brew", "install", "python@3.11"], check=True)

        else:  # Linux
            self.log_message("📥 Installing Python via package manager...")
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "python3.11", "python3.11-pip", "python3.11-venv"],
                             check=True)
            except:
                try:
                    subprocess.run(["sudo", "yum", "install", "-y", "python3.11", "python3.11-pip"],
                                 check=True)
                except:
                    raise Exception("Could not install Python automatically. Please install Python 3.9+ manually.")

    def create_virtual_environment(self):
        """Create Python virtual environment"""
        self.log_message("🐍 Creating virtual environment...")

        venv_path = self.install_dir / "venv"

        if venv_path.exists():
            shutil.rmtree(venv_path)

        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

        self.log_message("✅ Virtual environment created")

        return venv_path

    def get_venv_python(self, venv_path):
        """Get path to virtual environment Python"""
        system = self.system_info['platform']

        if system == "Windows":
            return venv_path / "Scripts" / "python.exe"
        else:
            return venv_path / "bin" / "python"

    def get_venv_pip(self, venv_path):
        """Get path to virtual environment pip"""
        system = self.system_info['platform']

        if system == "Windows":
            return venv_path / "Scripts" / "pip.exe"
        else:
            return venv_path / "bin" / "pip"

    def install_system_dependencies(self):
        """Install system-level dependencies"""
        system = self.system_info['platform']

        self.log_message("📦 Installing system dependencies...")

        if system == "Windows":
            # Windows dependencies
            try:
                subprocess.run(["pip", "install", "--upgrade", "pip"], check=True)
                self.log_message("✅ pip updated")
            except:
                pass  # May fail if not in admin mode

        elif system == "Darwin":  # macOS
            try:
                subprocess.run(["brew", "install", "gdal", "proj"], check=True)
                self.log_message("✅ GDAL and PROJ installed via Homebrew")
            except:
                self.log_message("⚠️  Could not install GDAL via Homebrew. Will try pip installation.")

        else:  # Linux
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "gdal-bin", "libgdal-dev", "libproj-dev"],
                             check=True)
                self.log_message("✅ GDAL and PROJ installed via apt")
            except:
                try:
                    subprocess.run(["sudo", "yum", "install", "-y", "gdal", "proj", "proj-devel"],
                                 check=True)
                    self.log_message("✅ GDAL and PROJ installed via yum")
                except:
                    self.log_message("⚠️  Could not install system dependencies automatically")

    def install_python_packages(self, venv_path):
        """Install Python packages from requirements.txt"""
        self.log_message("📦 Installing Python packages (this may take 10-15 minutes)...")

        pip_path = self.get_venv_pip(venv_path)
        requirements_path = self.install_dir / "requirements.txt"

        # Upgrade pip first
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)

        # Install packages with specific options for different platforms
        install_cmd = [
            str(pip_path), "install", "-r", str(requirements_path),
            "--no-cache-dir"
        ]

        # Platform-specific options
        if self.system_info['platform'] == "Windows":
            install_cmd.extend(["--prefer-binary"])

        try:
            subprocess.run(install_cmd, check=True, capture_output=True, text=True)
            self.log_message("✅ All Python packages installed successfully")
        except subprocess.CalledProcessError as e:
            self.log_message(f"❌ Package installation failed: {e}")
            self.log_message("🔄 Trying alternative installation method...")

            # Try installing problematic packages individually
            critical_packages = [
                "flask", "pandas", "numpy", "matplotlib", "plotly",
                "requests", "pyyaml", "jinja2"
            ]

            for package in critical_packages:
                try:
                    subprocess.run([str(pip_path), "install", package], check=True)
                    self.log_message(f"✅ {package} installed")
                except:
                    self.log_message(f"⚠️  {package} failed to install")

    def setup_configuration(self):
        """Set up initial configuration"""
        self.log_message("⚙️  Setting up configuration...")

        config_template = self.install_dir / "config.yaml"

        if not config_template.exists():
            self.log_message("❌ config.yaml not found. Please ensure you have the complete system.")
            return False

        # Create .env file for sensitive data
        env_file = self.install_dir / ".env"

        if not env_file.exists():
            env_content = """# Islamabad Smog Detection System - Environment Variables
# Copy this file and fill in your API credentials

# Copernicus Data Space Ecosystem (for Sentinel-5P data)
COPERNICUS_CLIENT_ID=your_client_id_here
COPERNICUS_CLIENT_SECRET=your_client_secret_here

# NASA Earthdata (for MODIS data)
NASA_EARTHDATA_USERNAME=your_username_here
NASA_EARTHDATA_PASSWORD=your_password_here

# Google Earth Engine
GEE_PROJECT_ID=your_project_id_here
GEE_SERVICE_ACCOUNT_KEY=path/to/service-account-key.json

# Email Configuration (for automated reports)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here

# System Settings
DEBUG=False
SECRET_KEY=your_secret_key_here
DATABASE_URL=sqlite:///smog_system.db
"""

            with open(env_file, "w") as f:
                f.write(env_content)

            self.log_message("✅ Environment template created (.env)")

        # Create data directories
        data_dirs = [
            "data/raw/sentinel5p",
            "data/raw/modis",
            "data/raw/gee",
            "data/processed",
            "data/exports",
            "logs"
        ]

        for dir_path in data_dirs:
            full_path = self.install_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        self.log_message("✅ Data directories created")

        return True

    def create_desktop_shortcut(self):
        """Create desktop shortcut for easy access"""
        self.log_message("🖥️  Creating desktop shortcut...")

        system = self.system_info['platform']
        shortcut_name = "Islamabad Smog Monitor"

        if system == "Windows":
            # Create Windows desktop shortcut
            desktop_path = Path.home() / "Desktop"
            shortcut_path = desktop_path / f"{shortcut_name}.bat"

            venv_python = self.get_venv_python(self.install_dir / "venv")
            app_script = self.install_dir / "src" / "web_interface" / "app.py"

            shortcut_content = f"""@echo off
cd /d "{self.install_dir}"
"{venv_python}" "{app_script}"
pause
"""

            with open(shortcut_path, "w") as f:
                f.write(shortcut_content)

            self.log_message(f"✅ Desktop shortcut created: {shortcut_path}")

        elif system == "Darwin":  # macOS
            # Create macOS app
            app_path = Path.home() / "Desktop" / f"{shortcut_name}.app"

            # Create app structure
            contents_dir = app_path / "Contents"
            contents_dir.mkdir(parents=True, exist_ok=True)

            # Create Info.plist
            info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>run.sh</string>
    <key>CFBundleIdentifier</key>
    <string>com.smogmonitor.app</string>
    <key>CFBundleName</key>
    <string>{shortcut_name}</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
</dict>
</plist>
"""

            with open(contents_dir / "Info.plist", "w") as f:
                f.write(info_plist)

            # Create executable script
            run_script = contents_dir / "MacOS" / "run.sh"
            run_script.parent.mkdir(parents=True, exist_ok=True)

            venv_python = self.get_venv_python(self.install_dir / "venv")
            app_script = self.install_dir / "src" / "web_interface" / "app.py"

            script_content = f"""#!/bin/bash
cd "{self.install_dir}"
"{venv_python}" "{app_script}"
"""

            with open(run_script, "w") as f:
                f.write(script_content)

            run_script.chmod(0o755)
            self.log_message(f"✅ macOS app created: {app_path}")

        else:  # Linux
            # Create Linux desktop entry
            desktop_dir = Path.home() / ".local" / "share" / "applications"
            desktop_dir.mkdir(parents=True, exist_ok=True)

            desktop_file = desktop_dir / f"{shortcut_name.lower().replace(' ', '-')}.desktop"

            venv_python = self.get_venv_python(self.install_dir / "venv")
            app_script = self.install_dir / "src" / "web_interface" / "app.py"

            desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name={shortcut_name}
Comment=Islamabad Smog Detection and Monitoring System
Exec={venv_python} {app_script}
Icon={self.install_dir}/docs/icon.png
Terminal=false
Categories=Science;Education;
"""

            with open(desktop_file, "w") as f:
                f.write(desktop_content)

            os.chmod(desktop_file, 0o644)
            self.log_message(f"✅ Desktop entry created: {desktop_file}")

    def test_installation(self):
        """Test if installation was successful"""
        self.log_message("🧪 Testing installation...")

        venv_python = self.get_venv_python(self.install_dir / "venv")

        try:
            # Test basic imports
            test_script = self.install_dir / "test_imports.py"

            test_code = """
import sys
sys.path.insert(0, 'src')

try:
    import flask
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly
    import yaml
    print("SUCCESS: All critical imports work")
except ImportError as e:
    print(f"ERROR: Import failed - {e}")
    sys.exit(1)
"""

            with open(test_script, "w") as f:
                f.write(test_code)

            result = subprocess.run([str(venv_python), str(test_script)],
                                  capture_output=True, text=True)

            test_script.unlink()  # Clean up

            if result.returncode == 0:
                self.log_message("✅ Installation test passed")
                return True
            else:
                self.log_message(f"❌ Installation test failed: {result.stderr}")
                return False

        except Exception as e:
            self.log_message(f"❌ Test failed with exception: {e}")
            return False

    def run_setup_wizard(self):
        """Run interactive setup wizard for configuration"""
        self.log_message("🧙 Running setup wizard...")

        print("\n" + "="*60)
        print("🌫️  ISLAMABAD SMOG DETECTION SYSTEM - SETUP WIZARD")
        print("="*60)
        print("\nThis wizard will help you configure your smog monitoring system.")
        print("You'll need API credentials from the following services:")
        print("• Copernicus Data Space Ecosystem (free)")
        print("• NASA Earthdata (free)")
        print("• Google Earth Engine (free tier available)")
        print("\nDon't worry if you don't have these ready - you can add them later!")
        print("-"*60)

        # Get basic configuration
        config = {
            "organization": input("\nOrganization name [e.g., Islamabad Environmental Agency]: ").strip() or "Islamabad Environmental Agency",
            "contact_email": input("Contact email: ").strip(),
            "auto_update": input("Enable automatic daily updates? [Y/n]: ").strip().lower() != 'n',
            "email_reports": input("Enable email reports? [Y/n]: ").strip().lower() != 'n',
        }

        # Save configuration
        config_file = self.install_dir / "user_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        self.log_message("✅ Configuration saved")

        print("\n✅ Setup complete! You can now:")
        print("• Run the system using the desktop shortcut")
        print("• Access the web dashboard at http://localhost:5000")
        print("• Edit configuration in config.yaml and .env files")
        print("• Read the user guide in docs/simple_user_guide.md")

    def install(self):
        """Main installation process"""
        try:
            self.log_message("🚀 Starting Islamabad Smog Detection System installation...")

            # Step 1: Check requirements
            self.check_system_requirements()

            # Step 2: Create virtual environment
            venv_path = self.create_virtual_environment()

            # Step 3: Install system dependencies
            self.install_system_dependencies()

            # Step 4: Install Python packages
            self.install_python_packages(venv_path)

            # Step 5: Set up configuration
            if not self.setup_configuration():
                return False

            # Step 6: Create desktop shortcut
            self.create_desktop_shortcut()

            # Step 7: Test installation
            if not self.test_installation():
                self.log_message("⚠️  Installation completed with warnings. Check the log file.")

            # Step 8: Run setup wizard
            self.run_setup_wizard()

            self.log_message("\n🎉 INSTALLATION COMPLETE!")
            self.log_message("="*50)
            self.log_message("Next steps:")
            self.log_message("1. Get API credentials (see docs/api_setup.md)")
            self.log_message("2. Edit .env file with your credentials")
            self.log_message("3. Double-click the desktop shortcut to start")
            self.log_message("4. Access dashboard at http://localhost:5000")
            self.log_message(f"5. Installation log saved to: {self.log_file}")

            return True

        except Exception as e:
            self.log_message(f"\n❌ INSTALLATION FAILED: {e}")
            self.log_message(f"Check the log file for details: {self.log_file}")
            return False

def main():
    """Main entry point"""
    print("🌫️  Islamabad Smog Detection System - Installer")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found!")
        print("Please run this installer from the smog-detection directory.")
        sys.exit(1)

    # Run installation
    installer = SmogSystemInstaller()
    success = installer.install()

    if success:
        input("\n✨ Installation successful! Press Enter to exit...")
    else:
        input("\n❌ Installation failed. Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()