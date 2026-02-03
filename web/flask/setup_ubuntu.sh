#!/bin/bash
# ============================================================================
# NKREPO Flask Backend - Ubuntu 22.04 Setup Script
# ============================================================================

set -e  # Exit on error

echo ""
echo "========================================================================"
echo "NKREPO Flask Backend Setup for Ubuntu 22.04"
echo "========================================================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Install it with: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "[*] Python found:"
python3 --version

# Check if pip is installed
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "[ERROR] pip is not installed"
    echo "Install it with: sudo apt install python3-pip"
    exit 1
fi

echo "[*] pip found:"
python3 -m pip --version

# Check if virtual environment module is available
if ! python3 -m venv --help &> /dev/null; then
    echo "[ERROR] python3-venv is not installed"
    echo "Install it with: sudo apt install python3-venv"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    echo "[+] Virtual environment created"
else
    echo "[*] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[*] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[*] Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo ""
echo "[*] Installing dependencies from requirements.txt..."
if pip install -r requirements.txt; then
    echo "[+] Dependencies installed successfully"
else
    echo "[WARNING] Some dependencies may have failed to install"
    echo "You can continue, but some features may not work"
fi

# Check for config.ini
if [ ! -f "config.ini" ]; then
    echo ""
    echo "[WARNING] config.ini not found!"
    echo ""
    if [ -f "config.ini.example" ]; then
        echo "[*] Creating config.ini from template..."
        cp config.ini.example config.ini
        echo ""
        echo "[!] IMPORTANT: Please edit config.ini and configure:"
        echo "    - MySQL connection settings"
        echo "    - VirusTotal API key"
        echo "    - File paths for your system"
        echo ""
        echo "Opening config.ini in default editor..."
        ${EDITOR:-nano} config.ini
    else
        echo "[ERROR] config.ini.example template not found"
        echo "Please create config.ini manually"
        exit 1
    fi
fi

# Create necessary directories
echo ""
echo "[*] Creating necessary directories..."

if [ ! -d "../vue/uploads" ]; then
    mkdir -p "../vue/uploads"
    echo "[+] Created uploads directory"
fi

# Check for 7zip installation
echo ""
echo "[*] Checking for p7zip..."
if ! command -v 7z &> /dev/null && ! command -v 7za &> /dev/null; then
    echo "[WARNING] 7zip (p7zip-full) is not installed"
    echo "Sample download functionality requires 7zip"
    echo "Install it with: sudo apt install p7zip-full"
else
    echo "[+] 7zip found"
fi

# Check for MySQL
echo ""
echo "[*] Checking for MySQL..."
if ! command -v mysql &> /dev/null; then
    echo "[WARNING] MySQL client is not installed"
    echo "Install MySQL server with: sudo apt install mysql-server"
else
    echo "[+] MySQL client found"
    mysql --version
fi

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Edit config.ini with your settings"
echo "2. Initialize the database: cd ../../db && python3 init_db.py"
echo "3. Run the Flask server: ./run_ubuntu.sh"
echo ""
echo "To activate the virtual environment manually:"
echo "  source venv/bin/activate"
echo ""

deactivate 2>/dev/null || true
