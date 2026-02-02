#!/bin/bash
# ============================================================================
# NKREPO Flask Backend - Run Server (Ubuntu 22.04)
# ============================================================================

echo ""
echo "========================================================================"
echo "Starting NKREPO Flask Backend Server"
echo "========================================================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run ./setup_ubuntu.sh first"
    exit 1
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate

# Check if config.ini exists
if [ ! -f "config.ini" ]; then
    echo "[ERROR] config.ini not found!"
    echo "Please copy config.ini.example to config.ini and configure it"
    exit 1
fi

# Set Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export PYTHONUNBUFFERED=1

echo "[*] Starting Flask server..."
echo "[*] Server will be available at the configured IP:5005"
echo "[*] Press Ctrl+C to stop the server"
echo ""

# Run Flask app
python app.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Flask server stopped with errors"
    echo "Check the error messages above"
    exit 1
fi
