#!/bin/bash
# ============================================================================
# NKREPO Vue Frontend - Run Development Server (Ubuntu 22.04)
# ============================================================================

echo ""
echo "========================================================================"
echo "Starting NKREPO Vue Development Server"
echo "========================================================================"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "[ERROR] node_modules not found!"
    echo "Please run ./setup_ubuntu.sh first"
    exit 1
fi

echo "[*] Starting Vue development server..."
echo "[*] Server will be available at: http://localhost:9528"
echo "[*] Press Ctrl+C to stop the server"
echo ""

# Run Vue development server
npm run dev

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Development server stopped with errors"
    exit 1
fi
