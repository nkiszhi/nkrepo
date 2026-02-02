#!/bin/bash
# ============================================================================
# NKREPO Vue Frontend - Production Build (Ubuntu 22.04)
# ============================================================================

echo ""
echo "========================================================================"
echo "Building NKREPO Vue Frontend for Production"
echo "========================================================================"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "[ERROR] node_modules not found!"
    echo "Please run ./setup_ubuntu.sh first"
    exit 1
fi

# Clean previous build
if [ -d "dist" ]; then
    echo "[*] Cleaning previous build..."
    rm -rf dist
fi

echo "[*] Building for production..."
echo "This may take a few minutes..."
echo ""

# Run production build
npm run build:prod

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Build Complete!"
    echo "========================================================================"
    echo ""
    echo "Production files are in the 'dist' directory"
    echo "Deploy these files to your web server"
    echo ""
else
    echo ""
    echo "[ERROR] Build failed!"
    echo "Check the error messages above"
    exit 1
fi
