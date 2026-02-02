#!/bin/bash
# ============================================================================
# NKREPO Vue Frontend - Ubuntu 22.04 Setup Script
# ============================================================================

set -e  # Exit on error

echo ""
echo "========================================================================"
echo "NKREPO Vue Frontend Setup for Ubuntu 22.04"
echo "========================================================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed"
    echo ""
    echo "Install Node.js 14+ with:"
    echo "  curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -"
    echo "  sudo apt install -y nodejs"
    exit 1
fi

echo "[*] Node.js found:"
node --version

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "[ERROR] npm is not installed"
    echo "npm should come with Node.js. Please reinstall Node.js"
    exit 1
fi

echo "[*] npm found:"
npm --version

# Install dependencies
echo ""
echo "[*] Installing dependencies from package.json..."
echo "This may take a few minutes..."
if npm install; then
    echo "[+] Dependencies installed successfully"
else
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

# Check environment configuration
echo ""
echo "[*] Checking environment configuration..."

if [ ! -f ".env.development" ] && [ ! -f ".env.development.local" ]; then
    echo "[WARNING] No development environment file found"
    echo "Create .env.development.local for local development settings"
fi

if [ ! -f ".env.production" ] && [ ! -f ".env.production.local" ]; then
    echo "[WARNING] No production environment file found"
    echo "Create .env.production.local for production build settings"
fi

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Configure environment files (.env.development.local)"
echo "2. Start development server: ./run_dev.sh"
echo "3. Build for production: ./build_prod.sh"
echo ""
