# NKREPO Dependencies & Installation Guide

Complete guide for installing all dependencies for the malware analysis tools.

## 📦 Quick Installation

### Option 1: Install Everything (Recommended)
```bash
# Main Flask application dependencies
cd web/flask
pip install -r requirements.txt

# Malware analysis utilities
cd ../../utils
pip install -r requirements.txt
```

### Option 2: Minimal Installation (Basic Functionality)
```bash
# Core tools only (no fuzzy hashing)
pip install lief pefile yara-python
```

### Option 3: Install by Tool
Choose what you need:

```bash
# For packer detection only
pip install lief

# For YARA scanning
pip install lief yara-python

# For hash calculation (all features)
pip install lief pefile ssdeep py-tlsh
```

## 🔧 Detailed Installation

### 1. Core Dependencies (Required)

#### LIEF (PE Parser)
**Purpose:** Modern PE parser for packer detection
**Used by:** `packer_detector.py`, `windows_packer_scanner.py`, `yara_packer_scanner.py`

```bash
pip install lief
```

**Verify:**
```bash
python -c "import lief; print(f'LIEF {lief.__version__} installed')"
```

#### pefile (PE Analysis)
**Purpose:** Classic PE parser for detailed analysis
**Used by:** `pe_hash_calculator.py` (imphash, authentihash)

```bash
pip install pefile
```

**Verify:**
```bash
python -c "import pefile; print('pefile installed')"
```

#### yara-python (YARA Rules)
**Purpose:** YARA rule scanning engine
**Used by:** `yara_packer_scanner.py`, `collect_yara_rules.py`

```bash
pip install yara-python
```

**Verify:**
```bash
python -c "import yara; print('YARA installed')"
```

---

### 2. Optional Dependencies (Enhanced Features)

#### SSDEEP (Fuzzy Hashing)
**Purpose:** Similarity detection for modified files
**Used by:** `pe_hash_calculator.py`

**Windows:**
```bash
# May require Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

pip install ssdeep
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libfuzzy-dev
pip install ssdeep
```

**macOS:**
```bash
brew install ssdeep
pip install ssdeep
```

**Verify:**
```bash
python -c "import ssdeep; print('SSDEEP installed')"
```

**Note:** If installation fails, skip it. The tools will work without SSDEEP.

#### TLSH (Trend Micro Locality Sensitive Hash)
**Purpose:** Advanced similarity detection
**Used by:** `pe_hash_calculator.py`

```bash
pip install py-tlsh
```

**Verify:**
```bash
python -c "import tlsh; print('TLSH installed')"
```

---

### 3. Web Application Dependencies

For the Flask web interface:

```bash
cd web/flask
pip install -r requirements.txt
```

**Key dependencies:**
- Flask 3.1.0 - Web framework
- PyTorch 2.5.1 - Deep learning models
- scikit-learn 1.5.2 - ML classifiers
- pandas 2.2.3 - Data analysis
- PyMySQL 1.1.1 - Database connector

---

## 🎯 Installation by Use Case

### Use Case 1: Malware Researcher (Full Stack)
```bash
# Install everything
pip install lief pefile yara-python ssdeep py-tlsh tqdm

# Verify
python utils/check_lief_version.py
python -c "import yara; print('YARA OK')"
python -c "import ssdeep; print('SSDEEP OK')"
```

### Use Case 2: Security Analyst (Packer Detection)
```bash
# Install core tools
pip install lief yara-python

# Verify
python utils/packer_detector.py "C:/Windows/System32/notepad.exe"
```

### Use Case 3: Hash Analysis Only
```bash
# Install hash calculation tools
pip install pefile ssdeep py-tlsh

# Verify
python utils/pe_hash_calculator.py "C:/Windows/System32/calc.exe"
```

### Use Case 4: YARA Scanning Only
```bash
# Install YARA
pip install yara-python

# Verify
python -c "import yara; yara.compile('yara_rules/packers_complete.yar'); print('OK')"
```

---

## 🐛 Troubleshooting

### Issue 1: LIEF Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement lief
```

**Solution:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Try again
pip install lief

# Or install specific version
pip install lief==0.14.0
```

---

### Issue 2: SSDEEP Compilation Errors (Windows)

**Error:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**

**Option A:** Install Visual C++ Build Tools
1. Download from: https://visualstudio.microsoft.com/downloads/
2. Install "Desktop development with C++"
3. Retry: `pip install ssdeep`

**Option B:** Use pre-compiled binary
```bash
# Try installing pre-built wheel
pip install --only-binary :all: ssdeep
```

**Option C:** Skip SSDEEP
```bash
# Install without SSDEEP
pip install lief pefile yara-python py-tlsh

# Tools will work, fuzzy hashing disabled
```

---

### Issue 3: yara-python Installation Fails

**Error:**
```
ERROR: Failed building wheel for yara-python
```

**Solution:**

**Windows:**
```bash
# Install from pre-built wheel
pip install yara-python --only-binary :all:
```

**Linux:**
```bash
# Install YARA development files first
sudo apt-get install automake libtool make gcc pkg-config

# Then install yara-python
pip install yara-python
```

**macOS:**
```bash
brew install yara
pip install yara-python
```

---

### Issue 4: Import Errors After Installation

**Error:**
```python
ModuleNotFoundError: No module named 'lief'
```

**Solution:**
```bash
# Check if installed in correct Python environment
pip list | grep lief

# If not found, install in current environment
python -m pip install lief

# Verify Python version
python --version  # Should be 3.8+
```

---

### Issue 5: Version Conflicts

**Error:**
```
ERROR: Package xyz has requirement abc>=1.0, but you have abc 0.9
```

**Solution:**
```bash
# Create virtual environment (recommended)
python -m venv nkrepo_env

# Activate
# Windows:
nkrepo_env\Scripts\activate
# Linux/Mac:
source nkrepo_env/bin/activate

# Install clean
pip install -r requirements.txt
```

---

## 📊 Dependency Matrix

| Tool | LIEF | pefile | YARA | SSDEEP | TLSH |
|------|------|--------|------|--------|------|
| packer_detector.py | ✅ Required | ❌ | ❌ | ❌ | ❌ |
| windows_packer_scanner.py | ✅ Required | ❌ | ❌ | ❌ | ❌ |
| yara_packer_scanner.py | ✅ Required | ❌ | ✅ Required | ❌ | ❌ |
| pe_hash_calculator.py | ❌ | ⚠️ Recommended | ❌ | ⚠️ Optional | ⚠️ Optional |
| collect_yara_rules.py | ❌ | ❌ | ❌ | ❌ | ❌ |

Legend:
- ✅ Required - Tool won't work without it
- ⚠️ Recommended - Tool works but features limited
- ❌ Not needed

---

## 🔄 Version Compatibility

### Tested Versions

| Package | Minimum | Tested | Latest |
|---------|---------|--------|--------|
| Python | 3.8 | 3.11 | 3.12 |
| lief | 0.14.0 | 0.17.3 | 0.17.3 |
| pefile | 2023.2.7 | 2024.8.26 | 2024.8.26 |
| yara-python | 4.3.0 | 4.5.1 | 4.5.1 |
| ssdeep | 3.4 | 3.4 | 3.4 |
| py-tlsh | 4.7.2 | 4.7.2 | 4.11.2 |

### Python Version Support
- ✅ Python 3.8 - Fully supported
- ✅ Python 3.9 - Fully supported
- ✅ Python 3.10 - Fully supported
- ✅ Python 3.11 - Fully supported
- ✅ Python 3.12 - Mostly supported (LIEF compatibility)
- ❌ Python 3.7 - Not supported (missing features)

---

## 🎓 Installation Best Practices

### 1. Use Virtual Environments
```bash
# Create virtual environment
python -m venv nkrepo_env

# Activate
# Windows:
nkrepo_env\Scripts\activate
# Linux/Mac:
source nkrepo_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install in Order
```bash
# 1. Update pip first
python -m pip install --upgrade pip

# 2. Install build dependencies (if needed)
pip install wheel setuptools

# 3. Install core dependencies
pip install lief pefile yara-python

# 4. Install optional dependencies
pip install ssdeep py-tlsh  # May fail - that's OK

# 5. Verify
python utils/check_lief_version.py
```

### 3. Keep Dependencies Updated
```bash
# Update all packages
pip install --upgrade lief pefile yara-python

# Or update from requirements
pip install --upgrade -r requirements.txt
```

### 4. Document Your Environment
```bash
# Save current versions
pip freeze > requirements-lock.txt

# This ensures reproducible installations
```

---

## 📋 Verification Checklist

After installation, verify everything works:

```bash
# 1. Check LIEF
python -c "import lief; print(f'✓ LIEF {lief.__version__}')"

# 2. Check pefile
python -c "import pefile; print('✓ pefile')"

# 3. Check YARA
python -c "import yara; print('✓ YARA')"

# 4. Check SSDEEP (optional)
python -c "import ssdeep; print('✓ SSDEEP')" 2>/dev/null || echo "⚠ SSDEEP not installed"

# 5. Check TLSH (optional)
python -c "import tlsh; print('✓ TLSH')" 2>/dev/null || echo "⚠ TLSH not installed"

# 6. Run compatibility checker
python utils/check_lief_version.py

# 7. Test packer detector
python utils/packer_detector.py "C:/Windows/System32/notepad.exe"

# 8. Test YARA rules
python -c "import yara; yara.compile('yara_rules/packers_complete.yar'); print('✓ YARA rules OK')"

# 9. Test hash calculator
python utils/pe_hash_calculator.py "C:/Windows/System32/calc.exe"
```

**Expected Output:**
```
✓ LIEF 0.17.3
✓ pefile
✓ YARA
✓ SSDEEP
✓ TLSH
[OK] LIEF is installed
...
[+] Tests passed!
```

---

## 🌐 Platform-Specific Instructions

### Windows 10/11

**Recommended Setup:**
1. Install Python 3.11 from python.org
2. Add Python to PATH during installation
3. Open Command Prompt as Administrator
4. Install dependencies:

```cmd
python -m pip install --upgrade pip
pip install lief pefile yara-python tqdm

REM Optional (may require Visual C++ Build Tools):
pip install ssdeep py-tlsh
```

---

### Ubuntu/Debian Linux

```bash
# Update system
sudo apt-get update

# Install build dependencies
sudo apt-get install -y python3-pip python3-dev build-essential libfuzzy-dev

# Install Python packages
pip3 install lief pefile yara-python ssdeep py-tlsh tqdm

# Verify
python3 utils/check_lief_version.py
```

---

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python ssdeep yara

# Install Python packages
pip3 install lief pefile yara-python ssdeep py-tlsh tqdm

# Verify
python3 utils/check_lief_version.py
```

---

## 📦 Docker Installation (Advanced)

Create a Docker container with all dependencies:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libfuzzy-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    lief \
    pefile \
    yara-python \
    ssdeep \
    py-tlsh \
    tqdm

# Copy tools
COPY utils /app/utils
COPY yara_rules /app/yara_rules

WORKDIR /app

# Verify installation
RUN python utils/check_lief_version.py
```

**Build and run:**
```bash
docker build -t nkrepo-analyzer .
docker run -v /path/to/samples:/samples nkrepo-analyzer \
    python utils/packer_detector.py /samples/malware.exe
```

---

## ✅ Installation Complete!

After following this guide, you should have:

- ✅ All core dependencies installed
- ✅ Optional dependencies (SSDEEP, TLSH) if possible
- ✅ All tools verified and working
- ✅ Environment ready for malware analysis

**Test your setup:**
```bash
python utils/windows_packer_scanner.py --preset quick --max-files 10
```

If this runs without errors, your installation is complete! 🎉

---

*NKREPO - NKAMG Malware Analysis System*
