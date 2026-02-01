# Requirements.txt Update Summary

**Date:** 2026-02-01
**Purpose:** Added dependencies for new malware analysis tools

---

## 📦 What Was Updated

### 1. Main Requirements (`web/flask/requirements.txt`)

**Added Dependencies:**
```
# Malware Analysis & PE Tools
lief>=0.14.0              # PE parser for packer detection
yara-python>=4.3.0        # YARA rule scanning
ssdeep>=3.4               # Fuzzy hashing
py-tlsh>=4.7.2            # TLSH similarity hash
```

**Organization:**
- ✅ Added category headers for better readability
- ✅ Grouped related packages together
- ✅ Added comments explaining each package

**Before:**
```
pefile==2024.8.26
requests-toolbelt==1.0.0
wordfreq
```

**After:**
```
# ============================================================================
# Malware Analysis & PE Tools
# ============================================================================
# PE file parsing and analysis
pefile==2024.8.26
lief>=0.14.0

# YARA rule scanning
yara-python>=4.3.0

# Fuzzy hashing for similarity detection
ssdeep>=3.4
py-tlsh>=4.7.2
```

---

### 2. Utilities Requirements (`utils/requirements.txt`)

**Created New File:**
- Dedicated requirements file for malware analysis utilities
- Detailed installation notes for each platform
- Tool-specific dependency matrix
- Troubleshooting guide

**Contents:**
```
lief>=0.14.0         # Core PE parser
pefile>=2024.8.26    # PE analysis
yara-python>=4.3.0   # YARA scanning
ssdeep>=3.4          # Fuzzy hashing (optional)
py-tlsh>=4.7.2       # TLSH hashing (optional)
tqdm>=4.65.0         # Progress bars
```

---

### 3. Dependencies Guide (`DEPENDENCIES_GUIDE.md`)

**Created Comprehensive Guide:**
- ✅ Quick installation commands
- ✅ Platform-specific instructions (Windows/Linux/macOS)
- ✅ Troubleshooting section
- ✅ Dependency matrix by tool
- ✅ Version compatibility table
- ✅ Verification checklist

---

## 🎯 New Dependencies Explained

### lief (>=0.14.0)
**Purpose:** Modern PE file parser
**Used By:**
- `packer_detector.py` - PE structure analysis
- `windows_packer_scanner.py` - System-wide scanning
- `yara_packer_scanner.py` - Combined YARA+LIEF detection

**Why:** LIEF provides better API compatibility and modern PE parsing features

**Installation:**
```bash
pip install lief
```

---

### yara-python (>=4.3.0)
**Purpose:** YARA rule scanning engine
**Used By:**
- `yara_packer_scanner.py` - YARA-based packer detection
- `collect_yara_rules.py` - Rule validation

**Why:** Industry standard for malware pattern matching

**Installation:**
```bash
pip install yara-python
```

---

### pefile (2024.8.26)
**Purpose:** Classic PE file analysis (already installed)
**Used By:**
- `pe_hash_calculator.py` - Imphash, authentihash, Rich header

**Why:** Provides specialized PE features like imphash

**Status:** ✅ Already in requirements.txt

---

### ssdeep (>=3.4)
**Purpose:** Fuzzy hashing for similarity detection
**Used By:**
- `pe_hash_calculator.py` - SSDEEP hash calculation

**Why:** Detects similar but modified files

**Installation:**
```bash
# May require Visual C++ Build Tools on Windows
pip install ssdeep
```

**Note:** Optional - tools work without it

---

### py-tlsh (>=4.7.2)
**Purpose:** Advanced similarity hashing
**Used By:**
- `pe_hash_calculator.py` - TLSH hash calculation

**Why:** More robust than SSDEEP for large files

**Installation:**
```bash
pip install py-tlsh
```

**Note:** Optional - tools work without it

---

## 📊 Dependency Matrix

### Core Tools (Required)

| Tool | lief | pefile | yara-python | ssdeep | tlsh |
|------|------|--------|-------------|--------|------|
| **packer_detector.py** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **windows_packer_scanner.py** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **yara_packer_scanner.py** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **pe_hash_calculator.py** | ❌ | ⚠️ | ❌ | ⚠️ | ⚠️ |
| **collect_yara_rules.py** | ❌ | ❌ | ❌ | ❌ | ❌ |

Legend:
- ✅ Required
- ⚠️ Optional (enhanced features)
- ❌ Not needed

---

## 🚀 Quick Installation

### Option 1: Install Everything
```bash
# Main application
cd web/flask
pip install -r requirements.txt

# Utilities
cd ../../utils
pip install -r requirements.txt
```

### Option 2: Minimal (Core Tools Only)
```bash
pip install lief pefile yara-python
```

### Option 3: No Optional Dependencies
```bash
# Skip ssdeep and tlsh if installation issues
pip install lief pefile yara-python tqdm
```

---

## ✅ Verification

After installation, verify everything works:

```bash
# 1. Check LIEF
python -c "import lief; print(f'LIEF {lief.__version__} installed')"

# 2. Check pefile
python -c "import pefile; print('pefile installed')"

# 3. Check YARA
python -c "import yara; print('YARA installed')"

# 4. Check optional
python -c "import ssdeep; print('SSDEEP installed')" 2>/dev/null || echo "SSDEEP not installed (OK)"
python -c "import tlsh; print('TLSH installed')" 2>/dev/null || echo "TLSH not installed (OK)"

# 5. Run compatibility checker
python utils/check_lief_version.py

# 6. Test tools
python utils/packer_detector.py "C:/Windows/System32/notepad.exe"
```

**Expected Output:**
```
LIEF 0.17.3 installed
pefile installed
YARA installed
SSDEEP not installed (OK)
TLSH not installed (OK)

[OK] LIEF is installed
  Version: 0.17.3
...
```

---

## 🐛 Common Installation Issues

### Issue 1: SSDEEP Compilation Fails (Windows)

**Error:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**
```bash
# Option A: Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Option B: Skip SSDEEP
pip install lief pefile yara-python py-tlsh
# (Tools will work without SSDEEP)
```

---

### Issue 2: yara-python Build Error

**Solution:**
```bash
# Try pre-built binary
pip install yara-python --only-binary :all:

# Or use conda (if available)
conda install -c conda-forge yara-python
```

---

### Issue 3: Import Errors

**Error:**
```python
ModuleNotFoundError: No module named 'lief'
```

**Solution:**
```bash
# Verify correct Python environment
pip list | grep lief

# Install in current environment
python -m pip install lief

# Or use virtual environment
python -m venv env
env\Scripts\activate  # Windows
source env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## 📚 Updated Files

### Modified Files
1. ✅ `web/flask/requirements.txt` - Added malware analysis dependencies

### New Files
1. ✅ `utils/requirements.txt` - Utilities-specific requirements
2. ✅ `DEPENDENCIES_GUIDE.md` - Comprehensive installation guide
3. ✅ `REQUIREMENTS_UPDATE_SUMMARY.md` - This file

---

## 📈 Before vs After

### Before (Old requirements.txt)
```
pefile==2024.8.26
requests-toolbelt==1.0.0
wordfreq
tld
xgboost
```

**Issues:**
- ❌ No LIEF (packer detector doesn't work)
- ❌ No YARA (YARA scanner doesn't work)
- ❌ No fuzzy hashing support
- ❌ No organization or comments

### After (Updated requirements.txt)
```
# ============================================================================
# Malware Analysis & PE Tools
# ============================================================================
pefile==2024.8.26
lief>=0.14.0
yara-python>=4.3.0
ssdeep>=3.4
py-tlsh>=4.7.2
```

**Benefits:**
- ✅ All tools work out of box
- ✅ Better organization
- ✅ Clear comments
- ✅ Optional dependencies marked
- ✅ Version constraints specified

---

## 🎯 Impact on Existing Setup

### Existing Users
If you already have the repository:

```bash
# Update dependencies
cd web/flask
pip install --upgrade -r requirements.txt

# Install new utilities dependencies
cd ../../utils
pip install -r requirements.txt

# Verify
python utils/check_lief_version.py
```

### New Users
Fresh installation:

```bash
# Clone repository
git clone https://github.com/your/nkrepo.git
cd nkrepo

# Install dependencies
pip install -r web/flask/requirements.txt
pip install -r utils/requirements.txt

# Verify
python utils/check_lief_version.py
```

---

## 🔄 Version Control

### requirements.txt versioning
- **pefile:** Exact version (==2024.8.26) - stable API
- **lief:** Minimum version (>=0.14.0) - compatible with all 0.14+
- **yara-python:** Minimum version (>=4.3.0) - API stable
- **ssdeep:** Minimum version (>=3.4) - rare updates
- **py-tlsh:** Minimum version (>=4.7.2) - compatible

### Why minimum versions?
- ✅ Ensures compatibility with older systems
- ✅ Allows security updates
- ✅ Doesn't break on newer versions
- ✅ Future-proof

---

## 📦 Package Sizes

Approximate download/install sizes:

| Package | Size | Build Time |
|---------|------|------------|
| lief | ~50 MB | Fast (pre-built) |
| pefile | ~1 MB | Instant |
| yara-python | ~5 MB | Medium |
| ssdeep | ~1 MB | Slow (compilation) |
| py-tlsh | ~2 MB | Medium |
| **Total** | **~60 MB** | **2-5 minutes** |

---

## ✅ Summary

### What Changed
- ✅ Added 4 new dependencies (lief, yara-python, ssdeep, py-tlsh)
- ✅ Organized requirements.txt with categories
- ✅ Created dedicated utils/requirements.txt
- ✅ Created comprehensive dependencies guide
- ✅ Added installation verification steps

### What Works Now
- ✅ Packer detection (LIEF-based)
- ✅ YARA scanning (signature-based)
- ✅ Hash calculation (11 hash types)
- ✅ Windows system scanning
- ✅ Fuzzy hashing (optional)

### Next Steps
1. Run: `pip install -r web/flask/requirements.txt`
2. Run: `pip install -r utils/requirements.txt`
3. Verify: `python utils/check_lief_version.py`
4. Test: `python utils/packer_detector.py "C:/Windows/System32/notepad.exe"`

**All tools should now work perfectly! 🎉**

---

*NKREPO - NKAMG Malware Analysis System*
*Updated: 2026-02-01*
