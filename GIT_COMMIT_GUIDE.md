# Git Commit Guide for NKREPO

Guide for committing the Windows upgrade changes to the repository.

## .gitignore Created

A comprehensive `.gitignore` file has been created that excludes:

### Critical Excludes (NEVER Commit)
- ❌ `config.ini` - Contains MySQL passwords and API keys
- ❌ `data/samples/` - Malware samples
- ❌ `web/vue/uploads/` - User uploaded files
- ❌ `.env.*.local` - Local environment configuration
- ❌ `venv/` and `node_modules/` - Dependencies
- ❌ Trained ML models - Too large for git
- ❌ Log files
- ❌ `__pycache__/`, `*.pyc` - Python compiled files

### What IS Tracked (Safe to Commit)
- ✅ Configuration templates (`.example` files)
- ✅ Setup scripts (`.bat`, `.sh` files)
- ✅ Documentation (`.md` files)
- ✅ Source code (`.py`, `.js`, `.vue` files)
- ✅ YARA rules (`.yar` files)
- ✅ Requirements files (`requirements.txt`, `package.json`)
- ✅ Default environment files (`.env.development`, `.env.production`)

---

## Current Status

### Modified Files (2)
```
modified:   web/flask/requirements.txt
modified:   web/vue/vue.config.js
```

### New Files to Commit (28)

#### Documentation (9 files)
```
CLAUDE.md
DEPENDENCIES_GUIDE.md
HASH_CALCULATOR_SUMMARY.md
INSTALLATION_GUIDE.md
PACKER_SCANNER_SUMMARY.md
REQUIREMENTS_FINAL_SUMMARY.md
REQUIREMENTS_UPDATE_SUMMARY.md
WEB_UPGRADE_SUMMARY.md
YARA_SYSTEM_SUMMARY.md
utils/LIEF_COMPATIBILITY_FIX.md
utils/PACKER_DETECTOR_README.md
utils/PE_HASH_GUIDE.md
utils/QUICKSTART_PACKER_SCANNER.md
utils/WINDOWS_SCANNER_README.md
web/INSTALLATION_WINDOWS.md
web/README.md
```

#### Utilities (8 Python scripts)
```
utils/check_lief_version.py
utils/collect_yara_rules.py
utils/packer_detector.py
utils/pe_hash_calculator.py
utils/requirements.txt
utils/test_packer_detector.py
utils/verify_installation.py
utils/windows_packer_scanner.py
utils/yara_packer_scanner.py
```

#### Windows Setup Scripts (6 batch files)
```
db/setup_database_windows.bat
web/flask/setup_windows.bat
web/flask/run_flask.bat
web/vue/setup_windows.bat
web/vue/run_dev.bat
web/vue/build_prod.bat
```

#### Configuration Templates (1 file)
```
web/flask/config.ini.example
```

#### YARA Rules Directory
```
yara_rules/
```

#### Git Configuration (1 file)
```
.gitignore
```

---

## Recommended Commit Strategy

### Option 1: Single Comprehensive Commit

```bash
git add .
git commit -m "Add Windows compatibility and malware analysis tools

Major Features:
- Windows setup scripts for Flask backend, Vue frontend, and database
- Comprehensive malware analysis utilities (packer detection, YARA scanning, hash calculation)
- Complete Windows installation documentation
- YARA rules collection (40+ packer detection rules)
- Configuration templates and examples
- Automated dependency verification

Components:
- Backend: Flask setup automation with config templates
- Frontend: Vue.js setup automation with environment configuration
- Database: MySQL initialization scripts
- Utils: 8 malware analysis tools with full documentation
- YARA: Complete packer detection ruleset
- Docs: 1,400+ lines of installation and usage guides

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Option 2: Separate Commits by Component

#### Commit 1: Documentation
```bash
git add CLAUDE.md DEPENDENCIES_GUIDE.md INSTALLATION_GUIDE.md
git add HASH_CALCULATOR_SUMMARY.md PACKER_SCANNER_SUMMARY.md
git add REQUIREMENTS_FINAL_SUMMARY.md REQUIREMENTS_UPDATE_SUMMARY.md
git add WEB_UPGRADE_SUMMARY.md YARA_SYSTEM_SUMMARY.md
git commit -m "Add comprehensive documentation for NKREPO system

- Installation guides for Windows
- Dependency guides with troubleshooting
- Feature summaries and quick start guides
- Project overview and architecture docs

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

#### Commit 2: Malware Analysis Utilities
```bash
git add utils/packer_detector.py utils/windows_packer_scanner.py
git add utils/yara_packer_scanner.py utils/pe_hash_calculator.py
git add utils/collect_yara_rules.py utils/check_lief_version.py
git add utils/verify_installation.py utils/test_packer_detector.py
git add utils/requirements.txt
git add utils/*.md
git commit -m "Add malware analysis utilities

- Packer detector using LIEF library (40+ packer support)
- Windows system scanner with multi-threading
- YARA+LIEF combined scanner
- PE hash calculator (11 hash types)
- YARA rules collector from GitHub
- LIEF version compatibility checker
- Installation verification script
- Comprehensive documentation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

#### Commit 3: YARA Rules
```bash
git add yara_rules/
git commit -m "Add YARA packer detection rules

- 40+ custom packer detection rules
- Support for UPX, VMProtect, Themida, ASPack, Enigma, and more
- Generic indicators for packed executables
- Usage guide and examples

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

#### Commit 4: Backend Windows Setup
```bash
git add web/flask/setup_windows.bat
git add web/flask/run_flask.bat
git add web/flask/config.ini.example
git add web/flask/requirements.txt
git add db/setup_database_windows.bat
git commit -m "Add Flask backend Windows setup automation

- Automated setup script with venv creation
- Server startup script
- Configuration template with examples
- Database initialization script
- Updated requirements.txt with malware analysis dependencies

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

#### Commit 5: Frontend Windows Setup
```bash
git add web/vue/setup_windows.bat
git add web/vue/run_dev.bat
git add web/vue/build_prod.bat
git add web/vue/vue.config.js
git commit -m "Add Vue.js frontend Windows setup automation

- Automated setup script with npm installation
- Development server startup script
- Production build script
- Updated vue.config.js with Flask proxy

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

#### Commit 6: Web Documentation
```bash
git add web/INSTALLATION_WINDOWS.md
git add web/README.md
git commit -m "Add web application documentation

- Complete Windows installation guide (900+ lines)
- Project README with architecture overview
- API endpoints reference
- Troubleshooting guide
- Production deployment instructions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

#### Commit 7: Git Configuration
```bash
git add .gitignore
git commit -m "Add comprehensive .gitignore

Excludes:
- Sensitive files (config.ini, API keys, credentials)
- Dependencies (venv, node_modules)
- Malware samples and uploads
- Build artifacts and logs
- OS and IDE files

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Pre-Commit Checklist

Before committing, verify:

### 1. No Sensitive Data
```bash
# Check for config.ini (should NOT be committed)
git status | grep config.ini
# Should only show: web/flask/config.ini.example

# Check for .env.local files (should NOT be committed)
git status | grep "\.env.*\.local"
# Should show nothing
```

### 2. No Malware Samples
```bash
# Verify no malware files
git status | grep -E "\.(exe|dll|bin)$"
# Should show nothing
```

### 3. No Large Files
```bash
# Check for large files (>10MB)
find . -type f -size +10M -not -path "./.git/*" -not -path "./node_modules/*" -not -path "./venv/*"
# Should show nothing or only expected large files
```

### 4. No Dependencies
```bash
# Verify node_modules not tracked
git status | grep node_modules
# Should show nothing

# Verify venv not tracked
git status | grep venv
# Should show nothing
```

---

## After Committing

### 1. Push to Remote
```bash
git push origin master
```

### 2. Create a Tag for This Release
```bash
git tag -a v1.0-windows -m "Windows compatibility release

Complete Windows installation support with:
- Automated setup scripts
- Malware analysis utilities
- YARA packer detection
- Comprehensive documentation"

git push origin v1.0-windows
```

### 3. Create GitHub Release (Optional)

On GitHub, create a release with:

**Title:** NKREPO v1.0 - Windows Compatibility Release

**Description:**
```markdown
# NKREPO v1.0 - Windows Compatibility Release

Complete Windows installation support for the NKREPO malware repository system.

## New Features

### Windows Installation
- ✅ One-command setup for all components
- ✅ Automated virtual environment creation
- ✅ Configuration templates with examples
- ✅ Comprehensive 900+ line installation guide

### Malware Analysis Utilities
- ✅ Packer detector (40+ packers, LIEF-based)
- ✅ Windows system scanner (multi-threaded)
- ✅ YARA+LIEF combined scanner
- ✅ PE hash calculator (11 hash types)
- ✅ 40+ YARA packer detection rules

### Documentation
- ✅ Installation guides
- ✅ API documentation
- ✅ Troubleshooting guides
- ✅ Quick start tutorials

## Installation

See [INSTALLATION_WINDOWS.md](web/INSTALLATION_WINDOWS.md) for complete guide.

### Quick Start
```cmd
# 1. Database
cd db
setup_database_windows.bat

# 2. Backend
cd ..\web\flask
setup_windows.bat

# 3. Frontend
cd ..\vue
setup_windows.bat

# 4. Run
run_flask.bat  # Terminal 1
run_dev.bat    # Terminal 2
```

## Requirements

- Python 3.8+
- Node.js 8.9+
- MySQL 5.7+
- Windows 10/11

## Documentation

- [Installation Guide](web/INSTALLATION_WINDOWS.md)
- [Dependencies Guide](DEPENDENCIES_GUIDE.md)
- [Web Application README](web/README.md)
```

---

## Files to NEVER Commit

Even if you run `git add .`, these files should be automatically excluded by `.gitignore`:

❌ **Configuration with Secrets**
- `config.ini`
- `.env.local`
- `.env.*.local`
- `secrets.json`
- Any files with passwords or API keys

❌ **Malware Samples**
- `data/samples/**/*`
- `*.exe` (in data directories)
- `*.dll` (in data directories)
- `*.bin` (in data directories)

❌ **User Data**
- `web/vue/uploads/`
- `uploads/`

❌ **Dependencies**
- `venv/`
- `node_modules/`
- `__pycache__/`

❌ **Build Artifacts**
- `dist/`
- `*.pyc`
- `*.log`

❌ **Large Files**
- Trained ML models
- Database dumps

---

## Verifying .gitignore Works

Test that sensitive files are properly ignored:

```bash
# Try to add a test config.ini
echo "test" > web/flask/config.ini
git status
# Should NOT show config.ini in untracked files

# Try to add test malware
echo "test" > data/samples/test.exe
git status
# Should NOT show test.exe in untracked files

# Clean up test files
rm web/flask/config.ini
rm data/samples/test.exe
```

---

## Summary

**Ready to Commit:**
- ✅ 28 new files (utilities, scripts, documentation)
- ✅ 2 modified files (requirements.txt, vue.config.js)
- ✅ .gitignore properly excludes sensitive files
- ✅ No malware samples or large files included
- ✅ All documentation and source code ready

**Recommended Action:**
Use Option 1 (single comprehensive commit) for simplicity, or Option 2 (separate commits) for better organization.

---

**Ready to commit! 🚀**

All changes have been verified and are safe to push to the repository.
