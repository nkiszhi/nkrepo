# PE Packer Scanner - Quick Start Guide

Complete guide to scan Windows systems for packed executables.

## 📦 What You Have

Three powerful tools for packer detection:

1. **packer_detector.py** - Single file analyzer
2. **windows_packer_scanner.py** - System-wide scanner
3. **test_packer_detector.py** - Testing utilities

## 🚀 Installation

### Requirements
```bash
pip install lief
```

### Verify Installation
```bash
python utils/check_lief_version.py
```

Expected output:
```
[OK] LIEF is installed
  Version: 0.17.3-03aca30b
[OK] LIEF.PE module is available
```

## 🎯 Quick Start Examples

### 1. Analyze a Single File
```bash
# Basic detection
python utils/packer_detector.py "C:/path/to/sample.exe"

# With detailed PE information
python utils/packer_detector.py "C:/path/to/sample.exe" --info
```

**Output Example:**
```
======================================================================
PE PACKER DETECTION RESULTS
======================================================================

File Information:
  Name:         upx_sample.exe
  Size:         45,056 bytes
  Architecture: 32-bit
  Sections:     3
  Imports:      1
  Entry Point:  0x00001000

Detection Verdict:
  Status:     PACKED
  Packer:     UPX
  Confidence: 85%

Suspicious Features Detected:
  [1] Packer signature detected: UPX (UPX0)
  [2] High entropy section: UPX1 (entropy=7.82)
  [3] Very few imports: 4 functions
  [4] Entry point not in .text section: UPX0
======================================================================
```

### 2. Quick System Scan (5-10 minutes)
```bash
python utils/windows_packer_scanner.py --preset quick --export-all
```

This scans:
- Windows System32 folder
- Your Downloads folder
- Exports to JSON, CSV, and HTML

### 3. Scan Your Malware Repository
```bash
python utils/windows_packer_scanner.py -d "C:/nkrepo/data/samples" --packed-only --export-json malware_analysis.json
```

This will:
- Scan all samples in your repository
- Show only packed files
- Export results to JSON for further analysis

### 4. Scan Specific Directory with Filtering
```bash
python utils/windows_packer_scanner.py -d "C:/Downloads" --min-confidence 70 --export-html report.html
```

This will:
- Scan Downloads folder
- Show only high-confidence detections (70%+)
- Generate beautiful HTML report

## 📊 Understanding Results

### Confidence Scores
| Score | Meaning | Action |
|-------|---------|--------|
| 85-100% | Definitely packed | High priority analysis |
| 70-84% | Very likely packed | Review manually |
| 40-69% | Possibly packed | Verify with additional tools |
| 1-39% | Low confidence | Likely false positive |
| 0% | Not packed | Clean file |

### Common Packers Detected
- **UPX** - Most common, easy to unpack
- **VMProtect** - Commercial protection, hard to unpack
- **Themida/WinLicense** - Commercial, very strong
- **ASPack** - Common in malware
- **Enigma** - Installer/trial protection
- **Unknown Packer** - Custom or unknown protection

## 🛠️ Common Workflows

### Workflow 1: Daily Security Scan
```bash
# Morning scan of common locations
python utils/windows_packer_scanner.py --preset quick --packed-only

# Review packed files
# Investigate any unexpected findings
```

### Workflow 2: New Sample Analysis
```bash
# 1. Quick check of single file
python utils/packer_detector.py "suspicious.exe"

# 2. If packed, view detailed info
python utils/packer_detector.py "suspicious.exe" --info

# 3. Check entry point and sections manually
```

### Workflow 3: Batch Sample Processing
```bash
# Scan entire sample directory
python utils/windows_packer_scanner.py \
    -d "C:/samples" \
    --packed-only \
    --min-confidence 50 \
    --export-all

# Results saved as:
# - packer_scan_YYYYMMDD_HHMMSS.json
# - packer_scan_YYYYMMDD_HHMMSS.csv
# - packer_scan_YYYYMMDD_HHMMSS.html
```

### Workflow 4: System Integrity Check
```bash
# Scan system files for anomalies
python utils/windows_packer_scanner.py \
    --preset system \
    --packed-only \
    --min-confidence 60 \
    --export-csv system_packed.csv

# Review CSV in Excel
# Investigate any unknown packed system files
```

## 📁 Output Files

### JSON Format (Best for Analysis)
```json
{
  "scan_info": { ... },
  "statistics": {
    "total_files": 2847,
    "packed_files": 23,
    "packer_distribution": { "UPX": 15, "VMProtect": 5 }
  },
  "results": [ ... ]
}
```

**Use for:**
- Import into analysis tools
- Database storage
- Automated processing

### CSV Format (Best for Spreadsheets)
Opens directly in Excel with columns:
- File Path, Filename, Is Packed, Packer Type
- Confidence %, Architecture, Size, Sections
- Imports, Suspicious Features

**Use for:**
- Excel pivot tables
- Quick filtering/sorting
- Data visualization

### HTML Report (Best for Presentation)
Beautiful web report with:
- Summary statistics cards
- Interactive charts
- Color-coded results table
- Professional styling

**Use for:**
- Sharing with team
- Management reports
- Documentation

## 🔥 Real-World Examples

### Example 1: Found UPX Malware in Downloads
```bash
$ python utils/windows_packer_scanner.py -d "C:/Users/*/Downloads" --packed-only

PACKED FILES (Top 20)
----------------------------------------------------------------------
File                                     Packer               Conf%
----------------------------------------------------------------------
suspicious_installer.exe                 UPX                  92
game_crack.exe                           UPX                  88
keygen.exe                               Unknown Packer       75
```

**Action:** Investigate these files immediately!

### Example 2: System File Tampering Detection
```bash
$ python utils/windows_packer_scanner.py --preset system --packed-only

Total Files Scanned:  4,823
Packed Files Found:   1
Unpacked Files:       4,822

PACKED FILES
----------------------------------------------------------------------
svchost.exe (C:/Windows/System32/fake/)  Unknown Packer       65
```

**Action:** This is suspicious! System files should NOT be packed.

### Example 3: Malware Repository Analysis
```bash
$ python utils/windows_packer_scanner.py -d "C:/nkrepo/data/samples" --export-all

Total Files Scanned:  1,523
Packed Files Found:   847
Unpacked Files:       676

PACKER DISTRIBUTION
----------------------------------------------------------------------
  UPX                               423 (27.8%)
  VMProtect                         198 (13.0%)
  Themida                            89 ( 5.8%)
  ASPack                             67 ( 4.4%)
  Unknown Packer                     70 ( 4.6%)
```

**Insight:** 56% of malware samples are packed!

## 🎓 Pro Tips

### Tip 1: Start Small
Always test with `--max-files 100` first to estimate scan time.

### Tip 2: Use Threading Wisely
```bash
# Fast machine (8+ cores)
python utils/windows_packer_scanner.py -t 8 --preset quick

# Average machine (4 cores)
python utils/windows_packer_scanner.py -t 4 --preset quick

# Slow machine or server
python utils/windows_packer_scanner.py -t 2 --preset quick
```

### Tip 3: Filter Smart
```bash
# Reduce noise - only show high confidence
--min-confidence 70

# Reduce results - only packed files
--packed-only

# Combine for best results
--packed-only --min-confidence 60
```

### Tip 4: Schedule Regular Scans
Create a batch file `daily_scan.bat`:
```batch
@echo off
cd C:\nkrepo\utils
python windows_packer_scanner.py --preset quick --packed-only --export-all
```

Schedule via Windows Task Scheduler to run daily.

### Tip 5: Export Everything
Always use `--export-all` for important scans. You can always filter later!

## ⚡ Performance Reference

| Files | Threads | Time (Est.) | Recommended For |
|-------|---------|-------------|-----------------|
| 100 | 4 | 30 sec | Testing |
| 1,000 | 4 | 3 min | Quick scans |
| 5,000 | 4 | 15 min | Medium scans |
| 20,000 | 8 | 45 min | Full program scan |
| 100,000+ | 8 | 2-4 hours | Complete system |

*Times vary based on CPU, disk speed, and file sizes*

## 🐛 Troubleshooting

### Problem: "Permission denied" errors
**Solution:**
```bash
# Run as Administrator (right-click CMD -> Run as Admin)
python utils/windows_packer_scanner.py --preset system
```

### Problem: Scan is too slow
**Solutions:**
1. Reduce threads: `-t 2`
2. Limit files: `--max-files 5000`
3. Scan specific folder instead of preset

### Problem: Too many false positives
**Solutions:**
1. Increase min confidence: `--min-confidence 70`
2. Filter by specific packers: `--packer-type UPX --packer-type VMProtect`
3. Review "Unknown Packer" results manually

### Problem: ModuleNotFoundError: 'lief'
**Solution:**
```bash
pip install lief
```

## 📚 Next Steps

1. **Run your first scan:**
   ```bash
   python utils/windows_packer_scanner.py --preset quick --export-all
   ```

2. **Review the HTML report** (opens in browser)

3. **Investigate any packed files** found

4. **Read the full documentation:**
   - `PACKER_DETECTOR_README.md` - Single file detection details
   - `WINDOWS_SCANNER_README.md` - System scanner guide
   - `LIEF_COMPATIBILITY_FIX.md` - Technical details

## 🎯 Common Use Cases

### Security Analyst
```bash
# Daily: Scan downloads and desktop
python utils/windows_packer_scanner.py --preset downloads --packed-only

# Weekly: Full system check
python utils/windows_packer_scanner.py --preset full --min-confidence 60 --export-all
```

### Malware Researcher
```bash
# Analyze sample collection
python utils/windows_packer_scanner.py -d "C:/samples" --export-json analysis.json

# Find specific packer
python utils/windows_packer_scanner.py -d "C:/samples" --packer-type UPX --export-csv upx_samples.csv
```

### System Administrator
```bash
# Check installed software
python utils/windows_packer_scanner.py --preset programs --packed-only --export-html software_report.html

# Monitor system integrity
python utils/windows_packer_scanner.py --preset system --min-confidence 70
```

## 📞 Getting Help

If you encounter issues:

1. **Check LIEF compatibility:**
   ```bash
   python utils/check_lief_version.py
   ```

2. **Test with a known file:**
   ```bash
   python utils/packer_detector.py "C:/Windows/System32/notepad.exe"
   ```

3. **Enable verbose mode:**
   ```bash
   python utils/windows_packer_scanner.py -v -d "C:/test" --max-files 10
   ```

---

**Ready to scan? Start with:**
```bash
python utils/windows_packer_scanner.py --preset quick --export-all
```

Happy hunting! 🔍
