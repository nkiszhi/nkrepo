# PE Packer Detector - Usage Guide

A practical PE packer detector for malware analysis using the LIEF library.

## Features

### Detection Dimensions (Weighted)
1. **Section Table Analysis** (Highest Weight)
   - Packer-specific section names (UPX, VMProtect, ASPack, Themida, Enigma, etc.)
   - RWX (Read-Write-Execute) permission sections
   - Missing standard `.text` section
   - Suspicious section names

2. **Import Table Anomaly Detection**
   - Files with <5 imported functions
   - Files only importing LoadLibraryA/GetProcAddress

3. **Entropy Analysis**
   - Shannon entropy calculation for all sections
   - Flags sections with entropy >7.0 (compressed/encrypted)

4. **Entry Point (OEP) Verification**
   - Flags files where entry point is NOT in `.text` section

### Supported Packers
- UPX (Ultimate Packer for Executables)
- VMProtect
- ASPack
- Themida/WinLicense
- Enigma Virtual Box
- PECompact
- NSPack
- Armadillo
- Obsidium
- PESpin
- Petite
- MPRESS

## Installation

### Requirements
- Python 3.8+
- LIEF library

### Install LIEF
```bash
pip install lief
```

Or install from the project requirements:
```bash
cd web/flask
pip install lief
```

## Usage

### Basic Usage
```bash
python utils/packer_detector.py <pe_file_path>
```

### With Detailed PE Information
```bash
python utils/packer_detector.py <pe_file_path> --info
```

### Examples

#### Example 1: Detect UPX-packed executable
```bash
python utils/packer_detector.py C:\samples\upx_packed.exe
```

**Expected Output:**
```
----------------------------------------------------------------------
PE Packer Detector - NKAMG Malware Analysis Tool
----------------------------------------------------------------------

[*] Analyzing PE file...

======================================================================
PE PACKER DETECTION RESULTS
======================================================================

File Information:
  Name:         upx_packed.exe
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
  [2] Packer signature detected: UPX (UPX1)
  [3] High entropy section: UPX1 (entropy=7.82)
  [4] Very few imports: 4 functions
  [5] Entry point not in .text section: UPX0 (RVA=0x00001000)
======================================================================
```

#### Example 2: Analyze legitimate (unpacked) executable
```bash
python utils/packer_detector.py C:\Windows\System32\notepad.exe --info
```

**Expected Output:**
```
----------------------------------------------------------------------
PE Packer Detector - NKAMG Malware Analysis Tool
----------------------------------------------------------------------

[*] Analyzing PE file...

======================================================================
PE FILE INFORMATION
======================================================================

File: notepad.exe
Size: 236,544 bytes
Architecture: 64-bit
Entry Point: 0x00006e40

--------------------------SECTION TABLE--------------------------------
Name         VirtAddr     VirtSize     RawSize      Entropy
----------------------------------------------------------------------
.text        0x00001000   0x00012c34   0x00013000   6.42
.rdata       0x00014000   0x00008a98   0x00009000   5.89
.data        0x0001d000   0x000001f0   0x00000200   3.12
.pdata       0x0001e000   0x00000d2c   0x00000e00   5.23
.rsrc        0x0001f000   0x00001560   0x00001600   4.78
.reloc       0x00021000   0x00000164   0x00000200   4.01

--------------------------IMPORT TABLE---------------------------------
KERNEL32.dll: 87 functions
USER32.dll: 34 functions
GDI32.dll: 12 functions
Total imported functions: 133
======================================================================

======================================================================
PE PACKER DETECTION RESULTS
======================================================================

File Information:
  Name:         notepad.exe
  Size:         236,544 bytes
  Architecture: 64-bit
  Sections:     6
  Imports:      3
  Entry Point:  0x00006e40

Detection Verdict:
  Status:     NOT PACKED
  Packer:     No Packer
  Confidence: 0%

No suspicious features detected.
======================================================================
```

#### Example 3: Unknown packer detection
```bash
python utils/packer_detector.py C:\samples\custom_packed.exe
```

**Expected Output:**
```
======================================================================
PE PACKER DETECTION RESULTS
======================================================================

File Information:
  Name:         custom_packed.exe
  Size:         128,000 bytes
  Architecture: 32-bit
  Sections:     2
  Imports:      1
  Entry Point:  0x00002000

Detection Verdict:
  Status:     PACKED
  Packer:     Unknown Packer
  Confidence: 62%

Suspicious Features Detected:
  [1] Missing standard .text section
  [2] RWX section found: .data
  [3] High entropy section: .data (entropy=7.91)
  [4] Very few imports: 2 functions
  [5] Only dynamic loading functions imported
  [6] Entry point not in .text section: .data (RVA=0x00002000)
======================================================================
```

## Programmatic Usage

### As a Library
```python
from utils.packer_detector import PEPackerDetector

# Initialize detector
detector = PEPackerDetector("C:\\samples\\malware.exe")

# Run detection
result = detector.detect()

# Access results
if result['is_packed']:
    print(f"Detected packer: {result['packer_type']}")
    print(f"Confidence: {result['confidence_score']}%")

    # List suspicious features
    for feature in result['suspicious_features']:
        print(f"- {feature}")

# Print detailed PE info (optional)
detector.load_pe()
detector.print_pe_info()
```

### Result Dictionary Structure
```python
{
    'is_packed': bool,              # True if packed, False otherwise
    'packer_type': str,             # "UPX", "VMProtect", "Unknown Packer", "No Packer", etc.
    'suspicious_features': list,    # List of detected anomalies
    'confidence_score': int,        # 0-100 confidence score
    'file_info': {
        'filename': str,
        'size': int,
        'architecture': str,        # "32-bit" or "64-bit"
        'sections': int,
        'imports': int,
        'entry_point': str          # Hex string (e.g., "0x00001000")
    }
}
```

## Testing

### Test Files
You can test the detector with various samples:

1. **Legitimate executables**: Windows system files (notepad.exe, calc.exe)
2. **UPX-packed samples**: Use UPX to pack test files
3. **Malware samples**: From your malware repository (data/samples/)

### Creating UPX Test Samples
```bash
# Download UPX from https://upx.github.io/
upx --best -o packed_sample.exe original_sample.exe
```

### Batch Testing
```python
import os
from utils.packer_detector import PEPackerDetector

samples_dir = "C:\\samples"
results = []

for filename in os.listdir(samples_dir):
    if filename.endswith('.exe'):
        filepath = os.path.join(samples_dir, filename)
        detector = PEPackerDetector(filepath)
        result = detector.detect()
        results.append((filename, result['is_packed'], result['packer_type']))

# Print summary
for filename, is_packed, packer in results:
    status = "PACKED" if is_packed else "NOT PACKED"
    print(f"{filename:<30} {status:<15} {packer}")
```

## Detection Accuracy

### Confidence Score Interpretation
- **0-29%**: Likely not packed (clean file)
- **30-59%**: Possibly packed (low confidence) - Unknown packer or weak indicators
- **60-84%**: Probably packed (medium confidence) - Multiple indicators
- **85-100%**: Definitely packed (high confidence) - Packer signature found

### Known Limitations
1. **False Positives**: Some legitimate executables with unusual characteristics may trigger warnings
2. **Custom Packers**: Unknown/custom packers may be detected as "Unknown Packer" with lower confidence
3. **Polymorphic Packers**: Advanced polymorphic packers may evade signature-based detection
4. **Hybrid Protection**: Files with multiple layers of protection may only detect the outer layer

### Reducing False Positives
- Use confidence score thresholds
- Verify with additional tools (VirusTotal, manual analysis)
- Whitelist known legitimate software

## Integration with NKREPO

### Use in Malware Pipeline
```python
# In your sample processing script
from utils.packer_detector import PEPackerDetector

def process_sample(sample_path):
    detector = PEPackerDetector(sample_path)
    result = detector.detect()

    # Store packer info in database
    if result['is_packed']:
        update_sample_metadata(
            sample_path,
            is_packed=True,
            packer_type=result['packer_type'],
            confidence=result['confidence_score']
        )

    return result
```

### API Endpoint (Flask)
```python
# In web/flask/app.py
from utils.packer_detector import PEPackerDetector

@app.route('/api/detect_packer', methods=['POST'])
def detect_packer_api():
    file = request.files['file']
    temp_path = save_temp_file(file)

    detector = PEPackerDetector(temp_path)
    result = detector.detect()

    return jsonify(result)
```

## Extending the Detector

### Adding New Packer Signatures
Edit `PEPackerDetector.PACKER_SIGNATURES` in `packer_detector.py`:

```python
PACKER_SIGNATURES = {
    # ... existing signatures ...
    'YourPackerName': [
        '.custom_section1',
        '.custom_section2',
        'CustomSectionPrefix'
    ]
}
```

### Adjusting Detection Weights
Modify `PEPackerDetector.WEIGHTS` to tune sensitivity:

```python
WEIGHTS = {
    'packer_signature': 40,      # Increase for stricter signature matching
    'missing_text_section': 25,
    'rwx_section': 20,
    'low_imports': 15,
    'high_entropy_section': 15,
    'oep_not_in_text': 20,
    'suspicious_section_name': 10
}
```

### Custom Entropy Thresholds
Adjust the entropy threshold in `check_entropy()`:

```python
# More strict (fewer false positives)
high_entropy_threshold = 7.5

# More lenient (better detection of weak compression)
high_entropy_threshold = 6.5
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'lief'"
**Solution**: Install LIEF library
```bash
pip install lief
```

### Issue: "[ERROR] Failed to parse PE file: module 'lief._lief.PE' has no attribute 'MACHINE_TYPES'"
**Solution**: This is a LIEF API compatibility issue. The detector has been updated to handle both old and new LIEF versions automatically.

**Quick fix**:
1. Update LIEF to the latest version:
```bash
pip install --upgrade lief
```

2. Or reinstall LIEF:
```bash
pip uninstall lief
pip install lief
```

3. Check your LIEF installation:
```bash
python utils/check_lief_version.py
```

The updated detector now uses:
- **Modern LIEF** (0.12+): Enum-based constants (`lief.PE.MACHINE_TYPES.AMD64`)
- **Legacy LIEF** (0.9-0.11): Raw integer values (`0x8664` for AMD64)
- **Fallback**: String matching for enum-like objects

### Issue: File not recognized as PE
**Solution**: Verify the file is a valid PE executable
```bash
# Use file command (Linux/Mac)
file sample.exe

# Or check file signature (should start with "MZ")
xxd -l 2 sample.exe
```

### Issue: High false positive rate
**Solution**: Increase confidence threshold
```python
# Only consider high-confidence detections
result = detector.detect()
if result['confidence_score'] >= 70:  # Adjust threshold
    print("Packed with high confidence")
```

## Exit Codes

- `0`: File is packed
- `1`: File is not packed (or error occurred)

Useful for scripting:
```bash
python utils/packer_detector.py sample.exe
if [ $? -eq 0 ]; then
    echo "File is packed"
else
    echo "File is not packed"
fi
```

## License

Part of NKREPO - NKAMG Malware Analysis System
For research and educational purposes only.
