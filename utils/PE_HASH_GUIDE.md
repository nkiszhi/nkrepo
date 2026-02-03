# PE File Hash Calculator - Complete Guide

Comprehensive hash calculator for PE files supporting 11+ hash types used in malware analysis.

## 📊 Supported Hash Types

### Cryptographic Hashes (Always Available)
- **MD5** - 128-bit cryptographic hash
- **SHA1** - 160-bit cryptographic hash
- **SHA256** - 256-bit cryptographic hash (most common)
- **SHA512** - 512-bit cryptographic hash

### PE-Specific Hashes
- **Imphash** - Import hash based on imported DLLs/functions
- **Authentihash** - PE authenticode hash (excludes signature)
- **Rich Header Hash** - Hash of Rich PE header (compiler info)
- **vHash** - VirusTotal-style similarity hash
- **PEhash** - Structural hash based on PE features

### Fuzzy Hashes (Similarity Detection)
- **SSDEEP** - Context-triggered piecewise hashing
- **TLSH** - Trend Micro Locality Sensitive Hash

## 🚀 Installation

### Basic Usage (Cryptographic Hashes Only)
No additional dependencies needed - uses Python built-in `hashlib`.

```bash
python utils/pe_hash_calculator.py sample.exe
```

### Full Functionality (All Hash Types)
```bash
# Install all optional dependencies
pip install pefile ssdeep py-tlsh

# Or install individually
pip install pefile      # For imphash, authentihash, Rich header
pip install ssdeep      # For SSDEEP fuzzy hash
pip install py-tlsh     # For TLSH fuzzy hash
```

## 📝 Usage

### Single File Analysis
```bash
# Basic hashes (MD5, SHA1, SHA256, SHA512)
python utils/pe_hash_calculator.py sample.exe

# With verbose output
python utils/pe_hash_calculator.py sample.exe -v
```

**Example Output:**
```
======================================================================
PE FILE HASH ANALYSIS
======================================================================

File: sample.exe
Path: C:\samples\sample.exe
Size: 45,056 bytes

----------------------------------------------------------------------
CRYPTOGRAPHIC HASHES
----------------------------------------------------------------------
MD5:        d41d8cd98f00b204e9800998ecf8427e
SHA1:       da39a3ee5e6b4b0d3255bfef95601890afd80709
SHA256:     e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SHA512:     cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce...

----------------------------------------------------------------------
PE-SPECIFIC HASHES
----------------------------------------------------------------------
Imphash:    f34d5f2d4577ed6d9ceec516c1f5a744
Authentihash: a1b2c3d4e5f6789...
Rich Header: 8a3f9c2b1d4e5f67...
vHash:      012345678901234567890123456789ab
PEhash:     abcdef1234567890abcdef1234567890

----------------------------------------------------------------------
FUZZY HASHES
----------------------------------------------------------------------
SSDEEP:     3:FGQEm+vFa1:FdzvL1
TLSH:       T1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7Q8R9S0T1U2V3W4X5Y6Z7
======================================================================
```

### Batch Processing
```bash
# Scan entire directory
python utils/pe_hash_calculator.py -d C:/samples/

# Recursive scan
python utils/pe_hash_calculator.py -d C:/samples/ -r

# Export to JSON
python utils/pe_hash_calculator.py -d C:/samples/ --export-json hashes.json

# Export to CSV
python utils/pe_hash_calculator.py -d C:/samples/ --export-csv hashes.csv

# Export both formats
python utils/pe_hash_calculator.py -d C:/samples/ --export-all
```

## 🎯 Hash Type Details

### 1. MD5 / SHA1 / SHA256 / SHA512
**Purpose:** File identification
**Use Case:** Quick file identification, duplicate detection
**Example:** `d41d8cd98f00b204e9800998ecf8427e`

**Usage:**
- Search in VirusTotal
- Check against known malware databases
- Identify exact file copies

**Limitations:**
- Changes with any file modification
- Not useful for packed/obfuscated variants

---

### 2. Imphash (Import Hash)
**Purpose:** Identify malware families based on imports
**Use Case:** Group malware samples by functionality
**Example:** `f34d5f2d4577ed6d9ceec516c1f5a744`

**How it works:**
- Extracts imported DLLs and functions
- Normalizes and hashes them
- Same imports = same imphash

**Usage:**
```python
# Find all samples with same imphash
SELECT * FROM samples WHERE imphash = 'f34d5f2d4577ed6d9ceec516c1f5a744'
```

**Benefits:**
- Resilient to packing (if imports are resolved)
- Groups malware families
- Works across different versions

**Limitations:**
- Different if packer changes imports
- Not useful for heavily packed files

---

### 3. Authentihash
**Purpose:** PE authenticode signature verification
**Use Case:** Verify signed binaries, detect tampering
**Example:** `a1b2c3d4e5f6789...`

**How it works:**
- Hashes PE file excluding signature
- Used by Windows code signing
- Verifies file hasn't been modified

**Usage:**
- Verify legitimate software
- Detect tampered signed files
- Check signature validity

---

### 4. Rich Header Hash
**Purpose:** Identify compiler/build environment
**Use Case:** Attribution, compiler identification
**Example:** `8a3f9c2b1d4e5f67...`

**How it works:**
- Hashes the Rich PE header
- Contains compiler version info
- Unique to build environment

**Usage:**
- Identify samples from same build
- Track malware development
- Compiler fingerprinting

**Note:** Not all PE files have Rich headers

---

### 5. vHash (VirusTotal Hash)
**Purpose:** Similarity detection
**Use Case:** Find related malware variants
**Example:** `012345678901234567890123456789ab`

**How it works:**
- Based on PE structure features
- Sections, imports, exports
- Similar files have similar vHash

**Usage:**
- Search VirusTotal for variants
- Cluster similar samples
- Identify malware families

---

### 6. PEhash
**Purpose:** Structural similarity
**Use Case:** Group structurally similar files
**Example:** `abcdef1234567890abcdef1234567890`

**How it works:**
- Hashes PE structural features
- Machine type, sections, timestamps
- Normalized to reduce noise

**Usage:**
- Alternative to vHash
- Less affected by minor changes
- Good for variant detection

---

### 7. SSDEEP (Fuzzy Hash)
**Purpose:** Similarity detection for modified files
**Use Case:** Find modified versions of malware
**Example:** `3:FGQEm+vFa1:FdzvL1`

**How it works:**
- Context-triggered piecewise hashing
- Compares file chunks
- Returns similarity score (0-100)

**Usage:**
```python
import ssdeep

# Compare two files
hash1 = "3:FGQEm+vFa1:FdzvL1"
hash2 = "3:FGQEm+vFa2:FdzvL2"

similarity = ssdeep.compare(hash1, hash2)
print(f"Similarity: {similarity}%")  # Output: 95%
```

**Benefits:**
- Detects similar but modified files
- Works with packed/obfuscated variants
- Gives similarity percentage

**Limitations:**
- Requires minimum file size (~4KB)
- Less precise than exact hashes

---

### 8. TLSH (Trend Micro Locality Sensitive Hash)
**Purpose:** Advanced similarity detection
**Use Case:** Large-scale malware clustering
**Example:** `T1A2B3C4D5E6F7G8...`

**How it works:**
- Locality-sensitive hashing
- More robust than SSDEEP for large files
- Better for clustering

**Usage:**
```python
import tlsh

# Compare two files
hash1 = tlsh.hash(file1_data)
hash2 = tlsh.hash(file2_data)

distance = tlsh.diff(hash1, hash2)
print(f"Distance: {distance}")  # Lower = more similar
```

**Benefits:**
- Better for files >10KB
- More accurate clustering
- Industry standard (Trend Micro)

---

## 💡 Use Case Examples

### Use Case 1: Malware Triage
```bash
# Calculate all hashes
python utils/pe_hash_calculator.py suspicious.exe

# Check SHA256 on VirusTotal
# Check imphash for known families
# Use SSDEEP to find variants
```

### Use Case 2: Duplicate Detection
```bash
# Calculate hashes for repository
python utils/pe_hash_calculator.py -d C:/samples/ --export-csv hashes.csv

# Find duplicates (same SHA256)
import pandas as pd
df = pd.read_csv('hashes.csv')
duplicates = df[df.duplicated(subset=['sha256'], keep=False)]
print(duplicates)
```

### Use Case 3: Family Clustering
```bash
# Export all hashes
python utils/pe_hash_calculator.py -d C:/malware/ --export-json hashes.json

# Cluster by imphash
import json
from collections import defaultdict

with open('hashes.json') as f:
    data = json.load(f)

families = defaultdict(list)
for item in data:
    imphash = item.get('imphash')
    if imphash:
        families[imphash].append(item['file_name'])

# Print families
for imphash, files in families.items():
    if len(files) > 1:
        print(f"\nFamily (imphash={imphash[:8]...): {len(files)} samples")
        for f in files[:5]:
            print(f"  - {f}")
```

### Use Case 4: Similarity Search
```python
import ssdeep

# Calculate SSDEEP for all files
hashes = {}
for file in files:
    calc = PEHashCalculator(file)
    result = calc.calculate_all_hashes()
    hashes[file] = result['ssdeep']

# Find similar files
target_hash = hashes['malware.exe']

for file, file_hash in hashes.items():
    if file != 'malware.exe':
        similarity = ssdeep.compare(target_hash, file_hash)
        if similarity > 80:
            print(f"{file}: {similarity}% similar")
```

### Use Case 5: Build Environment Tracking
```bash
# Extract Rich header hashes
python utils/pe_hash_calculator.py -d C:/malware/ --export-csv hashes.csv

# Group by Rich header
import pandas as pd
df = pd.read_csv('hashes.csv')
builds = df.groupby('rich_header_hash')['file_name'].apply(list)

# Samples from same build environment
for rich_hash, files in builds.items():
    if rich_hash != 'N/A' and len(files) > 1:
        print(f"\nSame build: {len(files)} samples")
        for f in files[:3]:
            print(f"  {f}")
```

## 🔄 Integration Examples

### Integration with Database
```python
import sqlite3
from pe_hash_calculator import PEHashCalculator

conn = sqlite3.connect('malware.db')

# Create table
conn.execute('''
    CREATE TABLE IF NOT EXISTS file_hashes (
        file_path TEXT PRIMARY KEY,
        md5 TEXT,
        sha256 TEXT,
        imphash TEXT,
        ssdeep TEXT,
        tlsh TEXT
    )
''')

# Calculate and store hashes
for sample in samples:
    calc = PEHashCalculator(sample)
    hashes = calc.calculate_all_hashes()

    conn.execute('''
        INSERT OR REPLACE INTO file_hashes
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        sample,
        hashes['md5'],
        hashes['sha256'],
        hashes['imphash'],
        hashes['ssdeep'],
        hashes['tlsh']
    ))

conn.commit()
```

### Integration with VirusTotal API
```python
import requests
from pe_hash_calculator import PEHashCalculator

VT_API_KEY = 'your_api_key'

calc = PEHashCalculator('sample.exe')
hashes = calc.calculate_all_hashes()

# Check SHA256 on VirusTotal
response = requests.get(
    f'https://www.virustotal.com/api/v3/files/{hashes["sha256"]}',
    headers={'x-apikey': VT_API_KEY}
)

if response.status_code == 200:
    vt_data = response.json()
    print(f"VT Detections: {vt_data['data']['attributes']['last_analysis_stats']}")
else:
    print("File not found on VirusTotal")
```

### Integration with Packer Detector
```python
from pe_hash_calculator import PEHashCalculator
from packer_detector import PEPackerDetector

# Calculate hashes
hash_calc = PEHashCalculator('sample.exe')
hashes = hash_calc.calculate_all_hashes()

# Detect packer
pack_det = PEPackerDetector('sample.exe')
packer_result = pack_det.detect()

# Combine results
analysis = {
    'file': 'sample.exe',
    'hashes': hashes,
    'packer': packer_result['packer_type'],
    'is_packed': packer_result['is_packed'],
    'confidence': packer_result['confidence_score']
}

print(f"File: {analysis['file']}")
print(f"SHA256: {hashes['sha256']}")
print(f"Imphash: {hashes['imphash']}")
print(f"Packer: {packer_result['packer_type']}")
```

## 📊 Performance Benchmarks

| Hash Type | Speed | Memory | Best For |
|-----------|-------|--------|----------|
| MD5/SHA1/SHA256 | Very Fast | Low | File identification |
| Imphash | Fast | Low | Family clustering |
| Authentihash | Fast | Low | Signature verification |
| Rich Header | Fast | Low | Build tracking |
| SSDEEP | Medium | Medium | Similarity (small files) |
| TLSH | Medium | Medium | Similarity (large files) |
| vHash/PEhash | Fast | Low | Variant detection |

**Batch Processing Speed** (1000 files):
- Cryptographic only: ~30 seconds
- All hash types: ~2-3 minutes
- SSDEEP/TLSH only: ~1-2 minutes

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pefile'"
```bash
pip install pefile
```

### Issue: "ModuleNotFoundError: No module named 'ssdeep'"
**Windows:**
```bash
# May need Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
pip install ssdeep
```

**Linux:**
```bash
sudo apt-get install libfuzzy-dev
pip install ssdeep
```

### Issue: "ModuleNotFoundError: No module named 'tlsh'"
```bash
pip install py-tlsh
```

### Issue: Imphash returns None
**Reason:** File has no import table (packed or corrupted)
**Solution:** This is expected for heavily packed files

### Issue: SSDEEP returns None
**Reason:** File too small (< 4KB required)
**Solution:** Use TLSH or cryptographic hashes instead

### Issue: Rich Header Hash returns None
**Reason:** Not all PE files have Rich headers
**Solution:** This is normal - Rich header is optional

## 📚 Best Practices

1. **Always calculate SHA256** - Industry standard identifier
2. **Use imphash for clustering** - Groups malware families
3. **Use SSDEEP for variants** - Detects similar files
4. **Combine multiple hashes** - No single hash is perfect
5. **Store all hashes in database** - Future analysis needs
6. **Export results regularly** - Backup your analysis
7. **Update dependencies** - New hash algorithms emerge

## 🎓 Advanced Usage

### Custom Hash Calculation
```python
from pe_hash_calculator import PEHashCalculator

class CustomHashCalculator(PEHashCalculator):
    def calculate_custom_hash(self):
        """Your custom hash algorithm"""
        # Implement your logic
        pass

calc = CustomHashCalculator('sample.exe')
custom = calc.calculate_custom_hash()
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor
from pe_hash_calculator import PEHashCalculator

def calc_hashes(file_path):
    calc = PEHashCalculator(file_path)
    return calc.calculate_all_hashes()

files = list(Path('samples/').glob('*.exe'))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(calc_hashes, files))

print(f"Processed {len(results)} files")
```

## 📖 References

- **Imphash:** https://www.mandiant.com/resources/blog/tracking-malware-import-hashing
- **SSDEEP:** https://ssdeep-project.github.io/ssdeep/
- **TLSH:** https://github.com/trendmicro/tlsh
- **Authentihash:** https://docs.microsoft.com/en-us/windows-hardware/drivers/install/authenticode
- **Rich Header:** http://bytepointer.com/articles/the_microsoft_rich_header.htm

---

**Ready to calculate hashes!**

```bash
python utils/pe_hash_calculator.py sample.exe --export-all
```
