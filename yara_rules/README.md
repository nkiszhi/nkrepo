# YARA Rules for Packer Detection

**Generated:** 2026-02-01 16:16:59
**Source:** Multiple GitHub repositories

## Statistics

- **Downloaded:** 1 files
- **Failed:** 0 files
- **Total Size:** 14,911 bytes

## Directory Structure

```
yara_rules/
├── packers/              # Individual packer detection rules
├── crypters/             # Crypter detection rules
├── combined_packers.yar  # All packer rules in one file
├── rules_index.json      # Metadata and index
└── README.md             # This file
```

## Usage

### With yara-python

```python
import yara

# Load combined rules
rules = yara.compile('yara_rules/combined_packers.yar')

# Scan a file
matches = rules.match('suspicious.exe')

for match in matches:
    print(f"Detected: {match.rule}")
    print(f"Tags: {match.tags}")
```

### With YARA CLI

```bash
# Scan single file
yara yara_rules/combined_packers.yar suspicious.exe

# Scan directory
yara -r yara_rules/combined_packers.yar /path/to/samples/
```

### Integration with Packer Detector

```python
from utils.packer_detector import PEPackerDetector
import yara

# LIEF-based detection
detector = PEPackerDetector('sample.exe')
lief_result = detector.detect()

# YARA-based detection
rules = yara.compile('yara_rules/combined_packers.yar')
yara_matches = rules.match('sample.exe')

# Combine results
if lief_result['is_packed'] or yara_matches:
    print("File is packed!")
    if lief_result['is_packed']:
        print(f"LIEF detected: {lief_result['packer_type']}")
    if yara_matches:
        print(f"YARA detected: {[m.rule for m in yara_matches]}")
```

## Sources

This collection includes YARA rules from:

- **Yara-Rules/rules** - Comprehensive packer collection
- **bartblaze/Yara-rules** - Quality security rules
- **InQuest/yara-rules** - Threat intelligence rules
- **elastic/protections-artifacts** - Elastic Security rules

## Updating Rules

To update the rules collection:

```bash
python utils/collect_yara_rules.py --update
```

## License

Each rule file maintains its original license from the source repository.
Please check individual files for license information.

## Contributing

To add new rule sources, edit `collect_yara_rules.py` and add to:
- `REPOSITORIES` dictionary
- `INDIVIDUAL_RULES` list

Then run the collector to update.
