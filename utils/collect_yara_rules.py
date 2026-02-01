# -*- coding: utf-8 -*-
"""
YARA Rules Collector for PE Packer Detection
Downloads and organizes YARA rules from popular GitHub repositories.

Author: NKAMG (Nankai Anti-Malware Group)
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import urllib.request
import urllib.error


class YaraRulesCollector:
    """
    Collector for downloading YARA rules from GitHub repositories.

    Focuses on packer, crypter, and protector detection rules.
    """

    # Popular GitHub repositories with packer YARA rules
    REPOSITORIES = {
        'Yara-Rules': {
            'url': 'https://raw.githubusercontent.com/Yara-Rules/rules/master',
            'categories': [
                'Packers/packer.yar',
                'Packers/pespin.yar',
                'Packers/upx.yar',
                'Packers/vmprotect.yar',
                'Packers/obsidium.yar',
                'Packers/themida.yar',
                'Packers/armadillo.yar',
                'Packers/aspack.yar',
                'Packers/pecompact.yar',
                'Packers/fsg.yar',
                'Packers/nspack.yar',
                'Packers/mpress.yar',
                'Packers/eziriz.yar',
                'Packers/enigma.yar',
                'Packers/petite.yar',
                'Packers/rlpack.yar',
                'Packers/exe32pack.yar',
                'Packers/kkrunchy.yar',
                'Packers/yoda.yar'
            ]
        },
        'Bartblaze': {
            'url': 'https://raw.githubusercontent.com/bartblaze/Yara-rules/master',
            'categories': [
                'Packers/packer.yar',
                'Packers/Themida.yar'
            ]
        },
        'InQuest': {
            'url': 'https://raw.githubusercontent.com/InQuest/yara-rules/master',
            'categories': [
                'Packers.yar'
            ]
        },
        'Elastic': {
            'url': 'https://raw.githubusercontent.com/elastic/protections-artifacts/main/yara/rules',
            'categories': [
                'Windows_Trojan_Generic.yar'
            ]
        }
    }

    # Additional individual rule URLs
    INDIVIDUAL_RULES = [
        {
            'name': 'generic_packers.yar',
            'url': 'https://gist.githubusercontent.com/andresriancho/f4146b64aa88b4f4e1ea/raw/packer.yar'
        }
    ]

    def __init__(self, output_dir: str = "yara_rules", verbose: bool = False):
        """
        Initialize the collector.

        Args:
            output_dir: Directory to store downloaded rules
            verbose: Enable verbose output
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.stats = {
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'total_size': 0
        }

        # Create directory structure
        self.packers_dir = self.output_dir / 'packers'
        self.crypters_dir = self.output_dir / 'crypters'
        self.index_file = self.output_dir / 'rules_index.json'

    def create_directories(self):
        """Create directory structure for YARA rules."""
        self.output_dir.mkdir(exist_ok=True)
        self.packers_dir.mkdir(exist_ok=True)
        self.crypters_dir.mkdir(exist_ok=True)

        print(f"[*] Created directory structure at: {self.output_dir}")
        print(f"    - Packers: {self.packers_dir}")
        print(f"    - Crypters: {self.crypters_dir}")

    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a file from URL.

        Args:
            url: URL to download from
            output_path: Path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.verbose:
                print(f"[*] Downloading: {url}")

            # Set user agent to avoid blocking
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (YARA Rules Collector)'}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()

                # Save file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(content)

                self.stats['downloaded'] += 1
                self.stats['total_size'] += len(content)

                if self.verbose:
                    print(f"[+] Saved: {output_path.name} ({len(content)} bytes)")

                return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                if self.verbose:
                    print(f"[-] Not found (404): {url}")
                self.stats['skipped'] += 1
            else:
                print(f"[!] HTTP Error {e.code}: {url}")
                self.stats['failed'] += 1
            return False

        except urllib.error.URLError as e:
            print(f"[!] URL Error: {url} - {e.reason}")
            self.stats['failed'] += 1
            return False

        except Exception as e:
            print(f"[!] Error downloading {url}: {e}")
            self.stats['failed'] += 1
            return False

    def collect_from_repositories(self):
        """Download YARA rules from configured repositories."""
        print(f"\n[*] Collecting YARA rules from {len(self.REPOSITORIES)} repositories...")

        for repo_name, repo_info in self.REPOSITORIES.items():
            print(f"\n[*] Repository: {repo_name}")

            base_url = repo_info['url']
            categories = repo_info['categories']

            for category in categories:
                url = f"{base_url}/{category}"
                filename = Path(category).name

                # Determine output directory
                if 'Packer' in category or 'packer' in filename.lower():
                    output_path = self.packers_dir / f"{repo_name.lower()}_{filename}"
                else:
                    output_path = self.output_dir / f"{repo_name.lower()}_{filename}"

                self.download_file(url, output_path)

    def collect_individual_rules(self):
        """Download individual YARA rules from direct URLs."""
        if not self.INDIVIDUAL_RULES:
            return

        print(f"\n[*] Collecting {len(self.INDIVIDUAL_RULES)} individual rules...")

        for rule_info in self.INDIVIDUAL_RULES:
            name = rule_info['name']
            url = rule_info['url']

            output_path = self.packers_dir / name
            self.download_file(url, output_path)

    def create_combined_rules(self):
        """Create combined YARA rule files for easy use."""
        print(f"\n[*] Creating combined rule files...")

        # Combine all packer rules
        combined_packers = self.output_dir / 'combined_packers.yar'
        combined_content = []

        combined_content.append('/*\n')
        combined_content.append(' * Combined YARA Rules for Packer Detection\n')
        combined_content.append(f' * Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        combined_content.append(' * Source: Multiple GitHub repositories\n')
        combined_content.append(' */\n\n')

        rule_count = 0
        for rule_file in self.packers_dir.glob('*.yar'):
            try:
                content = rule_file.read_text(encoding='utf-8', errors='ignore')

                # Add source comment
                combined_content.append(f'\n// Source: {rule_file.name}\n')
                combined_content.append(content)
                combined_content.append('\n')

                rule_count += 1

            except Exception as e:
                if self.verbose:
                    print(f"[!] Error reading {rule_file.name}: {e}")

        if rule_count > 0:
            combined_packers.write_text(''.join(combined_content), encoding='utf-8')
            print(f"[+] Created combined packer rules: {combined_packers.name} ({rule_count} files)")

    def create_index(self):
        """Create an index of all downloaded rules."""
        print(f"\n[*] Creating rules index...")

        index = {
            'generated': datetime.now().isoformat(),
            'statistics': self.stats,
            'categories': {}
        }

        # Index packers
        packer_rules = []
        for rule_file in self.packers_dir.glob('*.yar'):
            packer_rules.append({
                'name': rule_file.name,
                'path': str(rule_file.relative_to(self.output_dir)),
                'size': rule_file.stat().st_size,
                'category': 'packer'
            })

        index['categories']['packers'] = {
            'count': len(packer_rules),
            'rules': packer_rules
        }

        # Index crypters
        crypter_rules = []
        for rule_file in self.crypters_dir.glob('*.yar'):
            crypter_rules.append({
                'name': rule_file.name,
                'path': str(rule_file.relative_to(self.output_dir)),
                'size': rule_file.stat().st_size,
                'category': 'crypter'
            })

        index['categories']['crypters'] = {
            'count': len(crypter_rules),
            'rules': crypter_rules
        }

        # Save index
        self.index_file.write_text(json.dumps(index, indent=2), encoding='utf-8')
        print(f"[+] Index created: {self.index_file}")

    def create_readme(self):
        """Create README with usage instructions."""
        readme_path = self.output_dir / 'README.md'

        readme_content = f"""# YARA Rules for Packer Detection

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source:** Multiple GitHub repositories

## Statistics

- **Downloaded:** {self.stats['downloaded']} files
- **Failed:** {self.stats['failed']} files
- **Total Size:** {self.stats['total_size']:,} bytes

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
    print(f"Detected: {{match.rule}}")
    print(f"Tags: {{match.tags}}")
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
        print(f"LIEF detected: {{lief_result['packer_type']}}")
    if yara_matches:
        print(f"YARA detected: {{[m.rule for m in yara_matches]}}")
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
"""

        readme_path.write_text(readme_content, encoding='utf-8')
        print(f"[+] README created: {readme_path}")

    def print_summary(self):
        """Print collection summary."""
        print("\n" + "="*70)
        print("YARA RULES COLLECTION SUMMARY")
        print("="*70)
        print(f"\nDownloaded:    {self.stats['downloaded']} files")
        print(f"Failed:        {self.stats['failed']} files")
        print(f"Skipped:       {self.stats['skipped']} files")
        print(f"Total Size:    {self.stats['total_size']:,} bytes")
        print(f"\nOutput Directory: {self.output_dir.absolute()}")
        print("="*70 + "\n")

    def collect_all(self):
        """Main collection method."""
        print("="*70)
        print("YARA Rules Collector for PE Packer Detection")
        print("="*70)

        # Create directories
        self.create_directories()

        # Download rules
        self.collect_from_repositories()
        self.collect_individual_rules()

        # Create combined files
        self.create_combined_rules()

        # Create index and documentation
        self.create_index()
        self.create_readme()

        # Print summary
        self.print_summary()


def create_custom_packer_rules(output_dir: Path):
    """
    Create custom packer detection rules based on our detector.

    Args:
        output_dir: Directory to save custom rules
    """
    custom_rules_dir = output_dir / 'custom'
    custom_rules_dir.mkdir(exist_ok=True)

    # UPX detection rule
    upx_rule = """/*
    YARA Rule for UPX Packer Detection
    Author: NKAMG
    Description: Detects UPX packed executables
*/

rule UPX_Packer : packer {
    meta:
        description = "Detects UPX packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        reference = "https://upx.github.io/"

    strings:
        $upx1 = "UPX0" fullword
        $upx2 = "UPX1" fullword
        $upx3 = "UPX!" fullword
        $upx_sig = { 55 50 58 21 } // UPX! signature

    condition:
        uint16(0) == 0x5A4D and // MZ header
        (any of ($upx*))
}
"""

    # VMProtect detection rule
    vmp_rule = """/*
    YARA Rule for VMProtect Detection
    Author: NKAMG
    Description: Detects VMProtect packed executables
*/

rule VMProtect_Packer : packer {
    meta:
        description = "Detects VMProtect packed executables"
        author = "NKAMG"
        date = "2026-02-01"

    strings:
        $vmp1 = ".vmp0" fullword
        $vmp2 = ".vmp1" fullword
        $vmp3 = ".vmp2" fullword
        $vmprot = ".VMPROT" fullword

    condition:
        uint16(0) == 0x5A4D and
        any of them
}
"""

    # Themida detection rule
    themida_rule = """/*
    YARA Rule for Themida/WinLicense Detection
    Author: NKAMG
    Description: Detects Themida/WinLicense packed executables
*/

rule Themida_Packer : packer {
    meta:
        description = "Detects Themida/WinLicense packed executables"
        author = "NKAMG"
        date = "2026-02-01"

    strings:
        $themida1 = ".themida" fullword
        $themida2 = "Themida" fullword
        $winlicense = ".winlice" fullword
        $oreans = "Oreans" nocase

    condition:
        uint16(0) == 0x5A4D and
        any of them
}
"""

    # High entropy section rule
    high_entropy_rule = """/*
    YARA Rule for High Entropy Sections
    Author: NKAMG
    Description: Detects PE files with high entropy sections (possible packing)
*/

import "pe"
import "math"

rule High_Entropy_Section : suspicious {
    meta:
        description = "Detects PE files with high entropy sections"
        author = "NKAMG"
        date = "2026-02-01"

    condition:
        pe.is_pe and
        for any section in pe.sections : (
            math.entropy(section.raw_data_offset, section.raw_data_size) > 7.0
        )
}
"""

    # Generic packer rule
    generic_rule = """/*
    YARA Rule for Generic Packer Detection
    Author: NKAMG
    Description: Detects common packer indicators
*/

import "pe"

rule Generic_Packer_Indicators : packer {
    meta:
        description = "Detects generic packer indicators"
        author = "NKAMG"
        date = "2026-02-01"

    condition:
        pe.is_pe and
        (
            // Very few imports (< 5 functions)
            pe.number_of_imports < 5 or

            // Missing .text section
            not for any section in pe.sections : (
                section.name contains ".text"
            ) or

            // Entry point not in .text
            not for any section in pe.sections : (
                section.name contains ".text" and
                pe.entry_point >= section.virtual_address and
                pe.entry_point < (section.virtual_address + section.virtual_size)
            )
        )
}
"""

    # Save custom rules
    (custom_rules_dir / 'upx_custom.yar').write_text(upx_rule, encoding='utf-8')
    (custom_rules_dir / 'vmprotect_custom.yar').write_text(vmp_rule, encoding='utf-8')
    (custom_rules_dir / 'themida_custom.yar').write_text(themida_rule, encoding='utf-8')
    (custom_rules_dir / 'high_entropy.yar').write_text(high_entropy_rule, encoding='utf-8')
    (custom_rules_dir / 'generic_packer.yar').write_text(generic_rule, encoding='utf-8')

    print(f"[+] Created 5 custom YARA rules in: {custom_rules_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='YARA Rules Collector for PE Packer Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect rules to default directory (yara_rules/)
  python collect_yara_rules.py

  # Collect rules to custom directory
  python collect_yara_rules.py -o /path/to/rules

  # Collect with verbose output
  python collect_yara_rules.py -v

  # Update existing collection
  python collect_yara_rules.py --update

  # Only create custom rules
  python collect_yara_rules.py --custom-only
        """
    )

    parser.add_argument('-o', '--output', default='yara_rules',
                        help='Output directory for YARA rules (default: yara_rules/)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--update', action='store_true',
                        help='Update existing rule collection')
    parser.add_argument('--custom-only', action='store_true',
                        help='Only create custom rules (skip downloading)')

    args = parser.parse_args()

    # Create collector
    collector = YaraRulesCollector(output_dir=args.output, verbose=args.verbose)

    if args.custom_only:
        print("[*] Creating custom YARA rules only...")
        collector.create_directories()
        create_custom_packer_rules(collector.output_dir)
        print("[+] Custom rules created successfully!")
        return

    # Collect rules
    collector.collect_all()

    # Create custom rules
    create_custom_packer_rules(collector.output_dir)


if __name__ == '__main__':
    main()
