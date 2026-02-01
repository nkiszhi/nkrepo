# -*- coding: utf-8 -*-
"""
YARA-based Packer Scanner
Scans PE files using YARA rules and integrates with LIEF-based detector.

Author: NKAMG (Nankai Anti-Malware Group)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

try:
    import yara
except ImportError:
    print("[ERROR] yara-python is not installed")
    print("Install with: pip install yara-python")
    sys.exit(1)

try:
    from packer_detector import PEPackerDetector
except ImportError:
    print("[WARNING] Cannot import packer_detector - LIEF integration disabled")
    PEPackerDetector = None


class YaraPackerScanner:
    """
    Scanner that uses YARA rules to detect packers in PE files.

    Features:
    - Load rules from files or directories
    - Scan single files or directories
    - Combine YARA detection with LIEF-based detection
    - Export results to JSON/CSV
    """

    def __init__(self, rules_path: str, verbose: bool = False):
        """
        Initialize the scanner.

        Args:
            rules_path: Path to YARA rules file or directory
            verbose: Enable verbose output
        """
        self.rules_path = Path(rules_path)
        self.verbose = verbose
        self.rules = None
        self.scan_results = []

    def load_rules(self) -> bool:
        """
        Load YARA rules from file or directory.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.rules_path.is_file():
                # Load single file
                if self.verbose:
                    print(f"[*] Loading YARA rules from: {self.rules_path}")

                self.rules = yara.compile(filepath=str(self.rules_path))
                print(f"[+] YARA rules loaded successfully")
                return True

            elif self.rules_path.is_dir():
                # Load all .yar files from directory
                if self.verbose:
                    print(f"[*] Loading YARA rules from directory: {self.rules_path}")

                rule_files = {}
                for yar_file in self.rules_path.rglob('*.yar'):
                    # Use relative path as namespace
                    namespace = str(yar_file.relative_to(self.rules_path)).replace('\\', '/').replace('.yar', '')
                    rule_files[namespace] = str(yar_file)

                if not rule_files:
                    print(f"[!] No .yar files found in {self.rules_path}")
                    return False

                if self.verbose:
                    print(f"[*] Found {len(rule_files)} rule files")

                self.rules = yara.compile(filepaths=rule_files)
                print(f"[+] Loaded {len(rule_files)} YARA rule files")
                return True

            else:
                print(f"[!] Invalid rules path: {self.rules_path}")
                return False

        except yara.SyntaxError as e:
            print(f"[!] YARA syntax error: {e}")
            return False
        except Exception as e:
            print(f"[!] Error loading YARA rules: {e}")
            return False

    def scan_file(self, file_path: str, use_lief: bool = True) -> Dict:
        """
        Scan a file using YARA rules and optionally LIEF detector.

        Args:
            file_path: Path to file to scan
            use_lief: Whether to also use LIEF-based detection

        Returns:
            Scan result dictionary
        """
        result = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'yara_matches': [],
            'lief_detection': None,
            'is_packed': False,
            'packers_detected': []
        }

        try:
            # YARA scan
            matches = self.rules.match(file_path)

            if matches:
                result['yara_matches'] = [
                    {
                        'rule': match.rule,
                        'namespace': match.namespace,
                        'tags': list(match.tags),
                        'meta': match.meta
                    }
                    for match in matches
                ]

                # Extract packer names from matches
                for match in matches:
                    result['packers_detected'].append(match.rule)

                result['is_packed'] = True

            # LIEF scan (if available and enabled)
            if use_lief and PEPackerDetector:
                try:
                    detector = PEPackerDetector(file_path)
                    lief_result = detector.detect()

                    result['lief_detection'] = {
                        'is_packed': lief_result['is_packed'],
                        'packer_type': lief_result['packer_type'],
                        'confidence_score': lief_result['confidence_score'],
                        'suspicious_features': lief_result['suspicious_features']
                    }

                    # Combine detections
                    if lief_result['is_packed']:
                        result['is_packed'] = True
                        if lief_result['packer_type'] not in result['packers_detected']:
                            result['packers_detected'].append(lief_result['packer_type'])

                except Exception as e:
                    if self.verbose:
                        print(f"[!] LIEF detection failed for {Path(file_path).name}: {e}")

        except Exception as e:
            result['error'] = str(e)
            if self.verbose:
                print(f"[!] Error scanning {file_path}: {e}")

        return result

    def scan_directory(self, directory: str, recursive: bool = True,
                      use_lief: bool = True) -> List[Dict]:
        """
        Scan all PE files in a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            use_lief: Whether to use LIEF-based detection

        Returns:
            List of scan results
        """
        results = []
        dir_path = Path(directory)

        print(f"[*] Scanning directory: {directory}")

        # Find PE files
        if recursive:
            pe_files = list(dir_path.rglob('*.exe')) + list(dir_path.rglob('*.dll'))
        else:
            pe_files = list(dir_path.glob('*.exe')) + list(dir_path.glob('*.dll'))

        print(f"[*] Found {len(pe_files)} PE files")

        # Scan each file
        for i, pe_file in enumerate(pe_files, 1):
            if self.verbose:
                print(f"[{i}/{len(pe_files)}] Scanning: {pe_file.name}")

            result = self.scan_file(str(pe_file), use_lief=use_lief)
            results.append(result)

            # Show progress
            if not self.verbose and i % 10 == 0:
                print(f"[*] Progress: {i}/{len(pe_files)} files scanned")

        self.scan_results = results
        return results

    def print_results(self, results: List[Dict], packed_only: bool = False):
        """
        Print scan results to console.

        Args:
            results: List of scan results
            packed_only: Only show packed files
        """
        if packed_only:
            results = [r for r in results if r.get('is_packed', False)]

        if not results:
            print("\n[*] No results to display")
            return

        print("\n" + "="*70)
        print("YARA SCAN RESULTS")
        print("="*70)

        for result in results:
            print(f"\nFile: {result['file_name']}")
            print(f"Path: {result['file_path']}")

            if result.get('error'):
                print(f"  [ERROR] {result['error']}")
                continue

            # YARA matches
            if result['yara_matches']:
                print(f"  YARA Matches: {len(result['yara_matches'])}")
                for match in result['yara_matches']:
                    tags = f" [{', '.join(match['tags'])}]" if match['tags'] else ""
                    print(f"    - {match['rule']}{tags}")

            # LIEF detection
            if result.get('lief_detection'):
                lief = result['lief_detection']
                if lief['is_packed']:
                    print(f"  LIEF Detection: {lief['packer_type']} ({lief['confidence_score']}%)")

            # Combined result
            if result['is_packed']:
                packers = ', '.join(set(result['packers_detected']))
                print(f"  [PACKED] Packers: {packers}")
            else:
                print(f"  [NOT PACKED]")

        print("="*70)

    def generate_statistics(self, results: List[Dict]) -> Dict:
        """
        Generate statistics from scan results.

        Args:
            results: List of scan results

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_files': len(results),
            'packed_files': 0,
            'yara_detections': 0,
            'lief_detections': 0,
            'combined_detections': 0,
            'packer_distribution': defaultdict(int),
            'rule_distribution': defaultdict(int)
        }

        for result in results:
            if result.get('is_packed'):
                stats['packed_files'] += 1

            if result.get('yara_matches'):
                stats['yara_detections'] += 1

                # Count rule matches
                for match in result['yara_matches']:
                    stats['rule_distribution'][match['rule']] += 1

            if result.get('lief_detection', {}).get('is_packed'):
                stats['lief_detections'] += 1

            if result.get('yara_matches') and result.get('lief_detection', {}).get('is_packed'):
                stats['combined_detections'] += 1

            # Count packer types
            for packer in result.get('packers_detected', []):
                stats['packer_distribution'][packer] += 1

        return dict(stats)

    def print_statistics(self, stats: Dict):
        """
        Print statistics to console.

        Args:
            stats: Statistics dictionary
        """
        print("\n" + "="*70)
        print("SCAN STATISTICS")
        print("="*70)

        print(f"\nTotal Files:           {stats['total_files']}")
        print(f"Packed Files:          {stats['packed_files']}")
        print(f"YARA Detections:       {stats['yara_detections']}")
        print(f"LIEF Detections:       {stats['lief_detections']}")
        print(f"Combined Detections:   {stats['combined_detections']}")

        # Packer distribution
        if stats['packer_distribution']:
            print(f"\n{'-'*70}")
            print("PACKER DISTRIBUTION")
            print(f"{'-'*70}")

            sorted_packers = sorted(
                stats['packer_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            for packer, count in sorted_packers[:10]:  # Top 10
                print(f"  {packer:<40} {count:>5}")

        # Rule distribution
        if stats['rule_distribution']:
            print(f"\n{'-'*70}")
            print("TOP 10 TRIGGERED RULES")
            print(f"{'-'*70}")

            sorted_rules = sorted(
                stats['rule_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            for rule, count in sorted_rules[:10]:
                print(f"  {rule:<40} {count:>5}")

        print("="*70)

    def export_json(self, results: List[Dict], output_path: str):
        """
        Export results to JSON.

        Args:
            results: List of scan results
            output_path: Output file path
        """
        export_data = {
            'scan_type': 'yara_packer_scan',
            'rules_path': str(self.rules_path),
            'statistics': self.generate_statistics(results),
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"[+] Results exported to JSON: {output_path}")

    def export_csv(self, results: List[Dict], output_path: str):
        """
        Export results to CSV.

        Args:
            results: List of scan results
            output_path: Output file path
        """
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'File Name', 'File Path', 'Is Packed', 'Packers Detected',
                'YARA Matches', 'LIEF Detection', 'LIEF Confidence'
            ])

            # Data
            for result in results:
                yara_rules = ', '.join([m['rule'] for m in result.get('yara_matches', [])])
                packers = ', '.join(result.get('packers_detected', []))

                lief_detection = ''
                lief_confidence = ''
                if result.get('lief_detection'):
                    lief = result['lief_detection']
                    lief_detection = lief.get('packer_type', '')
                    lief_confidence = lief.get('confidence_score', '')

                writer.writerow([
                    result['file_name'],
                    result['file_path'],
                    'Yes' if result.get('is_packed') else 'No',
                    packers,
                    yara_rules,
                    lief_detection,
                    lief_confidence
                ])

        print(f"[+] Results exported to CSV: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='YARA-based Packer Scanner with LIEF Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan single file with combined rules
  python yara_packer_scanner.py -r yara_rules/combined_packers.yar -f sample.exe

  # Scan directory using all rules
  python yara_packer_scanner.py -r yara_rules/ -d C:/samples

  # Scan with YARA only (no LIEF)
  python yara_packer_scanner.py -r yara_rules/combined_packers.yar -d C:/samples --yara-only

  # Scan and export results
  python yara_packer_scanner.py -r yara_rules/ -d C:/samples --export-json results.json

  # Show only packed files
  python yara_packer_scanner.py -r yara_rules/ -d C:/samples --packed-only
        """
    )

    parser.add_argument('-r', '--rules', required=True,
                        help='Path to YARA rules file or directory')
    parser.add_argument('-f', '--file',
                        help='Scan single file')
    parser.add_argument('-d', '--directory',
                        help='Scan directory')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Scan directory recursively (default: True)')
    parser.add_argument('--yara-only', action='store_true',
                        help='Use YARA only (disable LIEF detection)')
    parser.add_argument('--packed-only', action='store_true',
                        help='Show only packed files')
    parser.add_argument('--export-json',
                        help='Export results to JSON file')
    parser.add_argument('--export-csv',
                        help='Export results to CSV file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if not args.file and not args.directory:
        print("[ERROR] Either --file or --directory must be specified")
        parser.print_help()
        sys.exit(1)

    # Create scanner
    scanner = YaraPackerScanner(rules_path=args.rules, verbose=args.verbose)

    # Load rules
    if not scanner.load_rules():
        sys.exit(1)

    # Scan
    use_lief = not args.yara_only

    if args.file:
        print(f"\n[*] Scanning file: {args.file}")
        result = scanner.scan_file(args.file, use_lief=use_lief)
        results = [result]
    else:
        results = scanner.scan_directory(
            args.directory,
            recursive=args.recursive,
            use_lief=use_lief
        )

    # Print results
    scanner.print_results(results, packed_only=args.packed_only)

    # Statistics
    stats = scanner.generate_statistics(results)
    scanner.print_statistics(stats)

    # Export
    if args.export_json:
        scanner.export_json(results, args.export_json)

    if args.export_csv:
        scanner.export_csv(results, args.export_csv)


if __name__ == '__main__':
    main()
