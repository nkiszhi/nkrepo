# -*- coding: utf-8 -*-
"""
Windows PE Packer Scanner
Scans Windows system for PE files and detects packers.

Author: NKAMG (Nankai Anti-Malware Group)
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from collections import defaultdict

try:
    from packer_detector import PEPackerDetector
except ImportError:
    print("[ERROR] Cannot import packer_detector module")
    print("Make sure packer_detector.py is in the same directory or in PYTHONPATH")
    sys.exit(1)


class WindowsPackerScanner:
    """
    Scanner for detecting packers in Windows PE files.

    Features:
    - Multi-threaded scanning
    - Progress tracking
    - Multiple output formats (JSON, CSV, HTML, console)
    - Filtering by confidence, packer type, etc.
    """

    # Common Windows PE file extensions
    PE_EXTENSIONS = ('.exe', '.dll', '.sys', '.ocx', '.scr', '.cpl', '.drv')

    # Preset scan locations
    SCAN_PRESETS = {
        'system': [
            'C:/Windows/System32',
            'C:/Windows/SysWOW64'
        ],
        'programs': [
            'C:/Program Files',
            'C:/Program Files (x86)'
        ],
        'downloads': [
            os.path.expanduser('~/Downloads'),
            os.path.expanduser('~/Desktop')
        ],
        'quick': [
            'C:/Windows/System32',
            os.path.expanduser('~/Downloads')
        ],
        'full': [
            'C:/Windows',
            'C:/Program Files',
            'C:/Program Files (x86)',
            os.path.expanduser('~')
        ]
    }

    def __init__(self, max_workers: int = 4, verbose: bool = False):
        """
        Initialize the scanner.

        Args:
            max_workers: Number of parallel threads for scanning
            verbose: Enable verbose output
        """
        self.max_workers = max_workers
        self.verbose = verbose
        self.results: List[Dict] = []
        self.stats = {
            'total_scanned': 0,
            'packed_found': 0,
            'errors': 0,
            'skipped': 0,
            'scan_time': 0
        }

    def find_pe_files(self, paths: List[str], recursive: bool = True,
                      max_files: Optional[int] = None) -> List[str]:
        """
        Find all PE files in the given paths.

        Args:
            paths: List of directories to search
            recursive: Whether to search recursively
            max_files: Maximum number of files to find (None = unlimited)

        Returns:
            List of PE file paths
        """
        pe_files = []
        files_found = 0

        print(f"[*] Searching for PE files in {len(paths)} location(s)...")

        for search_path in paths:
            path = Path(search_path)

            if not path.exists():
                print(f"[!] Path does not exist: {search_path}")
                continue

            if path.is_file():
                # Single file
                if path.suffix.lower() in self.PE_EXTENSIONS:
                    pe_files.append(str(path))
                    files_found += 1
                continue

            # Directory search
            try:
                if recursive:
                    pattern = '**/*'
                else:
                    pattern = '*'

                for file_path in path.glob(pattern):
                    if max_files and files_found >= max_files:
                        print(f"[*] Reached maximum file limit ({max_files})")
                        return pe_files

                    if file_path.is_file() and file_path.suffix.lower() in self.PE_EXTENSIONS:
                        pe_files.append(str(file_path))
                        files_found += 1

                        if self.verbose and files_found % 100 == 0:
                            print(f"[*] Found {files_found} PE files so far...")

            except PermissionError:
                if self.verbose:
                    print(f"[!] Permission denied: {search_path}")
                self.stats['skipped'] += 1
            except Exception as e:
                if self.verbose:
                    print(f"[!] Error searching {search_path}: {e}")
                self.stats['errors'] += 1

        print(f"[*] Found {len(pe_files)} PE files")
        return pe_files

    def scan_file(self, file_path: str) -> Optional[Dict]:
        """
        Scan a single PE file for packer detection.

        Args:
            file_path: Path to PE file

        Returns:
            Detection result dictionary or None on error
        """
        try:
            detector = PEPackerDetector(file_path)
            result = detector.detect()

            # Add additional metadata
            result['file_path'] = file_path
            result['scan_time'] = datetime.now().isoformat()

            return result

        except PermissionError:
            if self.verbose:
                print(f"[!] Permission denied: {file_path}")
            self.stats['skipped'] += 1
            return None
        except Exception as e:
            if self.verbose:
                print(f"[!] Error scanning {file_path}: {e}")
            self.stats['errors'] += 1
            return None

    def scan_files(self, file_paths: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Scan multiple PE files using thread pool.

        Args:
            file_paths: List of file paths to scan
            show_progress: Show progress during scanning

        Returns:
            List of scan results
        """
        results = []
        total_files = len(file_paths)

        print(f"\n[*] Scanning {total_files} PE files using {self.max_workers} threads...")
        print(f"[*] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.scan_file, file_path): file_path
                for file_path in file_paths
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                completed += 1
                result = future.result()

                if result:
                    results.append(result)
                    self.stats['total_scanned'] += 1

                    if result['is_packed']:
                        self.stats['packed_found'] += 1

                # Show progress
                if show_progress:
                    progress = (completed / total_files) * 100
                    packed_count = self.stats['packed_found']

                    # Simple progress bar
                    bar_length = 40
                    filled = int(bar_length * completed / total_files)
                    bar = '=' * filled + '-' * (bar_length - filled)

                    print(f"\r[{bar}] {progress:.1f}% | "
                          f"Scanned: {completed}/{total_files} | "
                          f"Packed: {packed_count}", end='')

        print()  # New line after progress bar

        self.stats['scan_time'] = time.time() - start_time
        self.results = results

        print(f"[*] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[*] Scan duration: {self.stats['scan_time']:.2f} seconds")

        return results

    def filter_results(self, results: List[Dict],
                      min_confidence: int = 0,
                      packed_only: bool = False,
                      packer_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter scan results based on criteria.

        Args:
            results: List of scan results
            min_confidence: Minimum confidence score (0-100)
            packed_only: Only return packed files
            packer_types: List of packer types to include (e.g., ['UPX', 'VMProtect'])

        Returns:
            Filtered results
        """
        filtered = results

        # Filter by packed status
        if packed_only:
            filtered = [r for r in filtered if r.get('is_packed', False)]

        # Filter by confidence
        if min_confidence > 0:
            filtered = [r for r in filtered if r.get('confidence_score', 0) >= min_confidence]

        # Filter by packer type
        if packer_types:
            packer_types_lower = [p.lower() for p in packer_types]
            filtered = [
                r for r in filtered
                if r.get('packer_type', '').lower() in packer_types_lower
            ]

        return filtered

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
            'packed_files': sum(1 for r in results if r.get('is_packed', False)),
            'unpacked_files': sum(1 for r in results if not r.get('is_packed', False)),
            'packer_distribution': defaultdict(int),
            'confidence_distribution': {
                'high (70-100%)': 0,
                'medium (40-69%)': 0,
                'low (1-39%)': 0,
                'none (0%)': 0
            },
            'architecture_distribution': defaultdict(int),
            'avg_confidence': 0,
            'scan_time': self.stats['scan_time']
        }

        total_confidence = 0

        for result in results:
            # Packer distribution
            packer = result.get('packer_type', 'Unknown')
            stats['packer_distribution'][packer] += 1

            # Confidence distribution
            confidence = result.get('confidence_score', 0)
            total_confidence += confidence

            if confidence >= 70:
                stats['confidence_distribution']['high (70-100%)'] += 1
            elif confidence >= 40:
                stats['confidence_distribution']['medium (40-69%)'] += 1
            elif confidence > 0:
                stats['confidence_distribution']['low (1-39%)'] += 1
            else:
                stats['confidence_distribution']['none (0%)'] += 1

            # Architecture distribution
            arch = result.get('file_info', {}).get('architecture', 'Unknown')
            stats['architecture_distribution'][arch] += 1

        # Calculate average confidence
        if results:
            stats['avg_confidence'] = total_confidence / len(results)

        return stats

    def print_summary(self, results: List[Dict], stats: Dict):
        """
        Print scan summary to console.

        Args:
            results: List of scan results
            stats: Statistics dictionary
        """
        print("\n" + "="*70)
        print("SCAN SUMMARY")
        print("="*70)

        print(f"\nTotal Files Scanned:  {stats['total_files']}")
        print(f"Packed Files Found:   {stats['packed_files']}")
        print(f"Unpacked Files:       {stats['unpacked_files']}")
        print(f"Errors/Skipped:       {self.stats['errors'] + self.stats['skipped']}")
        print(f"Scan Time:            {stats['scan_time']:.2f} seconds")
        print(f"Average Confidence:   {stats['avg_confidence']:.1f}%")

        # Packer distribution
        if stats['packed_files'] > 0:
            print(f"\n{'-'*70}")
            print("PACKER DISTRIBUTION")
            print(f"{'-'*70}")

            packer_dist = sorted(
                stats['packer_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            for packer, count in packer_dist:
                if packer != 'No Packer':
                    percentage = (count / stats['total_files']) * 100
                    print(f"  {packer:<30} {count:>5} ({percentage:>5.1f}%)")

        # Confidence distribution
        print(f"\n{'-'*70}")
        print("CONFIDENCE DISTRIBUTION")
        print(f"{'-'*70}")
        for level, count in stats['confidence_distribution'].items():
            percentage = (count / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
            print(f"  {level:<30} {count:>5} ({percentage:>5.1f}%)")

        # Architecture distribution
        print(f"\n{'-'*70}")
        print("ARCHITECTURE DISTRIBUTION")
        print(f"{'-'*70}")
        for arch, count in sorted(stats['architecture_distribution'].items()):
            percentage = (count / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
            print(f"  {arch:<30} {count:>5} ({percentage:>5.1f}%)")

        print("="*70 + "\n")

    def print_packed_files(self, results: List[Dict], limit: int = 20):
        """
        Print list of packed files.

        Args:
            results: List of scan results
            limit: Maximum number of files to display
        """
        packed = [r for r in results if r.get('is_packed', False)]

        if not packed:
            print("\n[*] No packed files found.")
            return

        # Sort by confidence (descending)
        packed_sorted = sorted(
            packed,
            key=lambda x: x.get('confidence_score', 0),
            reverse=True
        )

        print(f"\n{'-'*70}")
        print(f"PACKED FILES (Top {min(limit, len(packed))})")
        print(f"{'-'*70}")
        print(f"{'File':<40} {'Packer':<20} {'Conf%':<8}")
        print(f"{'-'*70}")

        for result in packed_sorted[:limit]:
            filename = Path(result['file_path']).name
            if len(filename) > 37:
                filename = filename[:34] + "..."

            packer = result.get('packer_type', 'Unknown')
            if len(packer) > 17:
                packer = packer[:14] + "..."

            confidence = result.get('confidence_score', 0)

            print(f"{filename:<40} {packer:<20} {confidence:<8}")

        if len(packed) > limit:
            print(f"\n[*] ... and {len(packed) - limit} more packed files")
            print(f"[*] Use --export to save full results")

    def export_json(self, results: List[Dict], output_path: str):
        """
        Export results to JSON file.

        Args:
            results: List of scan results
            output_path: Output file path
        """
        export_data = {
            'scan_info': {
                'scan_time': datetime.now().isoformat(),
                'scanner_version': '1.0',
                'total_files': len(results)
            },
            'statistics': self.generate_statistics(results),
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"[*] Results exported to JSON: {output_path}")

    def export_csv(self, results: List[Dict], output_path: str):
        """
        Export results to CSV file.

        Args:
            results: List of scan results
            output_path: Output file path
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'File Path', 'Filename', 'Is Packed', 'Packer Type',
                'Confidence %', 'Architecture', 'Size (bytes)',
                'Sections', 'Imports', 'Suspicious Features'
            ])

            # Data rows
            for result in results:
                file_info = result.get('file_info', {})
                suspicious = '; '.join(result.get('suspicious_features', []))

                writer.writerow([
                    result.get('file_path', ''),
                    file_info.get('filename', ''),
                    'Yes' if result.get('is_packed', False) else 'No',
                    result.get('packer_type', ''),
                    result.get('confidence_score', 0),
                    file_info.get('architecture', ''),
                    file_info.get('size', 0),
                    file_info.get('sections', 0),
                    file_info.get('imports', 0),
                    suspicious
                ])

        print(f"[*] Results exported to CSV: {output_path}")

    def export_html(self, results: List[Dict], output_path: str):
        """
        Export results to HTML report.

        Args:
            results: List of scan results
            output_path: Output file path
        """
        stats = self.generate_statistics(results)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Windows PE Packer Scan Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-box.packed {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-box.unpacked {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .stat-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #34495e; color: white; padding: 12px; text-align: left; font-weight: 600; }}
        td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .packed-yes {{ color: #e74c3c; font-weight: bold; }}
        .packed-no {{ color: #27ae60; }}
        .confidence-high {{ background: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; }}
        .confidence-medium {{ background: #f39c12; color: white; padding: 4px 8px; border-radius: 4px; }}
        .confidence-low {{ background: #3498db; color: white; padding: 4px 8px; border-radius: 4px; }}
        .chart {{ margin: 20px 0; }}
        .bar {{ background: #3498db; height: 25px; margin: 5px 0; border-radius: 4px; display: flex; align-items: center; padding-left: 10px; color: white; }}
        .timestamp {{ color: #7f8c8d; font-size: 12px; margin-top: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Windows PE Packer Scan Report</h1>
        <p><strong>Scan Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Scan Duration:</strong> {stats['scan_time']:.2f} seconds</p>

        <h2>📊 Summary Statistics</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Files Scanned</div>
                <div class="stat-value">{stats['total_files']}</div>
            </div>
            <div class="stat-box packed">
                <div class="stat-label">Packed Files</div>
                <div class="stat-value">{stats['packed_files']}</div>
            </div>
            <div class="stat-box unpacked">
                <div class="stat-label">Unpacked Files</div>
                <div class="stat-value">{stats['unpacked_files']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{stats['avg_confidence']:.1f}%</div>
            </div>
        </div>

        <h2>📦 Packer Distribution</h2>
        <div class="chart">
"""
        # Packer distribution chart
        max_count = max(stats['packer_distribution'].values()) if stats['packer_distribution'] else 1
        for packer, count in sorted(stats['packer_distribution'].items(), key=lambda x: x[1], reverse=True):
            if packer != 'No Packer':
                width = (count / max_count) * 100
                html += f'            <div class="bar" style="width: {width}%">{packer}: {count}</div>\n'

        html += """        </div>

        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>File Name</th>
                    <th>Path</th>
                    <th>Packed</th>
                    <th>Packer Type</th>
                    <th>Confidence</th>
                    <th>Arch</th>
                    <th>Size</th>
                </tr>
            </thead>
            <tbody>
"""

        # Results table (limit to first 1000 for performance)
        for result in results[:1000]:
            file_info = result.get('file_info', {})
            file_path = result.get('file_path', '')
            filename = file_info.get('filename', Path(file_path).name)

            is_packed = result.get('is_packed', False)
            packed_class = 'packed-yes' if is_packed else 'packed-no'
            packed_text = 'Yes' if is_packed else 'No'

            confidence = result.get('confidence_score', 0)
            if confidence >= 70:
                conf_class = 'confidence-high'
            elif confidence >= 40:
                conf_class = 'confidence-medium'
            else:
                conf_class = 'confidence-low'

            html += f"""                <tr>
                    <td>{filename}</td>
                    <td style="font-size: 11px; color: #7f8c8d;">{file_path}</td>
                    <td class="{packed_class}">{packed_text}</td>
                    <td>{result.get('packer_type', '')}</td>
                    <td><span class="{conf_class}">{confidence}%</span></td>
                    <td>{file_info.get('architecture', '')}</td>
                    <td>{file_info.get('size', 0):,}</td>
                </tr>
"""

        if len(results) > 1000:
            html += f"""                <tr>
                    <td colspan="7" style="text-align: center; font-style: italic; color: #7f8c8d;">
                        ... and {len(results) - 1000} more files (see JSON export for complete results)
                    </td>
                </tr>
"""

        html += f"""            </tbody>
        </table>

        <p class="timestamp">Generated by NKAMG Windows PE Packer Scanner v1.0</p>
    </div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"[*] Results exported to HTML: {output_path}")


def main():
    """
    Main entry point for Windows PE Packer Scanner.
    """
    parser = argparse.ArgumentParser(
        description='Windows PE Packer Scanner - Detect packed executables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick scan of system files and downloads
  python windows_packer_scanner.py --preset quick

  # Full system scan (may take a long time!)
  python windows_packer_scanner.py --preset full

  # Scan specific directory
  python windows_packer_scanner.py -d "C:/Users/*/Downloads"

  # Scan and export to all formats
  python windows_packer_scanner.py --preset quick --export-all

  # Scan and show only packed files with high confidence
  python windows_packer_scanner.py -d "C:/samples" --packed-only --min-confidence 70

Scan Presets:
  quick    - System32 and Downloads folders (fast)
  system   - Windows system directories
  programs - Program Files directories
  downloads- User downloads and desktop
  full     - Complete system scan (very slow!)
        """
    )

    # Scan options
    parser.add_argument('-d', '--directory', action='append', help='Directory to scan (can be used multiple times)')
    parser.add_argument('--preset', choices=['quick', 'system', 'programs', 'downloads', 'full'],
                        help='Use predefined scan locations')
    parser.add_argument('-r', '--recursive', action='store_true', default=True,
                        help='Scan directories recursively (default: True)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to scan')

    # Performance options
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='Number of scanning threads (default: 4)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    # Filtering options
    parser.add_argument('--packed-only', action='store_true',
                        help='Show only packed files')
    parser.add_argument('--min-confidence', type=int, default=0,
                        help='Minimum confidence score (0-100)')
    parser.add_argument('--packer-type', action='append',
                        help='Filter by packer type (can be used multiple times)')

    # Output options
    parser.add_argument('--export-json', help='Export results to JSON file')
    parser.add_argument('--export-csv', help='Export results to CSV file')
    parser.add_argument('--export-html', help='Export results to HTML report')
    parser.add_argument('--export-all', action='store_true',
                        help='Export to all formats (auto-generated filenames)')
    parser.add_argument('--no-summary', action='store_true',
                        help='Don\'t print summary to console')
    parser.add_argument('--list-packed', type=int, default=20, metavar='N',
                        help='List top N packed files (default: 20, 0 to disable)')

    args = parser.parse_args()

    # Determine scan paths
    scan_paths = []

    if args.preset:
        scan_paths = WindowsPackerScanner.SCAN_PRESETS.get(args.preset, [])
        print(f"[*] Using preset: {args.preset}")

    if args.directory:
        scan_paths.extend(args.directory)

    if not scan_paths:
        print("[ERROR] No scan paths specified. Use --directory or --preset")
        parser.print_help()
        sys.exit(1)

    # Create scanner
    scanner = WindowsPackerScanner(max_workers=args.threads, verbose=args.verbose)

    # Find PE files
    pe_files = scanner.find_pe_files(scan_paths, recursive=args.recursive, max_files=args.max_files)

    if not pe_files:
        print("[!] No PE files found to scan")
        sys.exit(0)

    # Scan files
    results = scanner.scan_files(pe_files, show_progress=not args.verbose)

    if not results:
        print("[!] No results generated")
        sys.exit(0)

    # Filter results
    filtered_results = scanner.filter_results(
        results,
        min_confidence=args.min_confidence,
        packed_only=args.packed_only,
        packer_types=args.packer_type
    )

    # Generate statistics
    stats = scanner.generate_statistics(filtered_results)

    # Print summary
    if not args.no_summary:
        scanner.print_summary(filtered_results, stats)

    # List packed files
    if args.list_packed > 0:
        scanner.print_packed_files(filtered_results, limit=args.list_packed)

    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.export_all:
        scanner.export_json(filtered_results, f'packer_scan_{timestamp}.json')
        scanner.export_csv(filtered_results, f'packer_scan_{timestamp}.csv')
        scanner.export_html(filtered_results, f'packer_scan_{timestamp}.html')
    else:
        if args.export_json:
            scanner.export_json(filtered_results, args.export_json)
        if args.export_csv:
            scanner.export_csv(filtered_results, args.export_csv)
        if args.export_html:
            scanner.export_html(filtered_results, args.export_html)


if __name__ == '__main__':
    main()
