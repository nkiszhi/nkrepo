# -*- coding: utf-8 -*-
"""
Test script for PE Packer Detector
Demonstrates usage and validates functionality.

Author: NKAMG (Nankai Anti-Malware Group)
"""

import os
import sys
from pathlib import Path
from packer_detector import PEPackerDetector, print_detection_result


def test_single_file(file_path: str, show_info: bool = False):
    """
    Test the packer detector on a single file.

    Args:
        file_path: Path to PE file
        show_info: Whether to show detailed PE information
    """
    print(f"\n{'='*70}")
    print(f"Testing: {file_path}")
    print(f"{'='*70}")

    detector = PEPackerDetector(file_path)

    # Optional: Show detailed PE info
    if show_info:
        if detector.load_pe():
            detector.print_pe_info()

    # Run detection
    result = detector.detect()

    # Print results
    print_detection_result(result)

    return result


def test_directory(directory_path: str, extensions: tuple = ('.exe', '.dll', '.sys')):
    """
    Test the packer detector on all PE files in a directory.

    Args:
        directory_path: Path to directory containing PE files
        extensions: File extensions to scan
    """
    print(f"\n{'='*70}")
    print(f"Scanning Directory: {directory_path}")
    print(f"{'='*70}\n")

    results = []
    pe_files = []

    # Find all PE files
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(extensions):
                pe_files.append(os.path.join(root, filename))

    if not pe_files:
        print(f"No PE files found with extensions {extensions}")
        return results

    print(f"Found {len(pe_files)} PE files to analyze\n")

    # Analyze each file
    for i, file_path in enumerate(pe_files, 1):
        print(f"[{i}/{len(pe_files)}] {Path(file_path).name}...", end=" ")

        detector = PEPackerDetector(file_path)
        result = detector.detect()

        # Compact output
        if result['is_packed']:
            print(f"✗ PACKED ({result['packer_type']}, {result['confidence_score']}%)")
        else:
            print(f"✓ NOT PACKED")

        results.append({
            'file': Path(file_path).name,
            'path': file_path,
            'is_packed': result['is_packed'],
            'packer_type': result['packer_type'],
            'confidence': result['confidence_score']
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    packed_count = sum(1 for r in results if r['is_packed'])
    unpacked_count = len(results) - packed_count

    print(f"Total files analyzed:  {len(results)}")
    print(f"Packed files:          {packed_count}")
    print(f"Unpacked files:        {unpacked_count}")

    # Show packed files
    if packed_count > 0:
        print(f"\nPacked files:")
        for r in results:
            if r['is_packed']:
                print(f"  - {r['file']:<30} {r['packer_type']:<20} ({r['confidence']}%)")

    return results


def test_entropy_calculation():
    """
    Test the entropy calculation function with known values.
    """
    print(f"\n{'='*70}")
    print("Testing Entropy Calculation")
    print(f"{'='*70}\n")

    test_cases = [
        (b'\x00' * 1000, "All zeros (low entropy)"),
        (b'\xFF' * 1000, "All 0xFF (low entropy)"),
        (bytes(range(256)) * 10, "Uniform distribution (high entropy)"),
        (os.urandom(1000), "Random data (high entropy)"),
    ]

    for data, description in test_cases:
        entropy = PEPackerDetector.calculate_entropy(data)
        print(f"{description:<35} Entropy: {entropy:.4f}")

    print("\nNote: Packed/encrypted sections typically have entropy > 7.0")


def test_packer_signatures():
    """
    Display all supported packer signatures.
    """
    print(f"\n{'='*70}")
    print("Supported Packer Signatures")
    print(f"{'='*70}\n")

    for packer, signatures in PEPackerDetector.PACKER_SIGNATURES.items():
        print(f"{packer}:")
        for sig in signatures:
            print(f"  - {sig}")
        print()


def interactive_test():
    """
    Interactive testing mode - prompts user for file path.
    """
    print(f"\n{'='*70}")
    print("Interactive Testing Mode")
    print(f"{'='*70}\n")

    while True:
        file_path = input("Enter PE file path (or 'quit' to exit): ").strip()

        if file_path.lower() in ['quit', 'q', 'exit']:
            break

        if not file_path:
            continue

        # Remove quotes if present
        file_path = file_path.strip('"').strip("'")

        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}\n")
            continue

        test_single_file(file_path, show_info=False)

        print("\n" + "-" * 70 + "\n")


def main():
    """
    Main test function with various testing modes.
    """
    print("="*70)
    print("PE Packer Detector - Test Suite")
    print("NKAMG (Nankai Anti-Malware Group)")
    print("="*70)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_packer_detector.py <file_or_directory>")
        print("  python test_packer_detector.py --entropy")
        print("  python test_packer_detector.py --signatures")
        print("  python test_packer_detector.py --interactive")
        print("\nExamples:")
        print("  python test_packer_detector.py C:\\samples\\malware.exe")
        print("  python test_packer_detector.py C:\\samples\\")
        print("  python test_packer_detector.py --interactive")
        sys.exit(1)

    arg = sys.argv[1]

    # Special modes
    if arg == "--entropy":
        test_entropy_calculation()
        return

    if arg == "--signatures":
        test_packer_signatures()
        return

    if arg == "--interactive" or arg == "-i":
        interactive_test()
        return

    # File or directory path
    path = Path(arg)

    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    # Test single file or directory
    if path.is_file():
        show_info = "--info" in sys.argv
        test_single_file(str(path), show_info=show_info)
    elif path.is_dir():
        test_directory(str(path))
    else:
        print(f"Error: Invalid path: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
