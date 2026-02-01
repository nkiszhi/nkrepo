# -*- coding: utf-8 -*-
"""
PE File Hash Calculator
Calculates multiple hash types for PE files used in malware analysis.

Supported hashes:
- MD5, SHA1, SHA256, SHA512
- Imphash (Import Hash)
- Authentihash (PE authenticode hash)
- Rich Header Hash
- SSDEEP (Fuzzy Hash)
- TLSH (Trend Micro Locality Sensitive Hash)
- vHash (VirusTotal-style hash)

Author: NKAMG (Nankai Anti-Malware Group)
"""

import os
import sys
import json
import csv
import hashlib
import struct
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# Try to import optional libraries
try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False
    print("[WARNING] pefile not installed - imphash will not be available")
    print("Install with: pip install pefile")

try:
    import ssdeep
    HAS_SSDEEP = True
except ImportError:
    HAS_SSDEEP = False
    print("[WARNING] ssdeep not installed - fuzzy hashing will not be available")
    print("Install with: pip install ssdeep")

try:
    import tlsh
    HAS_TLSH = True
except ImportError:
    HAS_TLSH = False
    print("[WARNING] tlsh not installed - TLSH hashing will not be available")
    print("Install with: pip install py-tlsh")

try:
    import lief
    HAS_LIEF = True
except ImportError:
    HAS_LIEF = False
    print("[WARNING] LIEF not installed - authentihash will not be available")


class PEHashCalculator:
    """
    Calculator for multiple hash types of PE files.

    Supports cryptographic hashes, fuzzy hashes, and PE-specific hashes.
    """

    def __init__(self, file_path: str, verbose: bool = False):
        """
        Initialize the hash calculator.

        Args:
            file_path: Path to PE file
            verbose: Enable verbose output
        """
        self.file_path = Path(file_path)
        self.verbose = verbose
        self.file_data = None
        self.pe = None
        self.lief_binary = None

    def load_file(self) -> bool:
        """
        Load the PE file into memory.

        Returns:
            True if successful, False otherwise
        """
        if not self.file_path.exists():
            print(f"[ERROR] File not found: {self.file_path}")
            return False

        try:
            # Read file data
            with open(self.file_path, 'rb') as f:
                self.file_data = f.read()

            # Load with pefile if available
            if HAS_PEFILE:
                try:
                    self.pe = pefile.PE(data=self.file_data)
                except pefile.PEFormatError:
                    if self.verbose:
                        print(f"[WARNING] Not a valid PE file (pefile)")
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] pefile error: {e}")

            # Load with LIEF if available
            if HAS_LIEF:
                try:
                    self.lief_binary = lief.parse(str(self.file_path))
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] LIEF error: {e}")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to load file: {e}")
            return False

    def calculate_md5(self) -> Optional[str]:
        """Calculate MD5 hash."""
        try:
            return hashlib.md5(self.file_data).hexdigest()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] MD5 calculation failed: {e}")
            return None

    def calculate_sha1(self) -> Optional[str]:
        """Calculate SHA1 hash."""
        try:
            return hashlib.sha1(self.file_data).hexdigest()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] SHA1 calculation failed: {e}")
            return None

    def calculate_sha256(self) -> Optional[str]:
        """Calculate SHA256 hash."""
        try:
            return hashlib.sha256(self.file_data).hexdigest()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] SHA256 calculation failed: {e}")
            return None

    def calculate_sha512(self) -> Optional[str]:
        """Calculate SHA512 hash."""
        try:
            return hashlib.sha512(self.file_data).hexdigest()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] SHA512 calculation failed: {e}")
            return None

    def calculate_imphash(self) -> Optional[str]:
        """
        Calculate Import Hash (imphash).

        Imphash is based on the imported DLLs and functions.
        Useful for identifying malware families.

        Returns:
            Imphash string or None
        """
        if not HAS_PEFILE or not self.pe:
            return None

        try:
            return self.pe.get_imphash()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Imphash calculation failed: {e}")
            return None

    def calculate_authentihash(self) -> Optional[str]:
        """
        Calculate Authenticode hash (authentihash).

        This is the hash used by Windows Authenticode signatures.
        Excludes the signature itself from the hash.

        Returns:
            Authentihash (SHA256) or None
        """
        if not self.pe:
            return None

        try:
            # Get the authentihash using pefile
            # This excludes checksum and certificate table
            return self._compute_authentihash_sha256()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Authentihash calculation failed: {e}")
            return None

    def _compute_authentihash_sha256(self) -> Optional[str]:
        """
        Compute authentihash manually.

        Authentihash excludes:
        - Checksum field
        - Certificate table entry in data directories
        - Certificate data

        Returns:
            SHA256 hash as hex string
        """
        if not self.pe:
            return None

        try:
            # Get file data
            data = bytearray(self.file_data)

            # Get checksum offset (in optional header)
            checksum_offset = (self.pe.DOS_HEADER.e_lfanew + 4 +
                              self.pe.FILE_HEADER.sizeof() + 64)

            # Zero out checksum (4 bytes)
            data[checksum_offset:checksum_offset + 4] = b'\x00' * 4

            # Get certificate table offset (in data directory)
            cert_table_offset = (self.pe.DOS_HEADER.e_lfanew + 4 +
                                self.pe.FILE_HEADER.sizeof() + 128)

            # Zero out certificate table entry (8 bytes: RVA + Size)
            data[cert_table_offset:cert_table_offset + 8] = b'\x00' * 8

            # If certificate data exists, exclude it
            if hasattr(self.pe, 'OPTIONAL_HEADER') and \
               len(self.pe.OPTIONAL_HEADER.DATA_DIRECTORY) > 4:
                cert_entry = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[4]  # Security directory
                if cert_entry.VirtualAddress and cert_entry.Size:
                    # Truncate at certificate start
                    cert_offset = cert_entry.VirtualAddress
                    data = data[:cert_offset]

            # Calculate SHA256
            return hashlib.sha256(bytes(data)).hexdigest()

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Authentihash computation error: {e}")
            return None

    def calculate_rich_header_hash(self) -> Optional[str]:
        """
        Calculate Rich PE header hash.

        The Rich header contains information about the build environment.
        Useful for identifying compiler toolchains.

        Returns:
            MD5 hash of Rich header or None
        """
        if not self.pe:
            return None

        try:
            # Find Rich header
            rich_header = self._extract_rich_header()

            if rich_header:
                return hashlib.md5(rich_header).hexdigest()
            else:
                return None

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Rich header hash calculation failed: {e}")
            return None

    def _extract_rich_header(self) -> Optional[bytes]:
        """
        Extract Rich header from PE file.

        Returns:
            Rich header bytes or None
        """
        try:
            # Rich header is between DOS stub and PE signature
            # Marked by "Rich" string followed by XOR key

            # Search for "Rich" signature
            rich_sig = b'Rich'
            rich_offset = self.file_data.find(rich_sig)

            if rich_offset == -1:
                return None

            # Get XOR key (4 bytes after "Rich")
            xor_key = struct.unpack('<I', self.file_data[rich_offset + 4:rich_offset + 8])[0]

            # Search backwards for DanS signature (XORed)
            dans_sig_xored = struct.pack('<I', 0x536E6144 ^ xor_key)  # "DanS" XORed

            # Start from Rich offset and search backwards
            start_offset = 0x80  # Typical start after DOS header
            dans_offset = self.file_data.rfind(dans_sig_xored, start_offset, rich_offset)

            if dans_offset == -1:
                return None

            # Extract Rich header (from DanS to end of Rich)
            rich_header = self.file_data[dans_offset:rich_offset + 8]

            # Clear (decode) the Rich header
            cleared_header = bytearray()
            for i in range(0, len(rich_header), 4):
                if i + 4 <= len(rich_header):
                    dword = struct.unpack('<I', rich_header[i:i + 4])[0]
                    cleared_dword = dword ^ xor_key
                    cleared_header.extend(struct.pack('<I', cleared_dword))

            return bytes(cleared_header)

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Rich header extraction error: {e}")
            return None

    def calculate_ssdeep(self) -> Optional[str]:
        """
        Calculate SSDEEP fuzzy hash.

        SSDEEP is used for similarity detection.
        Two similar files will have similar SSDEEP hashes.

        Returns:
            SSDEEP hash string or None
        """
        if not HAS_SSDEEP:
            return None

        try:
            return ssdeep.hash(self.file_data)
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] SSDEEP calculation failed: {e}")
            return None

    def calculate_tlsh(self) -> Optional[str]:
        """
        Calculate TLSH (Trend Micro Locality Sensitive Hash).

        TLSH is another fuzzy hash for similarity detection.
        More robust than SSDEEP for larger files.

        Returns:
            TLSH hash string or None
        """
        if not HAS_TLSH:
            return None

        try:
            # TLSH requires at least 256 bytes
            if len(self.file_data) < 256:
                return None

            return tlsh.hash(self.file_data)
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] TLSH calculation failed: {e}")
            return None

    def calculate_vhash(self) -> Optional[str]:
        """
        Calculate vHash (VirusTotal-style hash).

        vHash is a similarity hash used by VirusTotal.
        Based on PE structural features.

        Note: This is a simplified implementation.
        The actual VirusTotal vHash algorithm is proprietary.

        Returns:
            vHash-like string or None
        """
        if not self.pe:
            return None

        try:
            # Collect PE features for hashing
            features = []

            # Section names and characteristics
            for section in self.pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').rstrip('\x00')
                features.append(section_name)
                features.append(str(section.Characteristics))

            # Import table
            if hasattr(self.pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in self.pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8', errors='ignore')
                    features.append(dll_name)

            # Export table
            if hasattr(self.pe, 'DIRECTORY_ENTRY_EXPORT'):
                for exp in self.pe.DIRECTORY_ENTRY_EXPORT.symbols:
                    if exp.name:
                        exp_name = exp.name.decode('utf-8', errors='ignore')
                        features.append(exp_name)

            # Entry point
            features.append(str(self.pe.OPTIONAL_HEADER.AddressOfEntryPoint))

            # Image base
            features.append(str(self.pe.OPTIONAL_HEADER.ImageBase))

            # Combine features and hash
            feature_string = '|'.join(features)
            vhash = hashlib.md5(feature_string.encode()).hexdigest()

            return vhash

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] vHash calculation failed: {e}")
            return None

    def calculate_pehash(self) -> Optional[str]:
        """
        Calculate PEhash (structural hash).

        PEhash is based on PE structural features.
        Similar to vHash but uses different algorithm.

        Returns:
            PEhash string or None
        """
        if not self.pe:
            return None

        try:
            # Collect structural features
            features = []

            # Machine type
            features.append(str(self.pe.FILE_HEADER.Machine))

            # Subsystem
            features.append(str(self.pe.OPTIONAL_HEADER.Subsystem))

            # Number of sections
            features.append(str(self.pe.FILE_HEADER.NumberOfSections))

            # Timestamp (normalized to reduce noise)
            timestamp = self.pe.FILE_HEADER.TimeDateStamp
            features.append(str(timestamp // 3600))  # Normalize to hours

            # Section info
            for section in self.pe.sections:
                # Section name
                sec_name = section.Name.decode('utf-8', errors='ignore').rstrip('\x00')
                features.append(sec_name)

                # Section size (normalized)
                features.append(str(section.SizeOfRawData // 4096))  # Normalize to 4KB blocks

                # Section characteristics
                features.append(str(section.Characteristics))

            # Combine and hash
            feature_string = '|'.join(features)
            return hashlib.sha256(feature_string.encode()).hexdigest()[:32]

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] PEhash calculation failed: {e}")
            return None

    def calculate_all_hashes(self) -> Dict:
        """
        Calculate all available hash types.

        Returns:
            Dictionary with all hash values
        """
        if not self.load_file():
            return {
                'error': 'Failed to load file',
                'file_path': str(self.file_path)
            }

        hashes = {
            'file_path': str(self.file_path),
            'file_name': self.file_path.name,
            'file_size': len(self.file_data),
            'calculated_at': datetime.now().isoformat()
        }

        # Cryptographic hashes (always available)
        hashes['md5'] = self.calculate_md5()
        hashes['sha1'] = self.calculate_sha1()
        hashes['sha256'] = self.calculate_sha256()
        hashes['sha512'] = self.calculate_sha512()

        # PE-specific hashes (require pefile)
        hashes['imphash'] = self.calculate_imphash()
        hashes['authentihash'] = self.calculate_authentihash()
        hashes['rich_header_hash'] = self.calculate_rich_header_hash()
        hashes['vhash'] = self.calculate_vhash()
        hashes['pehash'] = self.calculate_pehash()

        # Fuzzy hashes (require additional libraries)
        hashes['ssdeep'] = self.calculate_ssdeep()
        hashes['tlsh'] = self.calculate_tlsh()

        # Add availability info
        hashes['_metadata'] = {
            'has_pefile': HAS_PEFILE,
            'has_ssdeep': HAS_SSDEEP,
            'has_tlsh': HAS_TLSH,
            'has_lief': HAS_LIEF,
            'is_pe': self.pe is not None
        }

        return hashes

    def print_hashes(self, hashes: Dict):
        """
        Print hashes in a formatted way.

        Args:
            hashes: Dictionary of hash values
        """
        print("\n" + "="*70)
        print("PE FILE HASH ANALYSIS")
        print("="*70)

        print(f"\nFile: {hashes.get('file_name', 'N/A')}")
        print(f"Path: {hashes.get('file_path', 'N/A')}")
        print(f"Size: {hashes.get('file_size', 0):,} bytes")

        print(f"\n{'-'*70}")
        print("CRYPTOGRAPHIC HASHES")
        print(f"{'-'*70}")
        print(f"MD5:        {hashes.get('md5', 'N/A')}")
        print(f"SHA1:       {hashes.get('sha1', 'N/A')}")
        print(f"SHA256:     {hashes.get('sha256', 'N/A')}")
        print(f"SHA512:     {hashes.get('sha512', 'N/A')}")

        print(f"\n{'-'*70}")
        print("PE-SPECIFIC HASHES")
        print(f"{'-'*70}")
        print(f"Imphash:    {hashes.get('imphash', 'N/A')}")
        print(f"Authentihash: {hashes.get('authentihash', 'N/A')}")
        print(f"Rich Header: {hashes.get('rich_header_hash', 'N/A')}")
        print(f"vHash:      {hashes.get('vhash', 'N/A')}")
        print(f"PEhash:     {hashes.get('pehash', 'N/A')}")

        print(f"\n{'-'*70}")
        print("FUZZY HASHES")
        print(f"{'-'*70}")
        print(f"SSDEEP:     {hashes.get('ssdeep', 'N/A')}")
        print(f"TLSH:       {hashes.get('tlsh', 'N/A')}")

        print("="*70 + "\n")


def batch_calculate_hashes(file_paths: List[str], verbose: bool = False) -> List[Dict]:
    """
    Calculate hashes for multiple files.

    Args:
        file_paths: List of file paths
        verbose: Enable verbose output

    Returns:
        List of hash dictionaries
    """
    results = []

    print(f"[*] Calculating hashes for {len(file_paths)} files...")

    for i, file_path in enumerate(file_paths, 1):
        print(f"[{i}/{len(file_paths)}] Processing: {Path(file_path).name}")

        calculator = PEHashCalculator(file_path, verbose=verbose)
        hashes = calculator.calculate_all_hashes()
        results.append(hashes)

    return results


def export_to_json(results: List[Dict], output_path: str):
    """Export results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] Results exported to JSON: {output_path}")


def export_to_csv(results: List[Dict], output_path: str):
    """Export results to CSV file."""
    if not results:
        return

    # Get all hash keys
    hash_keys = ['file_name', 'file_path', 'file_size', 'md5', 'sha1', 'sha256',
                 'sha512', 'imphash', 'authentihash', 'rich_header_hash',
                 'vhash', 'pehash', 'ssdeep', 'tlsh']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=hash_keys, extrasaction='ignore')
        writer.writeheader()

        for result in results:
            # Filter out metadata
            filtered = {k: v for k, v in result.items() if k in hash_keys}
            writer.writerow(filtered)

    print(f"[+] Results exported to CSV: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PE File Hash Calculator - Multiple hash types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hash Types:
  Cryptographic:  MD5, SHA1, SHA256, SHA512
  PE-Specific:    Imphash, Authentihash, Rich Header Hash, vHash, PEhash
  Fuzzy:          SSDEEP, TLSH

Examples:
  # Calculate all hashes for a single file
  python pe_hash_calculator.py sample.exe

  # Batch calculate for directory
  python pe_hash_calculator.py -d C:/samples/

  # Export to JSON
  python pe_hash_calculator.py -d C:/samples/ --export-json hashes.json

  # Export to CSV
  python pe_hash_calculator.py -d C:/samples/ --export-csv hashes.csv

  # Both formats
  python pe_hash_calculator.py -d C:/samples/ --export-all

Dependencies:
  Required:     (built-in libraries only for basic hashes)
  Optional:     pip install pefile ssdeep py-tlsh
        """
    )

    parser.add_argument('file', nargs='?', help='PE file to analyze')
    parser.add_argument('-d', '--directory', help='Directory to scan')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Scan directory recursively')
    parser.add_argument('--export-json', help='Export results to JSON')
    parser.add_argument('--export-csv', help='Export results to CSV')
    parser.add_argument('--export-all', action='store_true',
                        help='Export to both JSON and CSV')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if not args.file and not args.directory:
        parser.print_help()
        sys.exit(1)

    # Collect files
    file_paths = []

    if args.file:
        file_paths.append(args.file)

    if args.directory:
        dir_path = Path(args.directory)
        if args.recursive:
            file_paths.extend([str(f) for f in dir_path.rglob('*.exe')])
            file_paths.extend([str(f) for f in dir_path.rglob('*.dll')])
        else:
            file_paths.extend([str(f) for f in dir_path.glob('*.exe')])
            file_paths.extend([str(f) for f in dir_path.glob('*.dll')])

    if not file_paths:
        print("[!] No files to process")
        sys.exit(1)

    # Calculate hashes
    if len(file_paths) == 1:
        # Single file - print detailed output
        calculator = PEHashCalculator(file_paths[0], verbose=args.verbose)
        hashes = calculator.calculate_all_hashes()
        calculator.print_hashes(hashes)
        results = [hashes]
    else:
        # Multiple files - batch process
        results = batch_calculate_hashes(file_paths, verbose=args.verbose)

        # Print summary
        print(f"\n[*] Processed {len(results)} files")

    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.export_all:
        export_to_json(results, f'pe_hashes_{timestamp}.json')
        export_to_csv(results, f'pe_hashes_{timestamp}.csv')
    else:
        if args.export_json:
            export_to_json(results, args.export_json)
        if args.export_csv:
            export_to_csv(results, args.export_csv)


if __name__ == '__main__':
    main()
