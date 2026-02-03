# -*- coding: utf-8 -*-
"""
PE Packer Detector for Malware Analysis
Uses LIEF library to detect packed executables through multiple heuristics.

Author: NKAMG (Nankai Anti-Malware Group)
"""

import sys
import math
import lief
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PEPackerDetector:
    """
    PE Packer Detection Engine using multi-dimensional heuristics.

    Detection Features:
    - Section table analysis (packer signatures, RWX permissions)
    - Import table anomalies
    - Entropy analysis (Shannon entropy)
    - Entry point location verification
    """

    # Packer signature database - section name patterns
    PACKER_SIGNATURES = {
        'UPX': [
            'UPX0', 'UPX1', 'UPX2', '.UPX0', '.UPX1', '.UPX2',
            'UPX!', '.UPX!'
        ],
        'VMProtect': [
            '.vmp0', '.vmp1', '.vmp2', 'vmp0', 'vmp1', 'vmp2',
            '.VMPROT'
        ],
        'ASPack': [
            '.aspack', '.adata', 'ASPack', '.ASPack'
        ],
        'Themida': [
            '.themida', 'Themida', '.winlice', 'WinLicense',
            '.tmd', '.tmc'
        ],
        'Enigma': [
            '.enigma1', '.enigma2', 'enigma', 'EnigmaProtector'
        ],
        'PECompact': [
            'PEC2', 'PECompact2', 'pec1', 'pec2'
        ],
        'NSPack': [
            '.nsp0', '.nsp1', '.nsp2', 'nsp0', 'nsp1'
        ],
        'Armadillo': [
            '.arma', 'Armadillo'
        ],
        'Obsidium': [
            '.obsidi', 'Obsidium'
        ],
        'PESpin': [
            '.spin', 'PESpin'
        ],
        'Petite': [
            '.petite', 'petite'
        ],
        'MPRESS': [
            '.MPRESS', 'MPRESS1', 'MPRESS2'
        ]
    }

    # Common suspicious section names
    SUSPICIOUS_SECTION_NAMES = [
        '.packed', '.crypted', '.data1', '.data2', '.boot',
        '.perplex', '.edata', '.yP', '.gentee', '.shrink'
    ]

    # Standard PE section names (legitimate)
    STANDARD_SECTIONS = [
        '.text', '.code', '.data', '.rdata', '.bss',
        '.idata', '.edata', '.rsrc', '.reloc', '.tls'
    ]

    # Detection weights for confidence scoring
    WEIGHTS = {
        'packer_signature': 40,      # Highest weight - definitive indicator
        'missing_text_section': 25,   # Very suspicious
        'rwx_section': 20,            # Code injection indicator
        'low_imports': 15,            # Common in packed files
        'high_entropy_section': 15,   # Compression/encryption indicator
        'oep_not_in_text': 20,       # Entry point anomaly
        'suspicious_section_name': 10
    }

    def __init__(self, file_path: str):
        """
        Initialize the packer detector.

        Args:
            file_path: Path to the PE file to analyze
        """
        self.file_path = Path(file_path)
        self.binary: Optional[lief.PE.Binary] = None
        self.is_valid_pe = False
        self.is_64bit = False

    def load_pe(self) -> bool:
        """
        Load and parse the PE file using LIEF.

        Returns:
            True if PE loaded successfully, False otherwise
        """
        if not self.file_path.exists():
            print(f"[ERROR] File not found: {self.file_path}")
            return False

        try:
            # Parse PE file with LIEF
            self.binary = lief.parse(str(self.file_path))

            # Verify it's a PE file
            if self.binary is None or not isinstance(self.binary, lief.PE.Binary):
                print(f"[ERROR] Not a valid PE file: {self.file_path.name}")
                return False

            self.is_valid_pe = True

            # Determine architecture (compatible with different LIEF versions)
            machine_type = self.binary.header.machine

            # Try enum-based comparison first (newer LIEF versions)
            try:
                from lief.PE import MACHINE_TYPES
                if machine_type == MACHINE_TYPES.AMD64:
                    self.is_64bit = True
                elif machine_type == MACHINE_TYPES.I386:
                    self.is_64bit = False
            except (ImportError, AttributeError):
                # Fall back to raw value comparison (works with all versions)
                # IMAGE_FILE_MACHINE_AMD64 = 0x8664 (34404)
                # IMAGE_FILE_MACHINE_I386 = 0x014c (332)
                if isinstance(machine_type, int):
                    if machine_type == 0x8664 or machine_type == 34404:
                        self.is_64bit = True
                    elif machine_type == 0x014c or machine_type == 332:
                        self.is_64bit = False
                else:
                    # Handle enum-like objects
                    machine_str = str(machine_type).upper()
                    if 'AMD64' in machine_str or 'X64' in machine_str:
                        self.is_64bit = True
                    elif 'I386' in machine_str or 'X86' in machine_str:
                        self.is_64bit = False

            return True

        except Exception as e:
            print(f"[ERROR] Failed to parse PE file: {e}")
            return False

    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        """
        Calculate Shannon entropy of binary data.

        Entropy ranges from 0 (not random) to 8 (completely random).
        Packed/encrypted sections typically have entropy > 7.

        Args:
            data: Binary data to analyze

        Returns:
            Entropy value (0-8)
        """
        if not data:
            return 0.0

        # Count byte frequency
        entropy = 0.0
        byte_counts = [0] * 256

        for byte in data:
            byte_counts[byte] += 1

        # Calculate Shannon entropy
        data_len = len(data)
        for count in byte_counts:
            if count == 0:
                continue
            probability = count / data_len
            entropy -= probability * math.log2(probability)

        return entropy

    def check_section_table(self) -> Tuple[List[str], int]:
        """
        Analyze section table for packer indicators.

        Checks for:
        - Known packer section names
        - Suspicious section names
        - RWX (Read-Write-Execute) sections
        - Missing standard .text section

        Returns:
            Tuple of (suspicious_features, confidence_score)
        """
        suspicious = []
        score = 0

        if not self.binary.sections:
            suspicious.append("No sections found")
            return suspicious, 0

        section_names = [s.name.lower() for s in self.binary.sections]
        has_text_section = any('.text' in name for name in section_names)

        # Check for missing .text section
        if not has_text_section:
            suspicious.append("Missing standard .text section")
            score += self.WEIGHTS['missing_text_section']

        # Analyze each section
        for section in self.binary.sections:
            sec_name = section.name

            # Check for packer signatures
            for packer, signatures in self.PACKER_SIGNATURES.items():
                if any(sig.lower() in sec_name.lower() for sig in signatures):
                    suspicious.append(f"Packer signature detected: {packer} ({sec_name})")
                    score += self.WEIGHTS['packer_signature']
                    break

            # Check for suspicious section names
            if any(susp.lower() in sec_name.lower() for susp in self.SUSPICIOUS_SECTION_NAMES):
                suspicious.append(f"Suspicious section name: {sec_name}")
                score += self.WEIGHTS['suspicious_section_name']

            # Check for RWX sections (common in packed files)
            characteristics = section.characteristics

            # Try to use LIEF constants, fall back to raw values if not available
            try:
                from lief.PE import SECTION_CHARACTERISTICS
                is_readable = characteristics & SECTION_CHARACTERISTICS.MEM_READ
                is_writable = characteristics & SECTION_CHARACTERISTICS.MEM_WRITE
                is_executable = characteristics & SECTION_CHARACTERISTICS.MEM_EXECUTE
            except (ImportError, AttributeError):
                # Raw PE section characteristic flags
                # IMAGE_SCN_MEM_READ = 0x40000000
                # IMAGE_SCN_MEM_WRITE = 0x80000000
                # IMAGE_SCN_MEM_EXECUTE = 0x20000000
                is_readable = characteristics & 0x40000000
                is_writable = characteristics & 0x80000000
                is_executable = characteristics & 0x20000000

            if is_readable and is_writable and is_executable:
                suspicious.append(f"RWX section found: {sec_name}")
                score += self.WEIGHTS['rwx_section']

        return suspicious, score

    def check_import_table(self) -> Tuple[List[str], int]:
        """
        Analyze import table for anomalies.

        Packed files often have minimal imports or only dynamic loading functions.

        Returns:
            Tuple of (suspicious_features, confidence_score)
        """
        suspicious = []
        score = 0

        if not self.binary.has_imports:
            suspicious.append("No import table found")
            score += self.WEIGHTS['low_imports']
            return suspicious, score

        # Count total imported functions
        total_imports = 0
        imported_functions = []

        for library in self.binary.imports:
            for entry in library.entries:
                if entry.name:
                    total_imports += 1
                    imported_functions.append(entry.name.lower())

        # Check for suspiciously low import count
        if total_imports < 5:
            suspicious.append(f"Very few imports: {total_imports} functions")
            score += self.WEIGHTS['low_imports']

        # Check for only dynamic loading functions (common in packed files)
        dynamic_loader_funcs = ['loadlibrarya', 'loadlibraryw', 'getprocaddress']
        if total_imports <= 3 and all(func in dynamic_loader_funcs for func in imported_functions):
            suspicious.append("Only dynamic loading functions imported")
            score += self.WEIGHTS['low_imports']

        return suspicious, score

    def check_entropy(self) -> Tuple[List[str], int]:
        """
        Analyze section entropy for compression/encryption indicators.

        High entropy (>7) indicates compressed or encrypted data.

        Returns:
            Tuple of (suspicious_features, confidence_score)
        """
        suspicious = []
        score = 0
        high_entropy_threshold = 7.0

        for section in self.binary.sections:
            try:
                # Get section data
                section_data = bytes(section.content)

                if len(section_data) == 0:
                    continue

                # Calculate entropy
                entropy = self.calculate_entropy(section_data)

                # Flag high entropy sections (likely compressed/encrypted)
                if entropy > high_entropy_threshold:
                    suspicious.append(
                        f"High entropy section: {section.name} (entropy={entropy:.2f})"
                    )
                    score += self.WEIGHTS['high_entropy_section']

            except Exception as e:
                continue

        return suspicious, score

    def check_entry_point(self) -> Tuple[List[str], int]:
        """
        Verify entry point location (Original Entry Point - OEP).

        Legitimate PE files typically have their entry point in the .text section.
        Packed files often have OEP in non-standard sections.

        Returns:
            Tuple of (suspicious_features, confidence_score)
        """
        suspicious = []
        score = 0

        try:
            # Get entry point RVA (Relative Virtual Address)
            entry_point = self.binary.optional_header.addressof_entrypoint

            # Find which section contains the entry point
            ep_section = None
            for section in self.binary.sections:
                section_start = section.virtual_address
                section_end = section_start + section.virtual_size

                if section_start <= entry_point < section_end:
                    ep_section = section
                    break

            if ep_section:
                section_name = ep_section.name.lower()

                # Check if entry point is NOT in .text section
                if '.text' not in section_name and 'code' not in section_name:
                    suspicious.append(
                        f"Entry point not in .text section: {ep_section.name} "
                        f"(RVA=0x{entry_point:08x})"
                    )
                    score += self.WEIGHTS['oep_not_in_text']
            else:
                suspicious.append(f"Entry point in unknown section (RVA=0x{entry_point:08x})")
                score += self.WEIGHTS['oep_not_in_text']

        except Exception as e:
            suspicious.append(f"Failed to check entry point: {e}")

        return suspicious, score

    def detect(self) -> Dict:
        """
        Main detection method - runs all heuristics and aggregates results.

        Returns:
            Dictionary containing:
            - is_packed (bool): Whether file appears to be packed
            - packer_type (str): Identified packer or "Unknown Packer"/"No Packer"
            - suspicious_features (list): List of detected anomalies
            - confidence_score (int): 0-100 confidence score
            - file_info (dict): Basic PE information
        """
        if not self.load_pe():
            return {
                'is_packed': False,
                'packer_type': 'Invalid PE',
                'suspicious_features': ['Failed to load PE file'],
                'confidence_score': 0,
                'file_info': {}
            }

        all_suspicious = []
        total_score = 0
        detected_packer = None

        # Run all detection checks
        sec_suspicious, sec_score = self.check_section_table()
        all_suspicious.extend(sec_suspicious)
        total_score += sec_score

        imp_suspicious, imp_score = self.check_import_table()
        all_suspicious.extend(imp_suspicious)
        total_score += imp_score

        ent_suspicious, ent_score = self.check_entropy()
        all_suspicious.extend(ent_suspicious)
        total_score += ent_score

        ep_suspicious, ep_score = self.check_entry_point()
        all_suspicious.extend(ep_suspicious)
        total_score += ep_score

        # Identify specific packer from signatures
        for feature in all_suspicious:
            if "Packer signature detected:" in feature:
                detected_packer = feature.split(":")[1].split("(")[0].strip()
                break

        # Normalize confidence score to 0-100
        max_possible_score = sum(self.WEIGHTS.values()) * 2  # Multiple triggers possible
        confidence_score = min(100, int((total_score / max_possible_score) * 100))

        # Determine if file is packed (threshold: 30% confidence or specific packer detected)
        is_packed = confidence_score >= 30 or detected_packer is not None

        # Set packer type
        if detected_packer:
            packer_type = detected_packer
        elif is_packed:
            packer_type = "Unknown Packer"
        else:
            packer_type = "No Packer"

        # Gather basic file info
        file_info = {
            'filename': self.file_path.name,
            'size': self.file_path.stat().st_size,
            'architecture': '64-bit' if self.is_64bit else '32-bit',
            'sections': len(self.binary.sections),
            'imports': len(self.binary.imports) if self.binary.has_imports else 0,
            'entry_point': f"0x{self.binary.optional_header.addressof_entrypoint:08x}"
        }

        return {
            'is_packed': is_packed,
            'packer_type': packer_type,
            'suspicious_features': all_suspicious,
            'confidence_score': confidence_score,
            'file_info': file_info
        }

    def print_pe_info(self):
        """
        Print detailed PE file information for manual verification.
        Useful for understanding detection results.
        """
        if not self.is_valid_pe:
            print("[ERROR] PE file not loaded")
            return

        print("\n" + "="*70)
        print("PE FILE INFORMATION")
        print("="*70)

        # Basic info
        print(f"\nFile: {self.file_path.name}")
        print(f"Size: {self.file_path.stat().st_size:,} bytes")
        print(f"Architecture: {'64-bit' if self.is_64bit else '32-bit'}")
        print(f"Entry Point: 0x{self.binary.optional_header.addressof_entrypoint:08x}")

        # Section table
        print(f"\n{'SECTION TABLE':-^70}")
        print(f"{'Name':<12} {'VirtAddr':<12} {'VirtSize':<12} {'RawSize':<12} {'Entropy':<10}")
        print("-" * 70)

        for section in self.binary.sections:
            section_data = bytes(section.content)
            entropy = self.calculate_entropy(section_data) if section_data else 0.0

            print(f"{section.name:<12} "
                  f"0x{section.virtual_address:08x}  "
                  f"0x{section.virtual_size:08x}  "
                  f"0x{section.sizeof_raw_data:08x}  "
                  f"{entropy:.2f}")

        # Import table
        if self.binary.has_imports:
            print(f"\n{'IMPORT TABLE':-^70}")
            total_funcs = 0
            for library in self.binary.imports:
                func_count = len([e for e in library.entries if e.name])
                total_funcs += func_count
                print(f"{library.name}: {func_count} functions")
            print(f"Total imported functions: {total_funcs}")
        else:
            print(f"\n{'IMPORT TABLE':-^70}")
            print("No imports found")

        print("="*70 + "\n")


def print_detection_result(result: Dict):
    """
    Pretty-print detection results with color coding (optional).

    Args:
        result: Detection result dictionary from PEPackerDetector.detect()
    """
    # ANSI color codes (optional - remove if not needed)
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    print("\n" + "="*70)
    print(f"{BOLD}PE PACKER DETECTION RESULTS{RESET}")
    print("="*70)

    # File info
    if result['file_info']:
        info = result['file_info']
        print(f"\n{BLUE}File Information:{RESET}")
        print(f"  Name:         {info.get('filename', 'N/A')}")
        print(f"  Size:         {info.get('size', 0):,} bytes")
        print(f"  Architecture: {info.get('architecture', 'N/A')}")
        print(f"  Sections:     {info.get('sections', 0)}")
        print(f"  Imports:      {info.get('imports', 0)}")
        print(f"  Entry Point:  {info.get('entry_point', 'N/A')}")

    # Detection verdict
    print(f"\n{BOLD}Detection Verdict:{RESET}")

    if result['is_packed']:
        status_color = RED if result['packer_type'] != "Unknown Packer" else YELLOW
        print(f"  Status:     {status_color}PACKED{RESET}")
    else:
        print(f"  Status:     {GREEN}NOT PACKED{RESET}")

    print(f"  Packer:     {result['packer_type']}")
    print(f"  Confidence: {result['confidence_score']}%")

    # Suspicious features
    if result['suspicious_features']:
        print(f"\n{YELLOW}Suspicious Features Detected:{RESET}")
        for i, feature in enumerate(result['suspicious_features'], 1):
            print(f"  [{i}] {feature}")
    else:
        print(f"\n{GREEN}No suspicious features detected.{RESET}")

    print("="*70 + "\n")


# ============================================================================
# MAIN ENTRY POINT - FOR TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test entry point with example usage.

    Usage:
        python packer_detector.py <pe_file_path>
        python packer_detector.py sample.exe
    """

    print(f"{'-'*70}")
    print("PE Packer Detector - NKAMG Malware Analysis Tool")
    print(f"{'-'*70}\n")

    if len(sys.argv) < 2:
        print("Usage: python packer_detector.py <pe_file_path>")
        print("\nExample:")
        print("  python packer_detector.py C:\\samples\\malware.exe")
        print("  python packer_detector.py /opt/samples/upx_packed.exe")
        print("\nOptions:")
        print("  --info    Show detailed PE information")
        sys.exit(1)

    file_path = sys.argv[1]

    # Initialize detector
    detector = PEPackerDetector(file_path)

    # Optional: Print detailed PE info for manual analysis
    if "--info" in sys.argv:
        detector.load_pe()
        detector.print_pe_info()

    # Run detection
    print("[*] Analyzing PE file...")
    result = detector.detect()

    # Display results
    print_detection_result(result)

    # Exit code based on detection result
    sys.exit(0 if result['is_packed'] else 1)
