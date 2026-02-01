# -*- coding: utf-8 -*-
"""
LIEF Version Checker and Compatibility Tester
Helps diagnose LIEF API compatibility issues.

Author: NKAMG (Nankai Anti-Malware Group)
"""

import sys


def check_lief_installation():
    """
    Check if LIEF is installed and display version information.
    """
    print("="*70)
    print("LIEF Library Compatibility Checker")
    print("="*70)

    # Check if LIEF is installed
    try:
        import lief
        print(f"\n[OK] LIEF is installed")
        print(f"  Version: {lief.__version__}")
    except ImportError:
        print("\n[ERROR] LIEF is NOT installed")
        print("\nTo install LIEF, run:")
        print("  pip install lief")
        return False

    # Check LIEF.PE module
    try:
        import lief.PE
        print(f"[OK] LIEF.PE module is available")
    except ImportError:
        print(f"[ERROR] LIEF.PE module is NOT available")
        return False

    return True


def check_machine_types():
    """
    Check how to access MACHINE_TYPES in the current LIEF version.
    """
    print(f"\n{'-'*70}")
    print("Machine Types API Check")
    print(f"{'-'*70}")

    import lief

    # Method 1: lief.PE.MACHINE_TYPES
    try:
        from lief.PE import MACHINE_TYPES
        print("[OK] Can import MACHINE_TYPES from lief.PE")
        print(f"  AMD64 value: {MACHINE_TYPES.AMD64}")
        print(f"  I386 value: {MACHINE_TYPES.I386}")
        return "lief.PE.MACHINE_TYPES"
    except (ImportError, AttributeError) as e:
        print(f"[ERROR] Cannot import MACHINE_TYPES from lief.PE")
        print(f"  Error: {e}")

    # Method 2: Check if we can use raw values
    print("\n  Fallback: Using raw integer values")
    print(f"    AMD64 (64-bit): 0x8664 = {0x8664}")
    print(f"    I386 (32-bit):  0x014c = {0x014c}")

    return "raw_values"


def check_section_characteristics():
    """
    Check how to access SECTION_CHARACTERISTICS in the current LIEF version.
    """
    print(f"\n{'-'*70}")
    print("Section Characteristics API Check")
    print(f"{'-'*70}")

    import lief

    # Method 1: lief.PE.SECTION_CHARACTERISTICS
    try:
        from lief.PE import SECTION_CHARACTERISTICS
        print("[OK] Can import SECTION_CHARACTERISTICS from lief.PE")
        print(f"  MEM_READ: {hex(SECTION_CHARACTERISTICS.MEM_READ)}")
        print(f"  MEM_WRITE: {hex(SECTION_CHARACTERISTICS.MEM_WRITE)}")
        print(f"  MEM_EXECUTE: {hex(SECTION_CHARACTERISTICS.MEM_EXECUTE)}")
        return "lief.PE.SECTION_CHARACTERISTICS"
    except (ImportError, AttributeError) as e:
        print(f"[ERROR] Cannot import SECTION_CHARACTERISTICS from lief.PE")
        print(f"  Error: {e}")

    # Method 2: Check if we can use raw values
    print("\n  Fallback: Using raw integer values")
    print(f"    MEM_READ:    0x40000000")
    print(f"    MEM_WRITE:   0x80000000")
    print(f"    MEM_EXECUTE: 0x20000000")

    return "raw_values"


def test_parse_sample():
    """
    Test parsing a sample PE file if provided.
    """
    if len(sys.argv) < 2:
        print(f"\n{'-'*70}")
        print("PE File Parsing Test")
        print(f"{'-'*70}")
        print("No test file provided. Skipping parse test.")
        print("\nTo test with a PE file, run:")
        print("  python check_lief_version.py <pe_file_path>")
        return

    import lief
    from pathlib import Path

    file_path = Path(sys.argv[1])

    print(f"\n{'-'*70}")
    print("PE File Parsing Test")
    print(f"{'-'*70}")
    print(f"File: {file_path.name}")

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    try:
        binary = lief.parse(str(file_path))

        if binary is None:
            print("[ERROR] Failed to parse file (returned None)")
            return

        if not isinstance(binary, lief.PE.Binary):
            print(f"[ERROR] Not a PE file (type: {type(binary).__name__})")
            return

        print(f"[OK] Successfully parsed PE file")
        print(f"\nBasic Information:")
        print(f"  Sections: {len(binary.sections)}")

        # Test machine type detection
        machine = binary.header.machine
        print(f"  Machine type: {machine} ({type(machine).__name__})")

        # Try to determine architecture
        try:
            from lief.PE import MACHINE_TYPES
            if machine == MACHINE_TYPES.AMD64:
                arch = "64-bit"
            elif machine == MACHINE_TYPES.I386:
                arch = "32-bit"
            else:
                arch = "Unknown"
        except (ImportError, AttributeError):
            # Use raw values
            if isinstance(machine, int):
                if machine == 0x8664:
                    arch = "64-bit"
                elif machine == 0x014c:
                    arch = "32-bit"
                else:
                    arch = f"Unknown (0x{machine:04x})"
            else:
                machine_str = str(machine).upper()
                if 'AMD64' in machine_str or 'X64' in machine_str:
                    arch = "64-bit"
                elif 'I386' in machine_str or 'X86' in machine_str:
                    arch = "32-bit"
                else:
                    arch = f"Unknown ({machine})"

        print(f"  Architecture: {arch}")

        # Test section characteristics
        if binary.sections:
            sec = binary.sections[0]
            print(f"\nFirst section: {sec.name}")
            print(f"  Characteristics: 0x{sec.characteristics:08x}")

            try:
                from lief.PE import SECTION_CHARACTERISTICS
                is_exec = bool(sec.characteristics & SECTION_CHARACTERISTICS.MEM_EXECUTE)
                print(f"  Executable: {is_exec} (using SECTION_CHARACTERISTICS)")
            except (ImportError, AttributeError):
                is_exec = bool(sec.characteristics & 0x20000000)
                print(f"  Executable: {is_exec} (using raw value)")

    except Exception as e:
        print(f"[ERROR] Error parsing file: {e}")
        import traceback
        traceback.print_exc()


def print_recommendations():
    """
    Print recommendations based on findings.
    """
    print(f"\n{'='*70}")
    print("Recommendations")
    print(f"{'='*70}")
    print("\nThe packer_detector.py has been updated to handle both:")
    print("  1. Modern LIEF versions (using enums)")
    print("  2. Older LIEF versions (using raw integer values)")
    print("\nIf you still encounter issues:")
    print("  1. Update LIEF to the latest version:")
    print("     pip install --upgrade lief")
    print("  2. Or reinstall LIEF:")
    print("     pip uninstall lief")
    print("     pip install lief")
    print("\nSupported LIEF versions: 0.9.0 and above")


def main():
    """
    Main function to run all compatibility checks.
    """
    if not check_lief_installation():
        sys.exit(1)

    check_machine_types()
    check_section_characteristics()
    test_parse_sample()
    print_recommendations()

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
