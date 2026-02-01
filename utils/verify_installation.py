# -*- coding: utf-8 -*-
"""
NKREPO Installation Verification Script
Checks if all dependencies are properly installed.

Author: NKAMG (Nankai Anti-Malware Group)
"""

import sys
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def print_section(text):
    """Print formatted section."""
    print(f"\n{'-'*70}")
    print(text)
    print(f"{'-'*70}")


def check_dependency(name, import_name=None, version_attr='__version__'):
    """
    Check if a dependency is installed.

    Args:
        name: Display name of package
        import_name: Python import name (defaults to name)
        version_attr: Attribute to get version (default: __version__)

    Returns:
        Tuple of (installed: bool, version: str or None)
    """
    if import_name is None:
        import_name = name.lower().replace('-', '_')

    try:
        module = __import__(import_name)

        # Try to get version
        version = None
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        elif hasattr(module, '__version_info__'):
            version = '.'.join(map(str, module.__version_info__))

        return True, version

    except ImportError:
        return False, None
    except Exception as e:
        return False, str(e)


def check_file_exists(file_path, description):
    """Check if a file exists."""
    path = Path(file_path)
    exists = path.exists()

    if exists:
        size = path.stat().st_size
        print(f"  [OK] {description}: {file_path}")
        print(f"       Size: {size:,} bytes")
    else:
        print(f"  [MISSING] {description}: {file_path}")

    return exists


def main():
    """Main verification function."""
    print_header("NKREPO Installation Verification")
    print("Checking all dependencies and files...")

    results = {
        'core': {},
        'optional': {},
        'files': {}
    }

    # ========================================================================
    # Core Dependencies
    # ========================================================================
    print_section("Core Dependencies (Required)")

    core_deps = [
        ('LIEF', 'lief', '__version__'),
        ('pefile', 'pefile', '__version__'),
        ('yara-python', 'yara', '__version__'),
    ]

    for name, import_name, version_attr in core_deps:
        installed, version = check_dependency(name, import_name, version_attr)

        if installed:
            version_str = f"v{version}" if version else "installed"
            print(f"  [OK] {name:<20} {version_str}")
            results['core'][name] = True
        else:
            print(f"  [ERROR] {name:<20} NOT INSTALLED")
            results['core'][name] = False

    # ========================================================================
    # Optional Dependencies
    # ========================================================================
    print_section("Optional Dependencies (Enhanced Features)")

    optional_deps = [
        ('SSDEEP', 'ssdeep', '__version__'),
        ('TLSH', 'tlsh', None),
        ('tqdm', 'tqdm', '__version__'),
    ]

    for name, import_name, version_attr in optional_deps:
        installed, version = check_dependency(name, import_name, version_attr)

        if installed:
            version_str = f"v{version}" if version else "installed"
            print(f"  [OK] {name:<20} {version_str}")
            results['optional'][name] = True
        else:
            print(f"  [WARN] {name:<20} Not installed (optional)")
            results['optional'][name] = False

    # ========================================================================
    # Python Built-in Libraries
    # ========================================================================
    print_section("Python Built-in Libraries")

    builtin_libs = ['hashlib', 'json', 'csv', 'pathlib', 'argparse', 'urllib']

    for lib in builtin_libs:
        installed, _ = check_dependency(lib, lib, None)

        if installed:
            print(f"  [OK] {lib}")
        else:
            print(f"  [ERROR] {lib} - Python installation corrupted?")

    # ========================================================================
    # Tool Files
    # ========================================================================
    print_section("Tool Files")

    tools = {
        'utils/packer_detector.py': 'LIEF Packer Detector',
        'utils/windows_packer_scanner.py': 'Windows System Scanner',
        'utils/yara_packer_scanner.py': 'YARA+LIEF Scanner',
        'utils/pe_hash_calculator.py': 'PE Hash Calculator',
        'utils/collect_yara_rules.py': 'YARA Rules Collector',
        'utils/check_lief_version.py': 'LIEF Compatibility Checker',
    }

    for file_path, description in tools.items():
        exists = check_file_exists(file_path, description)
        results['files'][description] = exists

    # ========================================================================
    # YARA Rules
    # ========================================================================
    print_section("YARA Rules")

    yara_files = {
        'yara_rules/packers_complete.yar': 'Complete Packer Rules (40+ rules)',
        'yara_rules/custom/upx_custom.yar': 'UPX Custom Rule',
        'yara_rules/custom/vmprotect_custom.yar': 'VMProtect Custom Rule',
    }

    for file_path, description in yara_files.items():
        check_file_exists(file_path, description)

    # ========================================================================
    # Documentation
    # ========================================================================
    print_section("Documentation")

    docs = {
        'INSTALLATION_GUIDE.md': 'Installation Guide',
        'DEPENDENCIES_GUIDE.md': 'Dependencies Guide',
        'utils/PE_HASH_GUIDE.md': 'PE Hash Calculator Guide',
        'yara_rules/YARA_USAGE_GUIDE.md': 'YARA Usage Guide',
    }

    for file_path, description in docs.items():
        check_file_exists(file_path, description)

    # ========================================================================
    # Functional Tests
    # ========================================================================
    print_section("Functional Tests")

    # Test 1: LIEF functionality
    print("\n  Test 1: LIEF PE parsing")
    try:
        import lief
        test_file = "C:/Windows/System32/notepad.exe"
        if Path(test_file).exists():
            binary = lief.parse(test_file)
            if binary and hasattr(binary, 'sections'):
                print(f"    [OK] Successfully parsed {Path(test_file).name}")
                print(f"         Sections: {len(binary.sections)}")
            else:
                print(f"    [WARN] Parsed but unexpected format")
        else:
            print(f"    [SKIP] Test file not found")
    except Exception as e:
        print(f"    [ERROR] LIEF parsing failed: {e}")

    # Test 2: YARA compilation
    print("\n  Test 2: YARA rule compilation")
    try:
        import yara
        rules_file = "yara_rules/packers_complete.yar"
        if Path(rules_file).exists():
            rules = yara.compile(rules_file)
            print(f"    [OK] Successfully compiled {rules_file}")
        else:
            print(f"    [SKIP] YARA rules file not found")
    except Exception as e:
        print(f"    [ERROR] YARA compilation failed: {e}")

    # Test 3: pefile functionality
    print("\n  Test 3: pefile imphash calculation")
    try:
        import pefile
        test_file = "C:/Windows/System32/calc.exe"
        if Path(test_file).exists():
            pe = pefile.PE(test_file)
            imphash = pe.get_imphash()
            print(f"    [OK] Calculated imphash: {imphash}")
        else:
            print(f"    [SKIP] Test file not found")
    except Exception as e:
        print(f"    [WARN] pefile test failed: {e}")

    # Test 4: SSDEEP (optional)
    if results['optional'].get('SSDEEP'):
        print("\n  Test 4: SSDEEP fuzzy hashing")
        try:
            import ssdeep
            test_data = b"test data for ssdeep hashing" * 100
            hash_value = ssdeep.hash(test_data)
            print(f"    [OK] SSDEEP hash: {hash_value[:30]}...")
        except Exception as e:
            print(f"    [WARN] SSDEEP test failed: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("Installation Summary")

    # Core dependencies
    core_installed = sum(1 for v in results['core'].values() if v)
    core_total = len(results['core'])

    print(f"\nCore Dependencies:     {core_installed}/{core_total} installed")
    if core_installed < core_total:
        print("  [ACTION NEEDED] Install missing core dependencies:")
        for name, installed in results['core'].items():
            if not installed:
                print(f"    pip install {name.lower()}")

    # Optional dependencies
    optional_installed = sum(1 for v in results['optional'].values() if v)
    optional_total = len(results['optional'])

    print(f"\nOptional Dependencies: {optional_installed}/{optional_total} installed")
    if optional_installed < optional_total:
        print("  [OPTIONAL] Consider installing for enhanced features:")
        for name, installed in results['optional'].items():
            if not installed:
                print(f"    pip install {name.lower()}")

    # Overall status
    print("\n" + "="*70)

    if core_installed == core_total:
        print("✅ INSTALLATION COMPLETE - All core dependencies installed!")
        print("\nYou can now use:")
        print("  - Packer Detection: python utils/packer_detector.py <file>")
        print("  - YARA Scanning:    python utils/yara_packer_scanner.py -r yara_rules/ -f <file>")
        print("  - Hash Calculator:  python utils/pe_hash_calculator.py <file>")
        print("  - System Scanner:   python utils/windows_packer_scanner.py --preset quick")
    else:
        print("⚠️  INSTALLATION INCOMPLETE")
        print("\nMissing core dependencies. Install with:")
        print("  pip install -r utils/requirements.txt")
        print("\nOr install individually:")
        print("  pip install lief pefile yara-python")

    print("="*70 + "\n")

    # Exit code
    return 0 if core_installed == core_total else 1


if __name__ == '__main__':
    sys.exit(main())
