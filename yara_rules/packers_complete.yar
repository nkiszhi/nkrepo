/*
   YARA Rules for Packer Detection
   Comprehensive collection for PE malware analysis

   Author: NKAMG (Nankai Anti-Malware Group)
   Date: 2026-02-01
   Description: Detects common packers, protectors, and obfuscators
*/

import "pe"
import "math"

// ==================== UPX PACKER ====================

rule UPX_Packer {
    meta:
        description = "Detects UPX packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        reference = "https://upx.github.io/"
        severity = "medium"

    strings:
        $upx_sig1 = { 55 50 58 21 }  // UPX! signature
        $upx_sig2 = { 55 50 58 30 }  // UPX0
        $upx_str1 = "UPX0" fullword
        $upx_str2 = "UPX1" fullword
        $upx_str3 = "UPX2" fullword
        $upx_str4 = "UPX!" fullword

    condition:
        uint16(0) == 0x5A4D and  // MZ header
        (any of ($upx_sig*) or 2 of ($upx_str*))
}

rule UPX_Unpacked {
    meta:
        description = "Detects unpacked UPX file (contains overlay)"
        author = "NKAMG"

    strings:
        $overlay = { 55 50 58 21 [8] FF D5 80 3D [4] 00 74 }

    condition:
        uint16(0) == 0x5A4D and $overlay
}

// ==================== VMPROTECT ====================

rule VMProtect_Packer {
    meta:
        description = "Detects VMProtect packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        reference = "https://vmpsoft.com/"
        severity = "high"

    strings:
        $vmp_sec1 = ".vmp0" fullword
        $vmp_sec2 = ".vmp1" fullword
        $vmp_sec3 = ".vmp2" fullword
        $vmp_str = ".VMPROT" fullword
        $vmp_sig = { 9C 60 68 [4] B8 [4] 50 45 4D 41 52 4B }

    condition:
        uint16(0) == 0x5A4D and
        (any of ($vmp_sec*) or $vmp_str or $vmp_sig)
}

rule VMProtect_v2 {
    meta:
        description = "Detects VMProtect 2.x"
        author = "NKAMG"

    strings:
        $vmp2_1 = { 8B 4D ?? 8B 45 ?? 89 08 E8 }
        $vmp2_2 = { 8B 45 ?? 33 C9 83 E8 04 83 F8 28 }

    condition:
        uint16(0) == 0x5A4D and all of them
}

// ==================== THEMIDA / WINLICENSE ====================

rule Themida_WinLicense {
    meta:
        description = "Detects Themida/WinLicense protected executables"
        author = "NKAMG"
        date = "2026-02-01"
        reference = "https://www.oreans.com/"
        severity = "high"

    strings:
        $themida_sec1 = ".themida" fullword
        $themida_sec2 = ".tmd" fullword
        $themida_sec3 = ".tmc" fullword
        $winlic_sec = ".winlice" fullword
        $oreans = "Oreans" nocase
        $themida_str = "Themida" nocase

    condition:
        uint16(0) == 0x5A4D and
        (any of ($themida_sec*) or $winlic_sec or $oreans or $themida_str)
}

rule Themida_v2 {
    meta:
        description = "Detects Themida 2.x specific patterns"
        author = "NKAMG"

    strings:
        $tm2_1 = { B8 [4] 50 64 FF 35 [4] 64 89 25 [4] 33 C0 89 44 24 }
        $tm2_2 = { 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 CC }

    condition:
        uint16(0) == 0x5A4D and any of them
}

// ==================== ASPACK ====================

rule ASPack_Packer {
    meta:
        description = "Detects ASPack packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $aspack_sec1 = ".aspack" fullword
        $aspack_sec2 = ".adata" fullword
        $aspack_str = "ASPack" nocase
        $aspack_sig = { 60 E8 00 00 00 00 5D 81 ED [4] B8 [4] 03 C5 }

    condition:
        uint16(0) == 0x5A4D and
        (any of ($aspack_sec*) or $aspack_str or $aspack_sig)
}

// ==================== ENIGMA PROTECTOR ====================

rule Enigma_Protector {
    meta:
        description = "Detects Enigma Protector"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $enigma_sec1 = ".enigma1" fullword
        $enigma_sec2 = ".enigma2" fullword
        $enigma_str = "EnigmaProtector" nocase
        $enigma_sig = { 60 E8 00 00 00 00 5D 50 51 EB 0F }

    condition:
        uint16(0) == 0x5A4D and
        (any of ($enigma_sec*) or $enigma_str or $enigma_sig)
}

// ==================== PECOMPACT ====================

rule PECompact_Packer {
    meta:
        description = "Detects PECompact packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $pec_sec1 = "PEC2" fullword
        $pec_sec2 = "PECompact2" fullword
        $pec_sec3 = "pec1" fullword
        $pec_sec4 = "pec2" fullword
        $pec_sig = { B8 [4] 50 64 FF 35 [4] 64 89 25 [4] 33 C0 89 08 50 45 43 32 }

    condition:
        uint16(0) == 0x5A4D and
        (any of ($pec_sec*) or $pec_sig)
}

// ==================== NSPACK ====================

rule NSPack_Packer {
    meta:
        description = "Detects NSPack packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $nsp_sec1 = ".nsp0" fullword
        $nsp_sec2 = ".nsp1" fullword
        $nsp_sec3 = ".nsp2" fullword
        $nsp_sec4 = "nsp0" fullword
        $nsp_sec5 = "nsp1" fullword
        $nsp_sig = { 9C 60 E8 00 00 00 00 5D B8 [4] 2D [4] }

    condition:
        uint16(0) == 0x5A4D and
        (any of ($nsp_sec*) or $nsp_sig)
}

// ==================== MPRESS ====================

rule MPRESS_Packer {
    meta:
        description = "Detects MPRESS packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $mpress_sec1 = ".MPRESS" fullword
        $mpress_sec2 = ".MPRESS1" fullword
        $mpress_sec3 = ".MPRESS2" fullword
        $mpress_sig = { 60 E8 00 00 00 00 58 05 [4] 8B 30 03 F0 }

    condition:
        uint16(0) == 0x5A4D and
        (any of ($mpress_sec*) or $mpress_sig)
}

// ==================== ARMADILLO ====================

rule Armadillo_Packer {
    meta:
        description = "Detects Armadillo protected executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $arma_sec = ".arma" fullword
        $arma_str = "Armadillo" nocase
        $arma_sig = { 55 8B EC 6A FF 68 [4] 68 [4] 64 A1 00 00 00 00 50 }

    condition:
        uint16(0) == 0x5A4D and
        (any of them)
}

// ==================== OBSIDIUM ====================

rule Obsidium_Packer {
    meta:
        description = "Detects Obsidium protected executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "high"

    strings:
        $obs_sec = ".obsidi" fullword
        $obs_str = "Obsidium" nocase
        $obs_sig = { EB 02 [2] E8 [4] 5D 81 ED }

    condition:
        uint16(0) == 0x5A4D and
        (any of them)
}

// ==================== PESPIN ====================

rule PESpin_Packer {
    meta:
        description = "Detects PESpin packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $pespin_sec = ".spin" fullword
        $pespin_str = "PESpin" nocase
        $pespin_sig = { EB 01 68 60 E8 00 00 00 00 8B 1C 24 83 C3 12 }

    condition:
        uint16(0) == 0x5A4D and
        (any of them)
}

// ==================== PETITE ====================

rule Petite_Packer {
    meta:
        description = "Detects Petite packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $petite_sec = ".petite" fullword
        $petite_sig = { B8 [4] 66 9C 60 50 }

    condition:
        uint16(0) == 0x5A4D and
        (any of them)
}

// ==================== FSG ====================

rule FSG_Packer {
    meta:
        description = "Detects FSG packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $fsg_sec = ".fsg" fullword
        $fsg_str = "FSG!" fullword
        $fsg_sig = { 87 25 [4] 61 94 55 A4 B6 80 FF 13 }

    condition:
        uint16(0) == 0x5A4D and
        (any of them)
}

// ==================== YODA PROTECTOR ====================

rule Yoda_Protector {
    meta:
        description = "Detects Yoda Protector/Crypter"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    strings:
        $yoda_sec1 = ".yP" fullword
        $yoda_sec2 = ".y0da" fullword
        $yoda_sec3 = "yC" fullword
        $yoda_sig = { 55 8B EC 53 56 57 60 E8 00 00 00 00 5D 81 ED }

    condition:
        uint16(0) == 0x5A4D and
        (any of them)
}

// ==================== .NET PROTECTORS ====================

rule Dotfuscator {
    meta:
        description = "Detects Dotfuscator obfuscated .NET assemblies"
        author = "NKAMG"
        date = "2026-02-01"

    strings:
        $dotfus = "DotfuscatorAttribute" ascii
        $dotfus2 = "PreEmptive" ascii

    condition:
        uint16(0) == 0x5A4D and
        uint32(uint32(0x3C)) == 0x00004550 and // PE signature
        any of them
}

rule ConfuserEx {
    meta:
        description = "Detects ConfuserEx protected .NET assemblies"
        author = "NKAMG"
        date = "2026-02-01"

    strings:
        $confuser = "ConfusedByAttribute" ascii
        $confuser2 = "Confuser.Runtime" ascii

    condition:
        uint16(0) == 0x5A4D and
        any of them
}

// ==================== GENERIC PACKER INDICATORS ====================

rule Generic_Packer_High_Entropy {
    meta:
        description = "Detects PE files with suspiciously high entropy (possible packing)"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "info"

    condition:
        uint16(0) == 0x5A4D and
        pe.is_pe and
        for any section in pe.sections : (
            math.entropy(section.raw_data_offset, section.raw_data_size) > 7.2 and
            section.characteristics & pe.SECTION_EXECUTABLE
        )
}

rule Generic_Packer_Few_Imports {
    meta:
        description = "Detects PE files with very few imports (packer indicator)"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "info"

    condition:
        uint16(0) == 0x5A4D and
        pe.is_pe and
        pe.number_of_imports < 5
}

rule Generic_Packer_Suspicious_Sections {
    meta:
        description = "Detects PE files with suspicious section names"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "info"

    condition:
        uint16(0) == 0x5A4D and
        pe.is_pe and
        for any section in pe.sections : (
            section.name contains ".packed" or
            section.name contains ".crypted" or
            section.name contains ".protect" or
            section.name contains ".stub" or
            section.name contains ".data1"
        )
}

rule Generic_Packer_Missing_Text_Section {
    meta:
        description = "Detects PE files without .text section (common in packed files)"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "info"

    condition:
        uint16(0) == 0x5A4D and
        pe.is_pe and
        not for any section in pe.sections : (
            section.name contains ".text"
        )
}

rule Generic_Packer_RWX_Section {
    meta:
        description = "Detects PE files with Read-Write-Execute sections (packer indicator)"
        author = "NKAMG"
        date = "2026-02-01"
        severity = "medium"

    condition:
        uint16(0) == 0x5A4D and
        pe.is_pe and
        for any section in pe.sections : (
            (section.characteristics & pe.SECTION_MEM_READ) and
            (section.characteristics & pe.SECTION_MEM_WRITE) and
            (section.characteristics & pe.SECTION_MEM_EXECUTE)
        )
}

// ==================== ADDITIONAL PACKERS ====================

rule ExeStealth_Packer {
    meta:
        description = "Detects ExeStealth packed executables"
        author = "NKAMG"

    strings:
        $exestlth = "ExeStealth" nocase
        $exes_sec = ".edata" fullword

    condition:
        uint16(0) == 0x5A4D and any of them
}

rule PKLite_Packer {
    meta:
        description = "Detects PKLite packed executables"
        author = "NKAMG"

    strings:
        $pkl_sec1 = "pklstb" fullword
        $pkl_sec2 = ".pklstb" fullword
        $pkl_sig = "PKLITE" nocase

    condition:
        uint16(0) == 0x5A4D and any of them
}

rule WWPack_Packer {
    meta:
        description = "Detects WWPack32 packed executables"
        author = "NKAMG"

    strings:
        $wwp_sec1 = ".WWP32" fullword
        $wwp_sec2 = "WWP32" fullword

    condition:
        uint16(0) == 0x5A4D and any of them
}

rule NeoLite_Packer {
    meta:
        description = "Detects NeoLite packed executables"
        author = "NKAMG"

    strings:
        $neo_sec1 = ".neolite" fullword
        $neo_sec2 = ".neolit" fullword

    condition:
        uint16(0) == 0x5A4D and any of them
}
