/*
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
