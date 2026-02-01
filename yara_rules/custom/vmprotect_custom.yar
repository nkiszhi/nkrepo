/*
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
