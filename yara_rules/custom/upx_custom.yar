/*
    YARA Rule for UPX Packer Detection
    Author: NKAMG
    Description: Detects UPX packed executables
*/

rule UPX_Packer : packer {
    meta:
        description = "Detects UPX packed executables"
        author = "NKAMG"
        date = "2026-02-01"
        reference = "https://upx.github.io/"

    strings:
        $upx1 = "UPX0" fullword
        $upx2 = "UPX1" fullword
        $upx3 = "UPX!" fullword
        $upx_sig = { 55 50 58 21 } // UPX! signature

    condition:
        uint16(0) == 0x5A4D and // MZ header
        (any of ($upx*))
}
