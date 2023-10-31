rule exe32: PE
{
    meta:
        description = "PE 32-bit executable file"

    condition:
        uint16(0) == 0x5a4d and  // 'MZ' signature
        uint32(uint32(0x3c)) == 0x4550 and // 'PE\0\0' signature
        uint16(uint32(0x3c)+0x18) == 0x010b and // Magic number is PE32 0x010b
		(uint16(uint32(0x3C)+0x16) & 0x2000) != 0x2000 // Characteristic does not containt IMAGE_FILE_DLL 0x2000
}

rule exe64: PE
{
    meta:
        description = "PE 64bit executable file"

    condition:
        uint16(0) == 0x5a4d and // 'MZ' header
        uint32(uint32(0x3c)) == 0x4550 and // 'PE\0\0' signature
        uint16(uint32(0x3c)+0x18) == 0x020b and // Magic number is PE32+ 0x020b
		(uint16(uint32(0x3C)+0x16) & 0x2000) != 0x2000 // Characteristic does not containt IMAGE_FILE_DLL 0x2000
}


rule dll32: PE
{
	condition:
		uint16(0) == 0x5A4D and // 'MZ' header
        uint32(uint32(0x3c)) == 0x4550 and // 'PE\0\0' signature
        uint16(uint32(0x3c)+0x18) == 0x010b and // Magic number is PE32 0x010b
		(uint16(uint32(0x3C)+0x16) & 0x2000) == 0x2000 // Characteristic containts IMAGE_FILE_DLL 0x2000

}

rule dll64: PE
{
	condition:
		uint16(0) == 0x5A4D and // 'MZ' header
        uint32(uint32(0x3c)) == 0x4550 and // 'PE\0\0' signature
        uint16(uint32(0x3c)+0x18) == 0x020b and // Magic number is PE32+ 0x020b
		(uint16(uint32(0x3C)+0x16) & 0x2000) == 0x2000 // Characteristic containts IMAGE_FILE_DLL 0x2000
}

rule shell
{
    meta:
        description = "Shell scripts"

    strings:
        $s1 = "#!/bin/sh"
        $s2 = "#!/bin/bash"
        $s3 = "#!/bin/zsh"
        $s4 = "#!/bin/csh"
        $s5 = "#!/bin/tcsh"

    condition:
        $s1 at 0 or
        $s2 at 0 or
        $s3 at 0 or
        $s4 at 0 or
        $s5 at 0
}

rule elf
{
    meta:
        description = "ELF file"

    condition:
        uint32(0) == 0x464c457f // Generic ELF header '\7fELF'
}

rule zip
{
    meta:
        description = "ZIP file"

    strings:
        $sig = {50 4b 03 04 14 00} // ZIP header 'PK\03\04\14\00'

    condition:
        $sig at 0
}

rule pdf
{
    meta:
        description = "PDF file"

    condition:
        uint32(0) == 0x46445025 // PDF header '%PDF'
}

rule office97
{
    meta:
        description = "Microsoft Office 97, doc, dot, pps, xla, xls"

    strings:
        $sig = {d0 cf 11 e0 a1 b1 1a e1} // Header signature

    condition:
        $sig at 0
}

rule office
{
    meta:
        description = "Microsoft Office Open XML Format, docx, pptx, xlsx"

    strings:
        $sig = {50 4b 03 04 14 00 06 00} // Header 'PK\03\04\14\00\06\00'

    condition:
        $sig at 0
        
}

rule jar
{
    meta:
        description = "JAR Java archive"

    strings:
        $sig = {50 4B 03 04 14 00 08 00 08 00}

    condition:
        $sig at 0
}

rule rar
{
    meta:
        description = "RAR compressed archive file"

    strings:
        $sig1 = {52 61 72 21 1a 07 00} // RAR (v4.x) compressed archive file, 'RAR!\1a\07\00'
        $sig2 = {52 61 72 21 1a 07 01 00} //    RAR (v5) compressed archive file, 'RAR!\1a\07\01\00'

    condition:
        $sig1 at 0 or
        $sig2 at 0
}


rule png
{
    meta:
        description = "PNG file"

    strings:
        $sig = {89 50 4e 47} // PNG signature '.PNG'

    condition:
        $sig at 0
}
