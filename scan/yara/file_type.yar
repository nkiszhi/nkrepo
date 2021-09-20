rule IsAPK {
    condition:
        uint32be(0) == 0x504B0304
}

rule IsELF {
    condition:
        uint32(0) == 0x464C457F
}

rule IsPE {
    condition:
        uint16(0) == 0x5A4D and uint32(uint32(0x3C)) == 0x4550
}
