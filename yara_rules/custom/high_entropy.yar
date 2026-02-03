/*
    YARA Rule for High Entropy Sections
    Author: NKAMG
    Description: Detects PE files with high entropy sections (possible packing)
*/

import "pe"
import "math"

rule High_Entropy_Section : suspicious {
    meta:
        description = "Detects PE files with high entropy sections"
        author = "NKAMG"
        date = "2026-02-01"

    condition:
        pe.is_pe and
        for any section in pe.sections : (
            math.entropy(section.raw_data_offset, section.raw_data_size) > 7.0
        )
}
