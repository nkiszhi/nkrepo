/*
    YARA Rule for Generic Packer Detection
    Author: NKAMG
    Description: Detects common packer indicators
*/

import "pe"

rule Generic_Packer_Indicators : packer {
    meta:
        description = "Detects generic packer indicators"
        author = "NKAMG"
        date = "2026-02-01"

    condition:
        pe.is_pe and
        (
            // Very few imports (< 5 functions)
            pe.number_of_imports < 5 or

            // Missing .text section
            not for any section in pe.sections : (
                section.name contains ".text"
            ) or

            // Entry point not in .text
            not for any section in pe.sections : (
                section.name contains ".text" and
                pe.entry_point >= section.virtual_address and
                pe.entry_point < (section.virtual_address + section.virtual_size)
            )
        )
}
