#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import yara
import os
import argparse
#DIR_SCAN = "/nkrepo/temp_json/"

DIR_SCAN = "../../web/web_search/web_search/samples/"
YARA_RULE = "file_type.yar"
YARA_RULE_TEST = """
rule IsPE {
    condition:
        uint16(0) == 0x5A4D and uint32(uint32(0x3C)) == 0x4550
}
rule IsELF {
    condition:
        uint32(0) == 0x464C457F
}
rule IsAPK {
    condition:
        uint32be(0) == 0x504B0304
}
"""

def scan(rule=YARA_RULE_TEST):
    r = yara.compile(source=rule) # Complie text yara rule
    for f in os.listdir(DIR_SCAN): # Search all files in specified folder
        if len(f) != 64: # Only scan malware samples, not json files
            continue
        f_scan = DIR_SCAN + f # Get the absolute path of malware samples
        #print(f_scan)
        with open(f_scan, "rb") as _f:
            m = r.match(data=_f.read()) # Scan specified file
            if m:
                print("{}:{}".format(f_scan, m[0]))

def main():
    scan(YARA_RULE_TEST) 
    

if __name__ == "__main__":
    main()
