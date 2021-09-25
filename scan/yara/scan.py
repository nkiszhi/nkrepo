#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import yara
import os
<<<<<<< HEAD

DIR_SCAN = "/nkrepo/temp_json/"
=======
import argparse

DIR_SCAN = "/home/RaidDisk/new/nkrepo/DATA/"
YARA_RULE = "file_type.yar"
>>>>>>> fa646a3001f087b6718606fc6ad03747df0656ce
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

<<<<<<< HEAD
def scan(rule=YARA_RULE_TEST):
    r = yara.compile(source=rule) # Complie text yara rule
    for f in os.listdir(DIR_SCAN): # Search all files in specified folder
=======
def scan(f_yara=YARA_RULE, d=DIR_SCAN):
    # f_yara is a file containing yara rules
    # d is a directory containing samples to scan
    with open(f_yara, "r") as f: # Open yara file
        r_yara = f.read() # read yara rules
       
    i = 0 
    r = yara.compile(source=r_yara) # Complie text yara rule
    for f in os.listdir(d): # Search all files in specified folder
>>>>>>> fa646a3001f087b6718606fc6ad03747df0656ce
        if len(f) != 64: # Only scan malware samples, not json files
            continue
        f_scan = DIR_SCAN + f # Get the absolute path of malware samples
        #print(f_scan)
        with open(f_scan, "rb") as _f:
            m = r.match(data=_f.read()) # Scan specified file
            if m:
<<<<<<< HEAD
                print("{}:{}".format(f_scan, m[0]))

def main():
    scan(YARA_RULE_TEST) 
=======
                i = i + 1
                print("{}:{}, {}".format(i, m[0], f_scan))

def parse_args():
    parser = argparse.ArgumentParser(description = "Using yara rules to scan.")
    parser.add_argument("-r", "--rule", help="input yara rules", type=str, default=YARA_RULE)
    parser.add_argument("-d", "--dir", help="input scan dir", type=str, default=DIR_SCAN)
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    scan(args.rule, args.dir) 
>>>>>>> fa646a3001f087b6718606fc6ad03747df0656ce
    

if __name__ == "__main__":
    main()
