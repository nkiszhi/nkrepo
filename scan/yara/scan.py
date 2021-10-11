#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__contact__ = "zwang@nankai.edu.cn"

import yara
import os
import argparse
#DIR_SCAN = "/nkrepo/temp_json/"

DIR_SCAN = "../../web/samples/"
FILE_YARA = "file_type.yar"
FILE_RESULT = "scan_result.txt"
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

    n_sample = 0
    n_match = 0
    list_match = []

    r = yara.compile(source=rule) # Complie text yara rule
    for f in os.listdir(DIR_SCAN): # Search all files in specified folder
        n_sample = n_sample + 1
        if len(f) != 64: # Only scan malware samples, not json files
            continue
        f_scan = DIR_SCAN + f # Get the absolute path of malware samples
        #print(f_scan)
        with open(f_scan, "rb") as _f:
            m = r.match(data=_f.read()) # Scan specified file
            if m:
                n_match = n_match + 1
                match_info = "{} --> {}".format(f_scan, m[0])
                print(match_info)
                list_match.append(match_info)

    print("\n共扫描了{}个样本文件，Yara规则匹配了{}样本文件".format(n_sample, n_match))
    with open(FILE_RESULT, "w") as f:
        for i in list_match:
            f.write(i + "\n")

def parseargs():
    parser = argparse.ArgumentParser(description = "Scan samples with yara rules")
    parser.add_argument("-f", "--file_yara", help="input a file containing yara rules", type=str, default=FILE_YARA, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    f_yara = args.file_yara
    if not os.path.exists(f_yara):
        print("The yara file is not exist: {}".format(f_yara))
    with open(f_yara, "r") as f:
        yara_rule = f.read()
    scan(yara_rule) 
    

if __name__ == "__main__":
    main()
