#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = "Copyright (c) 2016 NKAMG"
__license__ = "GPL"
__contact__ = "zwang@nankai.edu.cn"

import sys
import os
import pefile
import peutils
import argparse
from greet import greet


def is_pe(file_name):
    try:
        pe = pefile.PE(file_name)
        return True
    except:
        return False


# Entropy
def get_sectionsalert(filename):
    pe = pefile.PE(filename)
    array = []
    for section in pe.sections:
        section.get_entropy()
        if section.SizeOfRawData == 0 or (section.get_entropy() > 0 and section.get_entropy() < 1) or section.get_entropy() > 7:
            sc   = section.Name
            md5  = section.get_hash_md5()
            sha1 = section.get_hash_sha1()
            array.append([sc, md5, sha1])
    if array:
        return array
    else:
        return False

def get_suspicious():
    print "Suspicious API Functions:"
    get_apialert()
    print "\nSuspicious API Anti-Debug:"
    get_apiantidbg(1)
    print "\nSuspicious Sections:"
    get_sectionsalert()

def parseargs():
    parser = argparse.ArgumentParser(description = "Scan PE file.")
    parser.add_argument("-f", "--file", help="The PE file to scan.", type=str)
    args = parser.parse_args()
    return args

def main():
    greet()
    args = parseargs()
    f_pe = args.file
    print(f_pe)
    if not is_pe(f_pe):
        print("[!] It is not a PE file.")

if __name__ == "__main__":
    main()
