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
import string
from greet import greet

def is_pe(file_name):
    try:
        pe = pefile.PE(file_name)
        return True
    except:
        return False


# Extract strings from PE file
def extract_string(file_name):

    printable_chars = set(string.printable)
    list_string = []

    with open(file_name, "rb") as f:
        data = f.read()

    str_found = ""
    for char in data:
        char = chr(char)
        if char in printable_chars:
            str_found += char
            continue 
        if len(str_found) >= 4:
            list_string.append(str_found)
            print(str_found)
            str_found = ""
            continue
        str_found = ""

    print(len(list_string))
    for s in list_string:
        print(s)
    return list_string
            

            
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
    print("Suspicious API Functions:")
    get_apialert()
    print("\nSuspicious API Anti-Debug:")
    get_apiantidbg(1)
    print("\nSuspicious Sections:")
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
    else:
        print("It is a PE file")
    printable = set(string.printable)
    print(printable)
    print(len(printable))
    extract_string(f_pe)

if __name__ == "__main__":
    main()
