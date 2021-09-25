#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import os
import argparse

F_INFO = "./info.csv"
FILE_MATCH_RESULT = "./match_result.txt"

def search(y, f, c, p, s):
    # y is the year in which the sample was found
    # f is the family of the samples
    # c is the category of the samples
    # p is the platform samples are running on
    # s is the Kaspersky scan result of the samples

    print("样本库样本查询条件：")
    print("1. Year：{}".format(y))
    print("2. Family：{}".format(f))
    print("3. Category：{}".format(c))
    print("4. Platform：{}".format(p))
    print("5. Scan Result：{}".format(s))

    f = f.strip()
    c = c.strip()
    s = s.strip()
    p = p.strip()

    with open(F_INFO, "r") as info:
        lines = info.readlines()

    list_match_result = [] 

    n = 0

    for l in lines:
        (sha256, category, platform, family, scan_result, year) = l.strip().split(",")
        #print(sha256)
        #print(category)
        #print(platform)
        #print(family)
        #print(scan_result)
        #print(year)
        sha256 = sha256.strip()
        category = category.strip()
        platform = platform.strip()
        family = family.strip()
        scan_result = scan_result.strip()
        year = year.strip()
        #print(sha256)
        #print(category)
        #print(platform)
        #print(family)
        #print(scan_result)
        #print(year)

        # Search year
        if y:
            #print(y)
            #print(year)
            if y != year:
                #print("Not match!")
                continue
            else:
                #print("Year match: {}".format(y))
                pass
        # Search family
        if f:
            #print(f)
            #print(family)
            if f != family:
                continue
            else:
                #print("Family match: {}".format(f))
                pass
        # Search category
        if c:
            #print(c)
            #print(category)
            if c != category:
                continue
            else:
                #print("Category match: {}".format(c))
                pass
        # Search Kaspersky scan result
        if s:
            #print(s)
            #print(scan_result)
            if s != scan_result:
                continue
            else:
                #print("Scan result match: {}".format(s))
                pass
        if p:
            #print(s)
            #print(scan_result)
            if p != platform:
                continue
            else:
                #print("Platform match: {}".format(s))
                pass

        n = n + 1
        print("{} Match: {}".format(n, l))
        list_match_result.append(sha256)

    with open(FILE_MATCH_RESULT, "w") as f:
        for r in list_match_result:
            #print(r)
            f.write("{}\n".format(r))




def parse_args():
    parser = argparse.ArgumentParser(description = "Search samples.")
    parser.add_argument("-y", "--year", help="Search by time", type=str, default="")
    parser.add_argument("-f", "--family", help="Search by familly", type=str, default="")
    parser.add_argument("-c", "--category", help="Search by malware class", type=str, default="")
    parser.add_argument("-p", "--platform", help="Search by malware running platform", type=str, default="")
    parser.add_argument("-s", "--scan_result", help="Search by kaspersky scan result", type=str, default="")
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    search(args.year, args.family, args.category, args.platform, args.scan_result) 
    

if __name__ == "__main__":
    main()
