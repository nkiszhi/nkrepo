#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import os
import argparse

F_INFO = "./info_feat.csv"
FILE_MATCH_RESULT = "./match_result.txt"

def search(year, family, category, platform, scan_result, feature):
    # year is the year in which the sample was found
    # family is the family of the samples
    # category is the category of the samples
    # platform is the platform samples are running on
    # scan_result is the Kaspersky scan result of the samples
    # feature is an opcode ngram string

    print("样本库样本查询条件：")
    print("1. Year：{}".format(year))
    print("2. Family：{}".format(family))
    print("3. Category：{}".format(category))
    print("4. Platform：{}".format(platform))
    print("5. Scan Result：{}".format(scan_result))
    print("6. Sample Feature:{}".format(feature))

    year = year.strip().lower()
    family = family.strip().lower()
    category = category.strip().lower()
    platform = platform.strip().lower()
    scan_result = scan_result.strip().lower()
    feature = feature.strip().lower()

    with open(F_INFO, "r") as info:
        lines = info.readlines()
        lines = [x.strip() for x in lines]

    list_match_result = [] 

    n = 0

    for l in lines:
        (md5, sha256, s_category, s_platform, s_family, s_scan_result, s_year, feat1, feat2, feat3, feat4, feat5) = l.strip().split(",")
        md5 = md5.strip().lower()
        sha256 = sha256.strip().lower()
        s_category = s_category.strip().lower()
        s_platform = s_platform.strip().lower()
        s_family = s_family.strip().lower()
        s_scan_result = s_scan_result.strip().lower()
        s_year = s_year.strip().lower()
        feat1 = feat1.strip().lower()
        feat2 = feat2.strip().lower()
        feat3 = feat3.strip().lower()
        feat4 = feat4.strip().lower()
        feat5 = feat5.strip().lower()
        list_feat = [feat1, feat2, feat3, feat4, feat5]

        # Search year
        if year:
            #print(y)
            #print(year)
            if year != s_year:
                #print("Not match!")
                continue
            else:
                #print("Year match: {}".format(y))
                pass
        # Search family
        if family:
            #print(f)
            #print(family)
            if family != s_family:
                continue
            else:
                #print("Family match: {}".format(f))
                pass
        # Search category
        if category:
            #print(c)
            #print(category)
            if c != category:
                continue
            else:
                #print("Category match: {}".format(c))
                pass
        # Search Kaspersky scan result
        if scan_result:
            #print(s)
            #print(scan_result)
            if scan_result != s_scan_result:
                continue
            else:
                #print("Scan result match: {}".format(s))
                pass
        if platform:
            #print(s)
            #print(scan_result)
            if platform != s_platform:
                continue
            else:
                #print("Platform match: {}".format(s))
                pass

        if feature:
            if feature not in list_feat:
                continue

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
    parser.add_argument("-feat", "--feature", help="Search by opcode ngram feature", type=str, default="") 
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    search(args.year, args.family, args.category, args.platform, args.scan_result, args.feature) 
    

if __name__ == "__main__":
    main()
