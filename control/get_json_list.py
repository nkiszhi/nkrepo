#!/usr/bin/env python
# -*- coding: utf-8 -*-


""" Get a list of all VirusTotal.com scan results.
"""

import os

hex_string = "0123456789abcdef"
sample_count = 0
json_list = []

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            for l in hex_string:
                folder = "../DATA/sha256/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                folder = os.path.abspath(folder)
                list_all = os.listdir(folder)
                for f in list_all:
                    if f[-5:] == ".json":
                        json_list.append(f)
                        print(f)

with open('json.txt', 'w') as f:
    for i in json_list:
        f.write("%s\n" % i)

print("In total there are " + str(len(json_list)) + " VirusTotal.com scan results. ")

