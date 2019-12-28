#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all samples in the Repo."""

from __future__ import print_function
import os

hex_string = "0123456789abcdef"
n_sample = 0
n_json = 0

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            for l in hex_string:
                folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                list_all = os.listdir(folder)
                for f in list_all:
                    if len(f) == 64:
                        n_sample += 1
                    if f[-5:] == ".json":
                        n_json += 1

print("There are {} malware samples in the repo.".format(n_sample))
print("{} samples are labeled by VirusTotal.com.".format(n_json))

