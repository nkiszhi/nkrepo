#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os

hex_string = "0123456789abcdef"
sample_count = 0

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            for l in hex_string:
                folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                list_all = os.listdir(folder)
                for f in list_all:
                    if len(f) == 64:
                        sample_count += 1

print("In total there are " + str(sample_count) + " malware samples.")

