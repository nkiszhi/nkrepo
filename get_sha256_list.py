
#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Get a SHA256 list of all samples in the repo
"""

from __future__ import print_function
import os

hex_string = "0123456789abcdef"
sample_count = 0
list_samples = []

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            for l in hex_string:
                folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                list_all = os.listdir(folder)
                for f in list_all:
                    if len(f) == 64:
                        list_samples.append(f)
f_sha256 = "sha256.txt"
with open(f_sha256, 'w') as f:
    for item in list_samples:
        f.write("%s\n" % item)
print("In total there are " + str(len(list_samples)) + " malware samples in the repo.")

