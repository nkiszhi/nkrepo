#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

hex_string = "0123456789abcdef"
sample_count = 0

for i in hex_string:
    for j in hex_string:
        for k in hex_string:
            for l in hex_string:
                folder = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/"
                #print folder
                #print len(os.listdir(folder))
                sample_count += len(os.listdir(folder))

print "In total there are {} malware samples".format(sample_count)


                
