#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Unzip tar.gz files
The tar.gz files are stored at temp folder.
The extracted samples are stored at samples folder.
"""

from __future__ import print_function
import os

files = os.listdir("temp")
for f in files:
    f = "temp/"+f
    print(f)
    os.system("tar xvzf {} -C samples".format(f)) 
exit()
