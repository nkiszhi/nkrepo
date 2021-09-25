#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Remove the ".added" suffix of torrent files at
DATA folder """

from __future__ import print_function
import os
import shutil

DIR_DATA = "./DATA/"
list_torrents = []
n_torrents = 0
paths = os.listdir(DIR_DATA)
for i in paths:
    if i.find(".torrent.added") != -1:
        t = i.split('.added')[0] 
        n_torrents = n_torrents + 1
        print("{}: {}".format(n_torrents, t))
        src_file = DIR_DATA+i
        dst_file = DIR_DATA+t
        print(src_file)
        print(dst_file)
        shutil.move(src_file, dst_file)
