#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" move virustotal scan results stored in results folder into repo.""" 

from __future__ import print_function
import os
import shutil

VT_RESULTS = "vt_results"
REPO = "DATA"


def mov_vt_results():
    n_mov = 0 # The number of moved json files
    n_del = 0 # The number of deleted json files
    n_json = 0
    json_files = os.listdir(VT_RESULTS)
    for f in json_files:
        if len(f) != 69:
            continue
        src_path = "{}/{}".format(VT_RESULTS, f)
        dst_path = "{}/{}/{}/{}/{}/{}".format(REPO, f[0], f[1], f[2], f[3], f)
        if not os.path.exists(dst_path):
            n_mov = n_mov + 1
            print("{}: {}".format(n_json, f))
            shutil.move(src_path, dst_path)
        else:
            n_del = n_del + 1
            os.remove(src_path)
    print("[o]: {} json files are stored in repo".format(n_mov))
    print("[!]: {} duplicated json files are removed.".format(n_del))

def main():
    mov_vt_results()

if __name__ == "__main__":
    main()
