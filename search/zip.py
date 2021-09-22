#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import argparse
import tarfile
import os
import shutil

DIR_SAMPLE = "./temp_folder/" # folder to store samples for downloading
DIR_SRC = "./samples/"        # folder containing all samples
FILE_TAR = "search.tgz"       # tgz file name
FILE_SHA256 = "match_result.txt"


# Copy all samples in the sha256 list from DATA repo into a temp folder DIR_TEMP
def copy_sample(file_sha256):
    # file_sha256 is a file containing a sha256 list.
    # DIR_TEMP is a temp folder to store samples
    print("\n[i] Copying samples.")

    n = 0
    if not os.path.exists(DIR_SAMPLE): # make sure DIR_DST is existed
        os.mkdir(DIR_SAMPLE)

    with open(file_sha256, "r") as f: # read sha256 values
        list_sha256 = f.readlines()

    for sha256 in list_sha256:
        sha256 = sha256.strip() # remove new line character
        file_src = DIR_SRC + sha256
        if not os.path.exists(file_src):
            continue
        file_dst = DIR_SAMPLE + sha256
        shutil.copyfile(file_src, file_dst) # copy samples
        n = n + 1
        print("{}: {}".format(n, sha256))


def make_tarfile(output_filename, source_dir):
    print("\n[i] Zipping samples.")
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    #print(source_dir)
    shutil.rmtree(source_dir) # remove the temp folder 

def parse_args():
    parser = argparse.ArgumentParser(description = "Generate a tar file containing samples.")
    parser.add_argument("-s", "--sha256_filename", help="The file containing sha256 values.", type=str, default=FILE_SHA256)
    parser.add_argument("-d", "--source_dir", help="The directory containint samples.", type=str, default=DIR_SAMPLE)
    parser.add_argument("-f", "--output_filename", help="The output tar file.", type=str, default=FILE_TAR)
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    copy_sample(args.sha256_filename)
    make_tarfile(args.output_filename, args.source_dir) 
    print("\n[i] Zipped file \"{}\" has been generated.\n".format(args.output_filename))
    

if __name__ == "__main__":
    main()
