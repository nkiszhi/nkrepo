#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

from torrentool.api import Torrent
import os
import argparse
import shutil

DIR_SAMPLES = "./samples/"
FILE_TORRENT = "search.torrent" # the torrent file of samples to download
FILE_SHA256 = "./match_result.txt" # contains a list of sha256.
DIR_DST = "./temp_folder/"  # a temp folder to store samples
DIR_SRC = "./samples/"       # a folder containing all samples

# Copy all samples in the sha256 list from DATA repo into a temp folder DIR_TEMP
def copy_sample(file_sha256):
    # file_sha256 is a file containing a sha256 list. 
    # DIR_TEMP is a temp folder to store samples
    print("\n[i] Copying samples")

    n = 0
    if os.path.exists(DIR_DST): # make sure DIR_DST is empty
        shutil.rmtree(DIR_DST) # delete temp folder
        os.mkdir(DIR_DST)      # recreate the temp folder
    else:
        os.mkdir(DIR_DST)

    with open(file_sha256, "r") as f: # read sha256 values
        list_sha256 = f.readlines()

    for sha256 in list_sha256:
        sha256 = sha256.strip() # remove new line character
        file_src = DIR_SRC + sha256
        if not os.path.exists(file_src):
            continue
        file_dst = DIR_DST + sha256
        shutil.copyfile(file_src, file_dst) # copy samples
        n = n + 1
        print("{}: {}".format(n, sha256))


def generate_torrent_file(dir_sample, file_torrent):
    # d_samples is the directory containing samples
    # f_name is the specified name for the new torrent file
    print("\n[i] Generating torrent file.")
    new_torrent = Torrent.create_from(dir_sample)
    new_torrent.to_file(file_torrent)

def parse_args():
    parser = argparse.ArgumentParser(description = "Generate a torrent file for downloading.")
    parser.add_argument("-s", "--sha256_filename", help="The file containing sha256 values.", type=str, default=FILE_SHA256)
    parser.add_argument("-d", "--sample_dir", help="The directory containing samples", type=str, default=DIR_SAMPLES)
    parser.add_argument("-f", "--output_filename", help="The name of the new torrent file", type=str, default=FILE_TORRENT)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    copy_sample(args.sha256_filename)
    generate_torrent_file(args.sample_dir, args.output_filename) 
    print("\n[i] The torrent file \"{}\" has been generated.".format(args.output_filename))
    

if __name__ == "__main__":
    main()
