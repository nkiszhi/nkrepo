#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Init DATA folder with 5 levels subfolders."""

import os
import argparse
HEX_STRING = "0123456789abcdef"
LIST_DATA_FOLDER = "list_data_folder.txt"
DIR_SHA256 = ""
DIR_MD5 = ""


def create_list_data_folder():
    list_folder = []
    for i in HEX_STRING:
        for j in HEX_STRING:
            for k in HEX_STRING:
                for l in HEX_STRING:
                    for m in HEX_STRING:
                        folder = i + "/" + j + "/" + k + "/" + l + "/" + m + "/"
                        list_folder.append(folder)
    with open(LIST_DATA_FOLDER, "w") as f:
        for folder in list_folder:
            f.write(folder + "\n")


def create_folder(list_folder):
    n = 0
    for folder in list_folder:
        if os.path.exists(folder):
            continue
        n = n + 1
        os.makedirs(folder)
    return n


# Create 4-tier folder architecture to store samples
def init_repo():
    list_folder = []
    with open(LIST_DATA_FOLDER, "r") as f:
        list_folder = f.readlines()
    list_folder = [x.strip() for x in list_folder]

    list_sha256_folder = [os.path.abspath(DIR_SHA256 + x) for x in list_folder]
    list_md5_folder = [os.path.abspath(DIR_MD5 + x) for x in list_folder]
    n_new_sha256_folder = create_folder(list_sha256_folder)
    n_new_md5_folder = create_folder(list_md5_folder)

    print("[o]: Created sha256 folder: {}.".format(n_new_sha256_folder))
    print("[o]: Created md5 folder: {}.".format(n_new_md5_folder))


def main():
    parser = argparse.ArgumentParser(prog="init_repo")
    parser.add_argument("-sha", "--sha256dir", help="The path of sha256.", default="../DATA/sha256/")
    parser.add_argument("-md5", "--md5dir", help="The path of md5.", default="../DATA/md5/")
    args = parser.parse_args()
    global DIR_SHA256
    global DIR_MD5
    DIR_SHA256 = args.sha256dir
    DIR_MD5 = args.md5dir

    if not os.path.exists(LIST_DATA_FOLDER):
        create_list_data_folder()
    init_repo()


if __name__ == "__main__":
    main()
