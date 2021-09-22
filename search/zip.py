#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

import argparse
import tarfile
import os.path

D_SAMPLE = "./samples/"
F_TAR = "sample.tgz"


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def parse_args():
    parser = argparse.ArgumentParser(description = "Generate a tar file containing samples.")
    parser.add_argument("-d", "--source_dir", help="The directory containint samples.", type=str, default=D_SAMPLE)
    parser.add_argument("-f", "--output_filename", help="The output tar file.", type=str, default=F_TAR)
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    make_tarfile(args.output_filename, args.source_dir) 
    

if __name__ == "__main__":
    main()
