#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "NKAMG"
__copyright__ = " Copyright (c) 2016 NKAMG"
__license__ = "GPL"

from torrentool.api import Torrent
import os
import argparse
import pandas as pd

D_SAMPLES = "./samples/"
F_NAME = "sample.torrent"

def generate_torrent_file(d_samples, f_name):
    # d_samples is the directory containing samples
    # f_name is the specified name for the new torrent file

    new_torrent = Torrent.create_from(d_samples)
    new_torrent.to_file(f_name)

def parse_args():
    parser = argparse.ArgumentParser(description = "Generate a torrent file for samples downloading.")
    parser.add_argument("-d", "--source_dir", help="The directory containing samples", type=str, default=D_SAMPLES)
    parser.add_argument("-f", "--output_filename", help="The name of the new torrent file", type=str, default=F_NAME)
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    generate_torrent_file(args.source_dir, args.output_filename) 
    

if __name__ == "__main__":
    main()
