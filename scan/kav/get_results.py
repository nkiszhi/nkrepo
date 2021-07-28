#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
from sys import platform
import datetime

KAV_LOG = "Kaspersky_log.txt" # Raw log file
KAV_RESULT = "Kaspersky_result.txt" # Extracted information from log file
DIR_FOLDER = "/nkrepo/DATA/sha256/"


def extract_info():
    ''' Read Kaspersky raw scan log file and extract samples detection information.
The extracted information is stored in Kaspersky_results.txt.'''

    pattern_sha256 = r'[a-f0-9]{64}'
    pattern_part_of_sha256 = r'[a-f0-9]{64}//'
    # Algorithm:Class.Platform.Famliy.Variant
    pattern_result = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)(\.[a-zA-Z0-9-]*)*'
    scan_list = []

    with open(KAV_LOG, mode = "r", encoding = "utf-8") as f:
        for i in f.readlines():
            scan_list.append(i.replace("\n",""))
            
    result = []
    for line in scan_list:
        # use regular expression to match each line
        if re.findall(pattern_result, line) and not re.findall(pattern_part_of_sha256, line):
            match_list = list(re.findall(pattern_result, line)[0]) # the list of match result
            algorithm = match_list[0].replace(":","")
            s_class = match_list[1] # sample class, such as Trojon, Worm, Backdoor.
            s_platform = match_list[2] # sample platform, such as Boot, Win32.
            family = match_list[3]
            other = match_list[4]

            if algorithm!="":
                whole_info = algorithm + ":" + s_class + "." + s_platform + "." + family + other
            else:
                whole_info = s_class + "." + s_platform + "." + family + other

            sha256 = re.findall(pattern_sha256,line)[0]
            date_list=[str(datetime.datetime.now().year), str(datetime.datetime.now().month), str(datetime.datetime.now().day)]
            write_line = "-".join(date_list) +", " + sha256 + ", " + whole_info + ", " + algorithm + ", " + s_class + ", " + s_platform + ", " + family
            if write_line not in result:
                result.append(write_line)
        # elif re.findall(pattern_sha256, line) and not re.findall(pattern_result, line) and not re.findall(pattern_part_of_sha256, line):
        #     sha256 = re.findall(pattern_sha256,line)[0]
        #     date_list=[str(datetime.datetime.now().year), str(datetime.datetime.now().month), str(datetime.datetime.now().day)]
        #     algorithm = " "
        #     s_class = " "
        #     s_platform = " "
        #     family = " "
        #     other = " "
        #     whole_info = "clean"

        #     write_line = "-".join(date_list) +", " + sha256 + ", " + whole_info + ", " + algorithm + ", " + s_class + ", " + s_platform + ", " + family
        #     if write_line not in result:
        #         result.append(write_line)


    with open(KAV_RESULT,"w") as f:
        for i in result:
            f.write(i + "\n")


def mov_repo():
    txtlist = []
    with open(KAV_RESULT) as f:
        txtlist = f.readlines()

    # start to write .kav files
    for item in txtlist:
        # replace "\n" to "" for each line
        item = item.replace("\n","")
        # extract sha256256 and information
        sha256 = item.split()[0].replace(",","")
        information = item.split()[1:]

        FOLDER = DIR_FOLDER + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/"
        if os.path.exists(FOLDER + sha256 + ".kav"):
            os.remove(FOLDER + sha256 + ".kav")
        with open(FOLDER + sha256 + ".kav", "w") as f:
            f.write(" ".join(information))


def main():
    extract_info() # Read Kaspersky scan result and extract class and family information
    mov_repo() # Save Kaspersky scan resluts into repo
    

if __name__ == "__main__":
    main()
