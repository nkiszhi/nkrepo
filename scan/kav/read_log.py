#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from sys import platform
import datetime

KAV_LOG = "kav_log.txt" # Raw log file
DIR_REPO = "/nkrepo/DATA/sha256/"

def save_result(sha256, result, algorithm, mal_class, mal_platform, mal_family):
    # Create kav file to store kav scan result, and save kav file into nkrepo.
    f_kav = DIR_REPO + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256 + ".kav"
    #print(f_kav)
    if os.path.exists(f_kav):
        os.remove(f_kav)
    with open(f_kav, "w") as f:
        f.write("{}, {}, {}, {}, {}".format(result, algorithm, mal_class, mal_platform, mal_family))

def read_log():
    ''' Read Kaspersky raw scan log file and extract samples detection information.
The extracted information is stored in kav_results.txt already.'''

    pattern_sha256 = r'[a-f0-9]{64}'
    # Algorithm:Class.Platform.Famliy.Variant
    pattern_result = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)(\.[a-zA-Z0-9-]*)*'
    scan_list = []
    _n = 0
    with open(KAV_LOG, mode = "r", encoding = "utf-8") as f:
        for _l in f.readlines():
            #print(_l)
            # Searching sha256 string
            sha256 = re.search(pattern_sha256, _l)
            if not sha256:
                # If no sha256 string found, continue to next kav log
                continue
            sha256 =  sha256.group()
            #print(sha256)
            # Searching Kaspersky scan result
            result = re.search(pattern_result, _l)
            if not result:
                # If no Kaspersky result is found, continue to next kav log
                continue
            # 1. detection method, such as HEUR...
            if result.group(1): 
                algorithm = result.group(1)
            else:
                algorithm = ""
            # 2. malware class, such as Trojan, Worm,...
            mal_class = result.group(2)
            # 3. malware running platform, such as Win32, Script, ...
            mal_platform = result.group(3)
            # 4. malware family, such as  
            mal_family = result.group(4)
            # 5. malware variant information
            mal_variant = result.group(5)
            # 6. detection result: Algorithm:Class.Platform.Family.Variant
            result = result.group()
            save_result(sha256, result, algorithm, mal_class, mal_platform, mal_family)
            _n = _n + 1
            print("{}: {} {}".format(_n, sha256, result))
    return
            
    #result = []
    #for line in scan_list:
    #    # use regular expression to match each line
    #    #if not re.findall(pattern_part_of_sha256, line):
    #    if re.findall(pattern_result, line) and not re.findall(pattern_part_of_sha256, line):
    #        match_list = list(re.findall(pattern_result, line)[0]) # the list of match result
    #        algorithm = match_list[0].replace(":","")
    #        s_class = match_list[1] # sample class, such as Trojon, Worm, Backdoor.
    #        s_platform = match_list[2] # sample platform, such as Boot, Win32.
    #        family = match_list[3] # sample family
    #        other = match_list[4] # sample variant

    #        if algorithm!="":
    #            whole_info = algorithm + ":" + s_class + "." + s_platform + "." + family + other
    #        else:
    #            whole_info = s_class + "." + s_platform + "." + family + other

    #        sha256 = re.findall(pattern_sha256,line)[0]
    #        date_list=[str(datetime.datetime.now().year), str(datetime.datetime.now().month), str(datetime.datetime.now().day)]
    #        write_line = "-".join(date_list) +", " + sha256 + ", " + whole_info + ", " + algorithm + ", " + s_class + ", " + s_platform + ", " + family
    #        if write_line not in result:
    #            result.append(write_line)
    #    # elif re.findall(pattern_sha256, line) and not re.findall(pattern_result, line) and not re.findall(pattern_part_of_sha256, line):
    #    #     sha256 = re.findall(pattern_sha256,line)[0]
    #    #     date_list=[str(datetime.datetime.now().year), str(datetime.datetime.now().month), str(datetime.datetime.now().day)]
    #    #     algorithm = " "
    #    #     s_class = " "
    #    #     s_platform = " "
    #    #     family = " "
    #    #     other = " "
    #    #     whole_info = "clean"

    #    #     write_line = "-".join(date_list) +", " + sha256 + ", " + whole_info + ", " + algorithm + ", " + s_class + ", " + s_platform + ", " + family
    #    #     if write_line not in result:
    #    #         result.append(write_line)


    #with open(KAV_RESULT,"w") as f:
    #    for i in result:
    #        f.write(i + "\n")


#def mov_repo():
#    txtlist = []
#    with open(KAV_RESULT) as f:
#        txtlist = f.readlines()
#
#    # start to write .kav files
#    for item in txtlist:
#        # replace "\n" to "" for each line
#        item = item.replace("\n","")
#        # extract sha256256 and information
#        sha256 = item.split()[0].replace(",","")
#        information = item.split()[1:]
#
#        FOLDER = DIR_FOLDER + sha256[0] + "/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/"
#        if os.path.exists(FOLDER + sha256 + ".kav"):
#            os.remove(FOLDER + sha256 + ".kav")
#        with open(FOLDER + sha256 + ".kav", "w") as f:
#            f.write(" ".join(information))


def main():
    read_log() # Read Kaspersky scan result and extract class and family information
    
if __name__ == "__main__":
    main()
