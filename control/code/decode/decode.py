#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool

UNZIP_DIR = "../../../DATA/sha256/"

    

def main():
    
    sha256 = input('sha256:')
    unzip_file = UNZIP_DIR + sha256[0] +"/" + sha256[1] + "/" + sha256[2] + "/" + sha256[3] + "/" + sha256[4] + "/" + sha256 + ".mal"
    unzip_new_file = UNZIP_DIR + sha256
    os.system("java -jar zyydecodeutil.jar " + unzip_file + " " + unzip_new_file )


if __name__=="__main__":
    main()