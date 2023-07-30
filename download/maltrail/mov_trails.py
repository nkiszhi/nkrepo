#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# mov_trails.py : mov maltrail trails.csv to nkrepo
# location: nkrepo/download/maltrail

import datetime
import os

DIR_DATA = os.path.abspath("../../DATA/trails")

def main():
    csv_name = "{}.csv".format(datetime.date.today())
    file_src = "/root/.maltrail/trails.csv"
    file_tmp = DIR_DATA + "/trails.csv"
    file_dst = DIR_DATA + "/" + csv_name

    if not os.path.exists(DIR_DATA):
        os.makedirs(DIR_DATA)

    print(file_src)
    print(file_dst)
    os.system("cp {} {}".format(file_src, DIR_DATA))
    os.system("mv {} {}".format(file_tmp, file_dst))

if __name__ == "__main__":
    main()




