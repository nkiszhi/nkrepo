#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Extract scanned txt files to generate CSV files

import os
import re
import time
import argparse
import csv

def save_csv(list_result):
    list_name = []
    list_scresult = []
    for name in list_result:
        list_name.append(name[0])
    list_name = list(set(list_name))
    print(list_name)
    for csv_name in list_name:
        data = []
        file_path = os.path.join('.', '%s.csv'%csv_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in list_result:
                    if csv_name == row[0]:
                        data.append(row)
                    else:
                        continue
                writer.writerows(data)
        else:
             with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in list_result:
                    if csv_name == row[0]:
                        data.append(row)
                    else:
                        continue
                writer.writerows(data)
            
def search_result(line):
    pattern_vs = r'(VirusShare_[0-9]{5})\\(VirusShare_[a-f0-9]{32})'
    pattern_result = r'([a-zA-Z0-9-]*:)?([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)\.([a-zA-Z0-9-]*)(\.[a-zA-Z0-9-]*)*'
    file_vs = re.search(pattern_vs, line)
    if not file_vs:
        return
    name_vs = file_vs.group(1)
    file_vs = file_vs.group(2)
    result = re.search(pattern_result, line)
    if not result:
        return
    if result.group(1):
        algorithm = result.group(1)
    else:
        algorithm = ""
    category = result.group(2)
    platform = result.group(3)
    family = result.group(4)
    mal_variant = result.group(5)
    result = result.group()
    return (name_vs,file_vs, category, platform, family, result)

def read_log(file_log):
    ''' Read Kaspersky log file and extract scan result.'''
    list_result = []
    _n = 0
    print(file_log)
    with open(file_log, mode="r", encoding="utf-8") as f:
        list_result = f.readlines()
    list_result = [x.strip() for x in list_result]
    list_result = list(filter(lambda x: len(x) > 50, list_result))
    list_result = [search_result(x) for x in list_result]
    list_result = list(filter(lambda x: x, list_result))
    list_result = list(set(list_result)) 
    save_csv(list_result)

def main():
    list_file = os.listdir(".")
    list_file = list(filter(lambda x: len(x)==10, list_file))
    n = [read_log(f) for f in list_file]

if __name__ == "__main__":
    main()
