#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#changed
import os, shutil
from multiprocessing import Pool

'''程序功能：
获取4层目录下不含有json和feat文件的样本，将其拷贝到移动硬盘上
'''
# 获取4级目录列表
hexstring = "0123456789abcdef"
hexstring_0 = "0123"
hexstring_4 = "4567"
hexstring_8 = "89ab"
hexstring_c = "cdef"
folder_list = []
folder_list_0 = []
folder_list_4 = []
folder_list_8 = []
folder_list_c = []
folder_list =   ["../DATA/sha256/" + i + "/" + j + "/" + k + "/" + l + "/" + m for i in hex_string for j in
              hex_string for k in hex_string for l in hex_string for m in hex_string]
folder_list_0 = ["../DATA/sha256/" + i + "/" + j + "/" + k + "/" + l + "/" + m for i in hexstring_0 for j in
              hex_string for k in hex_string for l in hex_string for m in hex_string]
folder_list_4 = ["../DATA/sha256/" + i + "/" + j + "/" + k + "/" + l + "/" + m for i in hexstring_4 for j in
              hex_string for k in hex_string for l in hex_string for m in hex_string]
folder_list_8 = ["../DATA/sha256/" + i + "/" + j + "/" + k + "/" + l + "/" + m for i in hexstring_8 for j in
              hex_string for k in hex_string for l in hex_string for m in hex_string]
folder_list_c = ["../DATA/sha256/" + i + "/" + j + "/" + k + "/" + l + "/" + m for i in hexstring_c for j in
              hex_string for k in hex_string for l in hex_string for m in hex_string]

print("folder finish")


# print(len(folder_list))
# print(16**4)

# 获得没有打过标签的样本
def get_unjsonfile_0(folder):
    print(folder)
    files = os.listdir(folder)
    for fil in files:
        if len(fil) == 64 and not os.path.exists(folder + "/" + fil + ".json") and not os.path.exists(
                folder + "/" + fil + ".feat"):
            shutil.copy(folder + "/" + fil, "/home/nkamg/share_kaspersky/sample/0_3/")
    print("finish " + folder)


def get_unjsonfile_4(folder):
    print(folder)
    files = os.listdir(folder)
    for fil in files:
        if len(fil) == 64 and not os.path.exists(folder + "/" + fil + ".json") and not os.path.exists(
                folder + "/" + fil + ".feat"):
            shutil.copy(folder + "/" + fil, "/home/nkamg/share_kaspersky/sample/4_7/")
    print("finish " + folder)


def get_unjsonfile_8(folder):
    print(folder)
    files = os.listdir(folder)
    for fil in files:
        if len(fil) == 64 and not os.path.exists(folder + "/" + fil + ".json") and not os.path.exists(
                folder + "/" + fil + ".feat"):
            shutil.copy(folder + "/" + fil, "/home/nkamg/share_kaspersky/sample/8_b/")
    print("finish " + folder)


def get_unjsonfile_c(folder):
    print(folder)
    files = os.listdir(folder)
    for fil in files:
        if len(fil) == 64 and not os.path.exists(folder + "/" + fil + ".json") and not os.path.exists(
                folder + "/" + fil + ".feat"):
            shutil.copy(folder + "/" + fil, "/home/nkamg/share_kaspersky/sample/c_f/")
    print("finish " + folder)


def main():
    # current_folder_list=folder_list[:2*16**3]
    # print(current_folder_list)
    p = Pool(5)
    # p.map(get_unjsonfile_0,folder_list_0)
    # p.map(get_unjsonfile_4,folder_list_4)
    p.map(get_unjsonfile_8, folder_list_8)
    p.map(get_unjsonfile_c, folder_list_c)


if __name__ == "__main__":
    main()