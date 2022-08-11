#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool



def encode(workfile):
    zip_name = workfile
    zip_folder = workfile + ".mal"
    os.system("java -jar zyyencodeutil.jar " + zip_name + " " + zip_folder+" 00000001")
    # os.remove(zip_name)  
    # delete
    
def main():
    count =0
    src_path = '../../../DATA/sha256/'
    # traverse DATA folder
    work_list=[]
    for root,dirs,files in os.walk(src_path,topdown=True):
        if not len(dirs) and len(files):
            for file in files:
                if len(file) != 64:
                    continue
                count = count + 1
                srcName = os.path.join(root, file)
                work_list.append(srcName)
                print(srcName)
    p = Pool(50)
    p.map(encode, work_list)
    print('all_encode_count:%d'%count)


if __name__=="__main__":
    main()