#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count the number of all malware samples in the repo."""


import pandas as pd
import os

hex_string="0123456789abcdef"

def get_all():
    n=0
    df= pd.DataFrame(columns=['sha256','pack']) 
    for i in hex_string:
        for j in hex_string:
            for k in hex_string:
                for l in hex_string:
                    f = "./DATA/" + i + "/"+ j + "/"+ k+ "/" + l + "/f_pack_info.csv"
                    if os.path.isfile(f):
                        df1 = pd.read_csv(f,header=None,names=['sha256','pack'])
                        df = pd.concat([df,df1], ignore_index=True)
                        n +=1
                        print(n)
                    print(df)
    df.to_csv("pack_info.csv")


def main():
    get_all()

if __name__=="__main__":
    main()
