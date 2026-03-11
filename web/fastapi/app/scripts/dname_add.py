# -*- coding: utf-8 -*-
"""
Created on 2022/1/3 22:19

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pandas as pd

def add_dname(dname, if_dga=False):
    """
    添加域名
    :param dname: 域名字符串
    :param if_dga: 是否为恶意
    :return:
    """
    dname_df = pd.read_csv("./data/sample/dname_data.csv", header=0)
    dname_df = dname_df.append([{"dname":dname, "label": int(if_dga)}], ignore_index=True)
    dname_df.to_csv("./data/sample/dname_data.csv", index=False)
    print(dname_df)

