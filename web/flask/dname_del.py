# -*- coding: utf-8 -*-
"""
Created on 2022/1/3 22:50

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pandas as pd
def del_dname(dname):
    """
    从数据中删除域名
    :param dname: 域名字符串
    :return:
    """
    dname_df = pd.read_csv("./data/sample/dname_data.csv", header=0)
    dname_df = dname_df[dname_df['dname'] != dname]
    dname_df.to_csv("./data/sample/dname_data.csv", index=False)
    print(dname_df)