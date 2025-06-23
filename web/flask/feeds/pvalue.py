# -*- coding: utf-8 -*-
"""
Created on 2022/1/3 13:05

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

@author : dengcongyi0701@163.com

Description:

"""


def cal_pValue(score_list, key, label):
    """
    计算p_value
    :param score_list: 训练集得分列表
    :param key: 测试样本得分
    :param label: 测试样本标签
    :return: p_value, 保留四位小数
    """
    count = 0
    if label == 0:
        temp = sorted(score_list, reverse=True)
        score_list = [i for i in temp if i <= 0.5]
        while count < len(score_list) and score_list[count] >= key:
            count += 1
    elif label == 1:
        temp = sorted(score_list, reverse=False)
        score_list = [i for i in temp if i > 0.5]
        while count < len(score_list) and score_list[count] <= key:
            count += 1
    p_value = count/len(score_list)
    return round(p_value, 4)

