# -*- coding: utf-8 -*-
"""
Created on 2020/8/16 12:38

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
from importlib import import_module
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from configparser import ConfigParser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 多模型检测
class MultiModelDetection:

    def __init__(self):
        self._cfg = dict()
        cp = ConfigParser()
        cp.read('config.ini')
        self._cfg["model_path"] = cp.get('files', 'model_path')
        self._cfg["train_add"] = cp.get('files', 'train_add')
        self._cfg["test_add"] = cp.get('files', 'test_add')
        self._cfg["algorithm_lst"] = cp.get('feeds', 'algorithm_list').split(',')
        self._cfg["classifier_lst"] = cp.get('feeds', 'classifier_list').split(',')
        self._load_models()

    def _load_models(self):
        """
        将训练好的多个模型全部预加载到内存中
        :return:
        """
        self._clf_list = list()
        for i in range(len(self._cfg["algorithm_lst"])):
            aMod = import_module('feeds.'+self._cfg["algorithm_lst"][i])
            aClass = getattr(aMod, self._cfg["classifier_lst"][i])
            clf = aClass()
            clf.load(self._cfg["model_path"])
            self._clf_list.append(clf)

    def multi_predict_single_dname(self, dname):
        """
        对单个域名进行多模型协同检测
        :param dname: 域名
        :return: (基础检测结果——字典类型，多模型检测结果——0安全1危险2可疑）
        """
        base_result = dict()
        base_result_t = dict()
        for i in range(len(self._clf_list)):
            clf_pre_rs = self._clf_list[i].predict_single_dname(self._cfg["model_path"], dname)
            base_result[self._cfg["classifier_lst"][i][:-10]] = [clf_pre_rs[0], format(clf_pre_rs[1], '.4f'),
                                                                 clf_pre_rs[2]]
            base_result_t[self._cfg["classifier_lst"][i][:-10]] = clf_pre_rs if clf_pre_rs[2] > 0.01 \
                else (2, clf_pre_rs[1], clf_pre_rs[2])
        rs_list = list()
        for j in base_result_t:
            rs_list.append(base_result_t[j][0])
        if len(set(rs_list)) == 1:
            if list(base_result_t.values())[0][0] != 2:
                result = list(base_result_t.values())[0][0]
                return base_result, result
            elif list(base_result_t.values())[0][0] == 2:  # 所有模型都表现很差
                sort_result = sorted(base_result_t.items(), key=lambda base_result_t: base_result_t[1][2], reverse=True)
                if sort_result[0][1][2] <= 0.5:
                    result = 2
                else:
                    result = sort_result[0][1][0]
                return base_result, result

        new_result = dict()
        for k in base_result_t:
            if base_result_t[k][0] != 2:
                new_result[k] = base_result_t[k]
        sort_result = sorted(new_result.items(), key=lambda new_result: new_result[1][2], reverse=True)
        if sort_result[0][1][2] <= 0.5:
            result = 2
        else:
            result = sort_result[0][1][0]

        return base_result, result


if __name__ == "__main__":
    # muldec = MultiModelDetection()

    from feeds.danalysis import LDAClassifier
    clf = LDAClassifier()
    # clf.train(r"./data/model", r"./data/features/train_features.csv")
    clf.load(r"./data/model")
    clf.predict(r"./data/model", r"./data/features/test_features.csv")


