# -*- coding: utf-8 -*-
"""
Created on 2022/1/3 13:51

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from feeds.pvalue import cal_pValue
from feature_extraction import get_feature


# 随机森林算法
class RFClassifier:

    def __init__(self):
        self.RF_clf = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                             random_state=23, n_jobs=-1, max_features=20)
        self.standardScaler = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        B-RF算法训练数据
        :param model_folder: 模型存储路径
        :param train_feature_add: 训练数据路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______RF Training_______")
        self.RF_clf.fit(x_train, y_train)
        mal_scores = np.array(self.RF_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/RF_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.RF_clf, open("{}/RF_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_folder: 模型存储路径
        :return:
        """
        self.RF_clf = pickle.load(open("{}/RF_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/RF_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param model_folder: 模型存储路径
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______RF Predicting_______")
        y_predict = self.RF_clf.predict(x_test)
        print("RF accuracy: ", accuracy_score(y_test, y_predict))
        print("RF precision: ", precision_score(y_test, y_predict))
        print("RF recall: ", recall_score(y_test, y_predict))
        print("RF F1: ", f1_score(y_test, y_predict))

    def predict_single_dname(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param model_folder: 模型存储路径
        :param dname: 域名
        :return:  [预测标签，恶意概率，可信度]
        """
        if not self.isload_:
            self.load(model_folder)
        dname = dname.strip('/').strip('.')
        dname = dname.replace("http://", '')
        dname = dname.replace("www.", "")
        if dname == "":
            label = 0
            prob = 0.0000
            p_value = 1.0000
            print("\nrf dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.RF_clf.predict(feature)
            prob = self.RF_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nrf dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value