# -*- coding: utf-8 -*-
"""
Created on 2022/1/3 13:59

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from feeds.pvalue import cal_pValue
from feature_extraction import get_feature


# AdaBoost算法
class ADABoostClassifier:

    def __init__(self):
        self.AdaBoost_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.8)
        self.standard_scaler_add = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        AdaBoost算法训练数据
        :param model_folder: 模型存储路径
        :param train_feature_add: 训练数据路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______AdaBoost Training_______")
        self.AdaBoost_clf.fit(x_train, y_train)
        mal_scores = np.array(self.AdaBoost_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/AdaBoost_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.AdaBoost_clf, open("{}/AdaBoost_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_folder: 模型存储路径
        :return:
        """
        self.AdaBoost_clf = pickle.load(open("{}/AdaBoost_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/AdaBoost_train_scores.npy".format(model_folder))
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
        print("_______AdaBoost Predicting_______")
        y_predict = self.AdaBoost_clf.predict(x_test)
        print("AdaBoost accuracy: ", accuracy_score(y_test, y_predict))
        print("AdaBoost precision: ", precision_score(y_test, y_predict))
        print("AdaBoost recall: ", recall_score(y_test, y_predict))
        print("AdaBoost F1: ", f1_score(y_test, y_predict))

    def predict_single_dname(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param model_folder: 模型存储路径
        :param dname: 域名
        :return: [预测标签，恶意概率，可信度]
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
            print("\nAdaBoost dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.AdaBoost_clf.predict(feature)
            prob = self.AdaBoost_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nAdaBoost dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value