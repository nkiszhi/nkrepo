# -*- coding: utf-8 -*-
"""
Created on 2020/8/16 12:38

@author : dengcongyi0701@163.com
          liying_china@163.com

Description:

"""
import warnings
import math
warnings.filterwarnings('ignore')
import pandas as pd
import pickle
import numpy as np
import string
import tld
import os
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score
from feature_extraction import wash_tld, get_feature
from sklearn.model_selection import StratifiedShuffleSplit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# KNN算法
class KNN_classifier:

    def __init__(self):
        self.KNN_clf = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski')
        self.standard_scaler_add = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        KNN算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_folder:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______KNN Training_______")
        self.KNN_clf.fit(x_train, y_train)
        mal_scores = np.array(self.KNN_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/KNN_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.KNN_clf, open("{}/KNN_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :return:
        """
        self.KNN_clf = pickle.load(open("{}/KNN_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/KNN_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______KNN Predicting_______")
        y_predict = self.KNN_clf.predict(x_test)
        print("KNN accuracy: ", self.KNN_clf.score(x_test, y_test))
        print("KNN precision: ", precision_score(y_test, y_predict, average='macro'))
        print("KNN recall: ", recall_score(y_test, y_predict, average='macro'))
        print("KNN F1: ", f1_score(y_test, y_predict, average='macro'))
        print("KNN TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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
            print("\nknn dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.KNN_clf.predict(feature)
            prob = self.KNN_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nknn dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value


# SVM算法
class SVM_classifier:

    def __init__(self):
        self.SVM_clf = SVC(kernel='linear', probability=True, random_state=23)
        self.standard_scaler_add = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        SVM算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______SVM Training_______")
        self.SVM_clf.fit(x_train, y_train)
        mal_scores = np.array(self.SVM_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/SVM_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.SVM_clf, open("{}/SVM_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :return:
        """
        self.SVM_clf = pickle.load(open("{}/RF_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/RF_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______SVM Predicting_______")
        y_predict = self.SVM_clf.predict(x_test)
        print("SVM accuracy: ", self.SVM_clf.score(x_test, y_test))
        print("SVM precision: ", precision_score(y_test, y_predict, average='macro'))
        print("SVM recall: ", recall_score(y_test, y_predict, average='macro'))
        print("SVM F1: ", f1_score(y_test, y_predict, average='macro'))
        print("SVM TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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
            label = self.SVM_clf.predict(feature)
            prob = self.SVM_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nsvm sld:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value

# 朴素贝叶斯算法
class GNB_classifier:

    def __init__(self):
        self.GNB_clf = GaussianNB()
        self.standard_scaler_add = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        GNB算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______GNB Training_______")
        self.GNB_clf.fit(x_train, y_train)
        mal_scores = np.array(self.GNB_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/GNB_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.GNB_clf, open("{}/GNB_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :return:
        """
        self.GNB_clf = pickle.load(open("{}/GNB_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/GNB_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______GNB Predicting_______")
        y_predict = self.GNB_clf.predict(x_test)
        print("GNB accuracy: ", self.GNB_clf.score(x_test, y_test))
        print("GNB precision: ", precision_score(y_test, y_predict, average='macro'))
        print("GNB recall: ", recall_score(y_test, y_predict, average='macro'))
        print("GNB F1: ", f1_score(y_test, y_predict, average='macro'))
        print("GNB TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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
            print("\nGNB dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.GNB_clf.predict(feature)
            prob = self.GNB_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nGNB dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value

# 逻辑回归算法
class LR_classifier:

    def __init__(self):
        self.LR_clf = LogisticRegression(penalty='l2')
        self.standard_scaler_add = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        逻辑回归算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_folder:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______LR Training_______")
        self.LR_clf.fit(x_train, y_train)
        mal_scores = np.array(self.LR_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/LR_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.LR_clf, open("{}/LR_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :return:
        """
        self.LR_clf = pickle.load(open("{}/LR_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/LR_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______LR Predicting_______")
        y_predict = self.LR_clf.predict(x_test)
        print("LR accuracy: ", self.LR_clf.score(x_test, y_test))
        print("LR precision: ", precision_score(y_test, y_predict, average='macro'))
        print("LR recall: ", recall_score(y_test, y_predict, average='macro'))
        print("LR F1: ", f1_score(y_test, y_predict, average='macro'))
        print("LR TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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
            print("\nLR dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.LR_clf.predict(feature)
            prob = self.LR_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nLR dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value

# 决策树分类算法
class DT_classifier:

    def __init__(self):
        self.DT_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        self.standardScaler = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        决策树算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______DT Training_______")
        self.DT_clf.fit(x_train, y_train)
        mal_scores = np.array(self.DT_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/DT_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.DT_clf, open("{}/DT_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :return:
        """
        self.DT_clf = pickle.load(open("{}/DT_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/DT_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______DT Predicting_______")
        y_predict = self.DT_clf.predict(x_test)
        print("DT accuracy: ", self.DT_clf.score(x_test, y_test))
        print("DT precision: ", precision_score(y_test, y_predict, average='macro'))
        print("DT recall: ", recall_score(y_test, y_predict, average='macro'))
        print("DT F1: ", f1_score(y_test, y_predict, average='macro'))
        print("DT TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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
            print("\ndt dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.DT_clf.predict(feature)
            prob = self.DT_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\ndt dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value

# 二分类随机森林(B_RF)算法
class RF_classifier:

    def __init__(self):
        self.RF_clf = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                             random_state=23, n_jobs=-1, max_features=20)
        self.standardScaler = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        B-RF算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
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
        :param model_add: 模型存储路径
        :return:
        """
        self.RF_clf = pickle.load(open("{}/RF_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/RF_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
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
        print("RF accuracy: ", self.RF_clf.score(x_test, y_test))
        print("RF precision: ", precision_score(y_test, y_predict, average='macro'))
        print("RF recall: ", recall_score(y_test, y_predict, average='macro'))
        print("RF F1: ", f1_score(y_test, y_predict, average='macro'))
        print("RF TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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
            print("\nrf sld:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value

# GDBT算法
class GDBT_classifier:

    def __init__(self):
        self.GDBT_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.8)
        self.standardScaler = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        GDBT算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______GDBT Training_______")
        self.GDBT_clf.fit(x_train, y_train)
        mal_scores = np.array(self.GDBT_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/GDBT_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.GDBT_clf, open("{}/GDBT_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :return:
        """
        self.GDBT_clf = pickle.load(open("{}/GDBT_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/GDBT_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______GDBT Predicting_______")
        y_predict = self.GDBT_clf.predict(x_test)
        print("GDBT accuracy: ", self.GDBT_clf.score(x_test, y_test))
        print("GDBT precision: ", precision_score(y_test, y_predict, average='macro'))
        print("GDBT recall: ", recall_score(y_test, y_predict, average='macro'))
        print("GDBT F1: ", f1_score(y_test, y_predict, average='macro'))
        print("GDBT TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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
            print("\nGDBT dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.GDBT_clf.predict(feature)
            prob = self.GDBT_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nGDBT dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value

# AdaBoost算法
class AdaBoost_classifier:

    def __init__(self):
        self.AdaBoost_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.8)
        self.standard_scaler_add = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        AdaBoost算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_folder:  模型存储路径
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
        :param model_add: 模型存储路径
        :return:
        """
        self.AdaBoost_clf = pickle.load(open("{}/AdaBoost_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/AdaBoost_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
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
        print("AdaBoost accuracy: ", self.AdaBoost_clf.score(x_test, y_test))
        print("AdaBoost precision: ", precision_score(y_test, y_predict, average='macro'))
        print("AdaBoost recall: ", recall_score(y_test, y_predict, average='macro'))
        print("AdaBoost F1: ", f1_score(y_test, y_predict, average='macro'))
        print("AdaBoost TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))

    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
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

# XGBoost算法
class XGBoost_classifier:

    def __init__(self):
        self.XGBoost_clf = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimator=100, silent=True,
                                         objective='binary:logistic')
        self.standardScaler = StandardScaler()
        self.train_score = None
        self.isload_ = False

    def train(self, model_folder, train_feature_add):
        """
        XGBoost算法训练数据
        :param train_feature_add: 训练数据路径
        :param model_add:  模型存储路径
        :return:
        """
        train_df = pd.read_csv(train_feature_add, index_col=['domain_name'])
        train_df = train_df.fillna('0.0')
        x_train = train_df.drop(['label'], axis=1).values
        y_train = train_df['label'].values
        print("_______XGBoost Training_______")
        self.XGBoost_clf.fit(x_train, y_train)
        mal_scores = np.array(self.XGBoost_clf.predict_proba(x_train))[:, 1]
        mal_scores = sorted(mal_scores)
        np.save(r"{}/XGBoost_train_scores.npy".format(model_folder), mal_scores)
        pickle.dump(self.XGBoost_clf, open("{}/XGBoost_model.pkl".format(model_folder), 'wb'))

    def load(self, model_folder):
        """
        将模型文件和归一化尺度读取到内存中
        :param model_add: 模型存储路径
        :param standard_scaler_add: 归一化scaler存储路径
        :return:
        """
        self.XGBoost_clf = pickle.load(open("{}/XGBoost_model.pkl".format(model_folder), 'rb'))
        self.standardScaler = pickle.load(open("{}/standardscalar.pkl".format(model_folder), 'rb'))
        self.train_score = np.load(r"{}/XGBoost_train_scores.npy".format(model_folder))
        self.isload_ = True

    def predict(self, model_folder, test_feature_add):
        """
        测试集进行测试，计算准确率等
        :param test_feature_add: 测试数据路径
        :return:
        """
        self.load(model_folder)
        test_df = pd.read_csv(test_feature_add, index_col=['domain_name'])
        test_df = test_df.fillna('0.0')
        x_test = test_df.drop(['label'], axis=1).values
        y_test = test_df['label'].values
        print("_______XGBoost Predicting_______")
        y_predict = self.XGBoost_clf.predict(x_test)
        print("XGBoost accuracy: ", self.XGBoost_clf.score(x_test, y_test))
        print("XGBoost precision: ", precision_score(y_test, y_predict, average='macro'))
        print("XGBoost recall: ", recall_score(y_test, y_predict, average='macro'))
        print("XGBoost F1: ", f1_score(y_test, y_predict, average='macro'))
        print("XGBoost TPR, FPR, thresholds: ", roc_curve(y_test, y_predict, pos_label=1))


    def predict_singleDN(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
        """
        if not self.isload_:
            self.load(model_folder)
        dname = dname.strip('/').strip('.')
        dname = dname.replace("http://", '')        
        dname = dname.replace("www.", "")
        # dname = wash_tld(dname)
        if dname == "":
            label = 0
            prob = 0.0000
            p_value = 1.0000
            print("\nxgboost dname:", dname)
            # print("label:", label)
            # print("mal_prob:", prob)
            # print("p_value:", p_value)
            print('label:{}, pro:{}, p_value:{}'.format(label, prob, p_value))
            return label, prob, p_value
        else:
            feature = self.standardScaler.transform(pd.DataFrame([get_feature(dname)]))
            label = self.XGBoost_clf.predict(feature)
            prob = self.XGBoost_clf.predict_proba(feature)
            p_value = cal_pValue(self.train_score, prob[0][1], label[0])
            print("\nxgboost sld:", dname)
            # print("label:", label[0])
            # print("mal_prob:", prob[0][1])
            # print("p_value:", p_value)
            print('label:{}, pro:{}, p_value:{}'.format(label[0], prob[0][1], p_value))
            return label[0], prob[0][1], p_value





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




class LSTM_classifier:
    def __init__(self):
        self.model = None
        self.valid_chars = {'q': 17, '0': 27, 'x': 24, 'd': 4, 'l': 12, 'm': 13, 'v': 22, 'n': 14, 'c': 3, 'g': 7, '7': 34, 'u': 21, '5': 32, 'p': 16, 'h': 8, 'b': 2, '6': 33, '-': 38, 'z': 26, '3': 30, 'f': 6, 't': 20, 'j': 10, '1': 28, '4': 31, 's': 19, 'o': 15, 'w': 23, '9': 36, 'r': 18, 'i': 9, 'e': 5, 'y': 25, 'a': 1, '.': 37, '2': 29, '_': 39, '8': 35, 'k': 11}
        self.maxlen = 178
        self.max_features = 40
        self.max_epoch = 2  # 20
        self.batch_size = 128
        self.tld_list = []
        with open(r'./data/tld.txt', 'r', encoding='utf8') as f:
            for i in f.readlines():
                self.tld_list.append(i.strip()[1:])

        score_df = pd.read_csv(r"./data/lstm_score_rank.csv", names=['score'])
        self.score_l = score_df['score'].tolist()

    def build_binary_model(self):
        """Build LSTM model for two-class classification"""
        self.model = Sequential()
        self.model.add(Embedding(self.max_features, 128, input_length=self.maxlen))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy',optimizer='rmsprop')

    def create_class_weight(self, labels_dict, mu):
        """Create weight based on the number of sld name in the dataset"""
        labels_dict = dict(labels_dict)
        keys = labels_dict.keys()
        total = labels_dict[1] + labels_dict[0]
        class_weight = dict()
        for key in keys:
            score = math.pow(total/float(labels_dict[key]), mu)
            class_weight[key] = score
        return class_weight

    def train(self, train_feature_add, model_add, model_weight):
        """
        训练模型
        :param train_feature_add: 训练数据
        :param model_add: 模型json文件
        :param model_weight: 模型权重文件
        :return:
        """
        # 获取训练和测试数据， domain，label
        train_df = pd.read_csv(train_feature_add, header=[0])
        train_df = train_df.iloc[:, 0:2]
        train_df["domain_name"] = train_df["domain_name"].apply(self.data_pro)
        sld = train_df["domain_name"].to_list()
        label = train_df["label"].to_list()
        X = [[self.valid_chars[y] for y in x] for x in sld]
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        y = np.array(label)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=4)
        for train, test in sss.split(X, y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            print("---train:{}---test:{}----y_train:{}----y_test:{}".format(len(X_train), len(X_test), len(y_train),
                                                                            len(y_test)))
            # shuffle
            np.random.seed(4)  # 1024
            index = np.arange(len(X_train))
            np.random.shuffle(index)
            X_train = np.array(X_train)[index]
            y_train = np.array(y_train)[index]
            # build model
            self.build_binary_model()
            # train
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
            for train, test in sss1.split(X_train, y_train):
                X_train, X_holdout, y_train, y_holdout = X_train[train], X_train[test], y_train[train], y_train[test]  # holdout验证集
            labels_dict = collections.Counter(y_train)
            class_weight = self.create_class_weight(labels_dict, 0.3)
            print('----class weight:{}'.format(class_weight))
            # 20
            best_acc = 0.0
            best_model = None
            for ep in range(self.max_epoch):
                self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=1, class_weight=class_weight)
                t_probs = self.model.predict_proba(X_holdout)
                t_result = [0 if x <= 0.5 else 1 for x in t_probs]
                t_acc = accuracy_score(y_holdout, t_result)
                print("epoch:{}--------val acc:{}---------best_acc:{}".format(ep, t_acc, best_acc))
                if t_acc > best_acc:
                    best_model = self.model
                    best_acc = t_acc
            model_json = best_model.to_json()
            # 模型的权重保存在HDF5中
            # 模型的结构保存在JSON文件或者YAML文件中
            with open(model_add, "w") as json_file:
                json_file.write(model_json)
                self.model.save_weights(model_weight)
            print("Saved two-class model to disk")


    def load(self, model_add, model_weight_add):
        """
        将模型文件和权重值读取
        :param model_add: 模型存储路径
        :param model_weight_add: 权重存储路径
        :return:
        """
        with open(model_add, 'r') as json_file:
            model = json_file.read()
        self.model = model_from_json(model)
        self.model.load_weights(model_weight_add)

    def data_pro(self, url):
        """
        预处理字符串
        :param url:
        :return:
        """
        url = url.strip().strip('.').strip('/')
        url = url.replace("http://", '')
        url = url.split('/')[0]
        url = url.split('?')[0]
        url = url.split('=')[0]
        dn_list = url.split('.')
        for i in reversed(dn_list):
            if i in self.tld_list:
                dn_list.remove(i)
            elif i == 'www':
                dn_list.remove(i)
            else:
                continue
        short_url = ''.join(dn_list)
        short_url = short_url.replace('[', '').replace(']', '')
        short_url = short_url.lower()
        return short_url

    def cal_p(self, s):
        """
        计算p_value, 二分查找
        :param s: float
        :return:
        """
        flag = 0  # score偏da的对应的
        for i in range(len(self.score_l)):
            if self.score_l[i] <= 0.5000000000000000:
                flag = i - 1
                break
        # print("flag:{}".format(flag))
        if s >= self.score_l[0]:
            return 1.0
        if s <= self.score_l[-1]:
            return 1.0
        if s == self.score_l[flag]:
            # return 1 / ((flag + 1) * 1.0)
            return 0.0

        high_index = len(self.score_l)
        low_index = 0
        while low_index < high_index:
            mid = int((low_index + high_index) / 2)
            if s > self.score_l[mid]:
                high_index = mid - 1
            elif s == self.score_l[mid]:
                if s > 0.5:
                    return (flag - mid + 1) / ((flag + 1) * 1.0)
                else:
                    # return (len(self.score_l) - mid) / ((len(self.score_l) - flag - 1) * 1.0)
                    return (mid - flag) / ((len(self.score_l) - flag - 1) * 1.0)
            else:
                low_index = mid + 1
        if s > 0.5:
            # print(low_index, (flag - low_index), ((flag + 1) * 1.0))
            return round((flag - low_index) / ((flag + 1) * 1.0), 4)
        else:
            # print(low_index, len(score_l) - low_index, (len(score_l) - flag - 1) * 1.0)
            # return (len(self.score_l) - low_index) / ((len(self.score_l) - flag - 1) * 1.0)
            return round((low_index - flag) / ((len(self.score_l) - flag - 1) * 1.0), 4)

    def predict_singleDN(self, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param dname: 域名
        :return:
        """
        dname = dname.strip(string.punctuation)
        short_url = self.data_pro(dname)
        print("\nlstm sld-----{}".format(short_url))

        sld_int = [[self.valid_chars[y] for y in x] for x in [short_url]]
        sld_int = sequence.pad_sequences(sld_int, maxlen=self.maxlen)
        sld_np = np.array(sld_int)
        # 编译模型
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        if short_url == '':
            score = 0.0
            p_value = 1.0
            label = 0
            print('label:{}, pro:{}, p_value:{}'.format(label, score, p_value))
            return label, score, p_value
        else:
            scores = self.model.predict(sld_np)
            score = scores[0][0]
            p_value = self.cal_p(score)

            if score > 0.5:
                label = 1
            else:
                label = 0
            print('label:{}, pro:{}, p_value:{}'.format(label, score, p_value))
            return label, score, p_value


if __name__ == "__main__":
    standard_scaler_add = r"./data/model/standardscalar.pkl"
    LSTM_model_add = r"./data/model/LSTM_model.json"
    LSTM_model_weight = r"./data/model/LSTM_model.h5"
    tld_path = r'./data/tld.txt'
    score_path = r"./data/lstm_score_rank.csv"
    model_path = r"./data/model"
    train_add = r"./data/features/train_features.csv"
    test_add = r"./data/features/test_features.csv"
