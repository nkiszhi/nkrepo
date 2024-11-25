from feeds.svm import SVMClassifier
from feeds.knn import KNNClassifier
from feeds.adaboost import ADABoostClassifier
from feeds.decisiontree import DTClassifier
from feeds.gbdt import GBDTClassifier
from feeds.logisticregression import LRClassifier
from feeds.randomforest import RFClassifier
from feeds.xgboost import XGBoostClassifier
from feeds.naivebayes import GNBClassifier
from feeds.lstm import LSTMClassifier

dst = './data/model/'
train_feature_add = './data/features/test_features.csv'

def SVM_Training (dst,train_feature_add):
    model = SVMClassifier()
    model.train(dst,train_feature_add)

def KNN_Training (dst,train_feature_add):
    model = KNNClassifier()
    model.train(dst,train_feature_add)

def DT_Training (dst,train_feature_add):
    model = DTClassifier()
    model.train(dst,train_feature_add)

def GBDT_Training (dst,train_feature_add):
    model = GBDTClassifier()
    model.train(dst,train_feature_add)

def LR_Training (dst,train_feature_add):
    model = LRClassifier()
    model.train(dst,train_feature_add)

def RF_Training (dst,train_feature_add):
    model = RFClassifier()
    model.train(dst,train_feature_add)

def ADABoost_Training (dst,train_feature_add):
    model = ADABoostClassifier()
    model.train(dst,train_feature_add)

def XGBoost_Training (dst,train_feature_add):
    model = XGBoostClassifier()
    model.train(dst,train_feature_add)
    
def GNB_Training (dst,train_feature_add):
    model = GNBClassifier()
    model.train(dst,train_feature_add)

def LST_Training (dst,train_feature_add):
    model = LSTMClassifier()
    model.train(dst,train_feature_add)

if __name__ == '__main__':
#    SVM_Training (dst,train_feature_add)
    KNN_Training (dst,train_feature_add)
    DT_Training (dst,train_feature_add)
    GBDT_Training (dst,train_feature_add)
    LR_Training (dst,train_feature_add)
    RF_Training (dst,train_feature_add)
    ADABoost_Training (dst,train_feature_add)
    XGBoost_Training (dst,train_feature_add)
    GNB_Training (dst,train_feature_add)
    LST_Training (dst,train_feature_add)
    
