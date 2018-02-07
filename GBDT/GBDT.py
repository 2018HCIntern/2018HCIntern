import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import time, os, random, sys
import math
import warnings

warnings.filterwarnings('ignore')


class GBDT:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        print('Transforming features by using GBDT')





    def XGBdt(self, n_trees=30, max_depth=7, learning_rate=0.1):


        X_train = self.x_train
        y_train = self.y_train
        X_test = self.x_test
        y_test = self.y_test



        num_round = n_trees
        max_depth = max_depth
        learning_rate = learning_rate



        dtrain = xgb.DMatrix(X_train.values, y_train.values)
        dtest = xgb.DMatrix(X_test.values, y_test.values)
        param = {'silent': 1, 'objective': 'binary:logistic', 'max_depth': max_depth, 'eta': learning_rate}


        bst = xgb.train(param, dtrain, num_round)

        y_pred = bst.predict(dtrain, pred_leaf=True)

        num_leaf = np.max(y_pred)
        X_train_leaves_xgb = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
            X_train_leaves_xgb[i][temp] += 1

        y_pred = bst.predict(dtest, pred_leaf=True)
        X_test_leaves_xgb = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
            X_test_leaves_xgb[i][temp] += 1

        return X_train_leaves_xgb, X_test_leaves_xgb


    def LightGBMdt(self, n_trees=30, max_depth=7, learning_rate=0.1):

        X_train = self.x_train
        y_train = self.y_train
        X_test = self.x_test

        num_trees = n_trees
        max_depth = max_depth
        learning_rate = learning_rate




        lgb_train = lgb.Dataset(X_train, y_train)

        # specify your configurations as a dict
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'num_leaves': 63,
            'num_trees': num_trees,
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': max_depth

        }

        # number of leaves,will be used in feature transformation
        num_leaf = 63

        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_train,
                        verbose_eval=False)

        y_pred = gbm.predict(X_train, pred_leaf=True)

        X_train_leaves_lgb = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
            X_train_leaves_lgb[i][temp] += 1

        y_pred = gbm.predict(X_test, pred_leaf=True)

        X_test_leaves_lgb = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
            X_test_leaves_lgb[i][temp] += 1

        return X_train_leaves_lgb, X_test_leaves_lgb


    def get_leaf_indices(self, ensemble, x):
        x = x.astype(np.float32)
        trees = ensemble.estimators_
        n_trees = trees.shape[0]
        indices = []

        for i in range(n_trees):
            tree = trees[i][0].tree_
            indices.append(tree.apply(x))

        indices = np.column_stack(indices)
        return indices


    def GBCdt(self, n_trees=30, max_depth=7, learning_rate=0.1):

        X_train =self.x_train
        y_train = self.y_train
        X_test = self.x_test


        num_trees = n_trees
        max_depth = max_depth
        learning_rate = learning_rate

        gbclf = GradientBoostingClassifier(n_estimators=num_trees, max_depth=max_depth, verbose=0, learning_rate=learning_rate)
        gbclf.fit(X_train, y_train)
        leaf = self.get_leaf_indices

        y_pred = leaf(gbclf, X_train.values)
        num_leaf = np.max(y_pred)
        X_train_leaves_gbc = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
            X_train_leaves_gbc[i][temp] += 1

        y_pred = leaf(gbclf, X_test.values)
        X_test_leaves_gbc = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
        for i in range(0, len(y_pred)):
            temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
            X_test_leaves_gbc[i][temp] += 1

        return X_train_leaves_gbc, X_test_leaves_gbc

class  StackingFeatures:


    def GBDTstack(self, X_train_leaves_xgb, X_test_leaves_xgb, X_train_leaves_lgb, X_test_leaves_lgb, X_train_leaves_gbc, X_test_leaves_gbc):
        print('Stacking Features')
        X_train_leaves_xgb, X_test_leaves_xgb = X_train_leaves_xgb, X_test_leaves_xgb
        X_train_leaves_lgb, X_test_leaves_lgb = X_train_leaves_lgb, X_test_leaves_lgb
        X_train_leaves_gbc, X_test_leaves_gbc = X_train_leaves_gbc, X_test_leaves_gbc


        X_train_leaves = np.hstack([X_train_leaves_xgb, X_train_leaves_lgb, X_train_leaves_gbc])
        X_test_leaves = np.hstack([X_test_leaves_xgb, X_test_leaves_lgb, X_test_leaves_gbc])

        return X_train_leaves, X_test_leaves

    def AddFeature(self, X_train_leaves, X_test_leaves):
        print('Adding Original Feature')
        X_train_ext = hstack([X_train_leaves, X_train])
        X_test_ext = hstack([X_test_leaves, X_test])

        return X_train_ext, X_test_ext


class ClassifyingScore:

    def __init__(self, x_train, y_train, x_test, y_test, clf='lr'):
        self.bestC = 0
        self.auc_best = 0
        self.acc = 0
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        if clf == 'lr':
            self.LRscore()
        elif clf == 'nb':
            self.NBscore()
        elif clf == 'svc':
            self.SVCscore()
        elif clf == 'knn':
            self.KNNscore()
        elif clf == 'Perc':
            self.Pscore()
        elif clf == 'lsv':
            self.lSVCscore()
        elif clf == 'sgd':
            self.SGDscore()
        elif clf == 'xgd':
            self.XGBscore()
        elif clf == 'lgm':
            self.LightGBMscore()

    def LRscore(self):
        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        auc_best = 0

        # ---------------------------------------------------------------------------------------------
        # regularization applied testing
        c = np.array([1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
        for t in range(0, len(c)):
            lm = LogisticRegression(penalty='l2', C=c[t])  # logestic model construction
            lm.fit(X_train_leaves, y_train)  # fitting the data

            y_pred_est = lm.predict_proba(X_test_leaves)  # Give the probabilty on each label

            auc = roc_auc_score(y_test, y_pred_est[:, 1])
            if auc_best < auc:
                auc_best = auc
                # bestC = c[t]
                # acc = accuracy_score(y_test, y_pred_est[:, 1].round())

        # ---------------------------------------------------------------------------------------------

        # print('best C value: %.2f' % bestC)
        print('GBDT+LR auc : %.5f' % auc_best)
        # print('GBDT+LR accuracy: %.5f' % acc)

    def NBscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        gnb = GaussianNB()

        gnb.fit(X_train_leaves, y_train)
        Y_pred_nb = gnb.predict_proba(X_test_leaves)[:, 1]
        gnb_auc = roc_auc_score(y_test, Y_pred_nb)
        print('GBDT + GNB auc: %.5f' % gnb_auc)

    def SVCscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        svc = SVC(probability=True)
        svc.fit(X_train_leaves, y_train)
        Y_pred_svc = svc.predict_proba(X_test_leaves)[:, 1]
        svc_auc = roc_auc_score(y_test, Y_pred_svc)
        print('GBDT + SVC auc: %.5f' % svc_auc)

    def KNNscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_leaves, y_train)
        Y_pred_knn = knn.predict_proba(X_test_leaves)[:, 1]
        knn_auc = roc_auc_score(y_test, Y_pred_knn)
        print('GBDT + KNN auc : %.5f' % knn_auc)

    def Pscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        perceptron = Perceptron()
        perceptron.fit(X_train_leaves, y_train)
        y_pred_perc = perceptron.predict(X_test_leaves)
        perc_auc = roc_auc_score(y_test, y_pred_perc)
        print('GBDT + Perceptron auc : %.5f' % perc_auc)

    def lSVCscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        lin = LinearSVC()
        lin.fit(X_train_leaves, y_train)
        y_pred_lin = lin.predict(X_test_leaves)
        lin_auc = roc_auc_score(y_test, y_pred_lin)
        print('GBDT + Linear SVC auc : %.5f' % lin_auc)

    def SGDscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        sgd = SGDClassifier(loss='log')
        sgd.fit(X_train_leaves, y_train)
        Y_pred_sgd = sgd.predict_proba(X_test_leaves)[:, 1]
        sgd_auc = roc_auc_score(y_test, Y_pred_sgd)
        print('GBDT + SGD auc : %.5f' % sgd_auc)

    def XGBscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test
        xgb = XGBClassifier()
        xgb.fit(X_train_leaves, y_train)
        Y_pred_xgb = xgb.predict(X_test_leaves)
        xgb_auc = roc_auc_score(y_test, Y_pred_xgb)
        print('GBDT + XGB auc: %.5f' % xgb_auc)

    def LightGBMscore(self):

        X_train_leaves = self.x_train
        y_train = self.y_train
        X_test_leaves = self.x_test
        y_test = self.y_test

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'l2', 'auc'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0

        }

        lgb_train = lgb.Dataset(X_train_leaves, y_train)
        lgb_eval = lgb.Dataset(X_test_leaves, y_test, reference=lgb_train)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=20,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=5,
                        verbose_eval=False)
        y_pred_lgb2 = gbm.predict(X_test_leaves, num_iteration=gbm.best_iteration)
        lgb_auc2 = roc_auc_score(y_test, y_pred_lgb2)

        print('GBDT + lightGBM auc : %.5f' % lgb_auc2)


