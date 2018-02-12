import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score,accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
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


def get_leaf_indices(ensemble, x):
    x = x.astype(np.float32)
    trees = ensemble.estimators_
    n_trees = trees.shape[0]
    indices = []

    for i in range(n_trees):
        tree = trees[i][0].tree_
        indices.append(tree.apply(x))

    indices = np.column_stack(indices)
    return indices


def GBDT_lr(File):
    # XGBoost

    start = time.clock()
    train_df, test_df = train_test_split(File, train_size=0.75)
    X_train = train_df.drop(train_df.columns[0], axis=1)
    y_train = train_df[train_df.columns[0]]
    X_test = test_df.drop(test_df.columns[0], axis=1)
    y_test = test_df[test_df.columns[0]]

    dtrain = xgb.DMatrix(X_train.values, y_train.values)
    dtest = xgb.DMatrix(X_test.values, y_test.values)
    param = {'silent': 1, 'objective': 'binary:logistic', 'max_depth': 8}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 20
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

    # lightGBM

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 63,
        'num_trees': 30,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
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

    # GBC

    gbclf = GradientBoostingClassifier(n_estimators=30, max_depth=7, verbose=0)
    gbclf.fit(X_train, y_train)
    leaf = get_leaf_indices

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

    X_train_leaves = np.hstack([X_train_leaves_xgb, X_train_leaves_lgb, X_train_leaves_gbc])
    X_test_leaves = np.hstack([X_test_leaves_xgb, X_test_leaves_lgb, X_test_leaves_gbc])

    bestC = 0
    auc_best = 0
    acc = 0
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
            bestC = c[t]
            acc = accuracy_score(y_test, y_pred_est[:, 1].round())

    # ---------------------------------------------------------------------------------------------

    print('best C value: %.2f' % bestC)
    print('GBDT+LR auc : %.5f' % auc_best)
    print('GBDT+LR accuracy: %.5f' % acc)
    f_time = time.clock() - start
    print('GBDT+LR time taken: %.2f' % f_time)

    '''
    #-----------------------------------------------------------------------------
    #lightGBM
    start = time.clock()
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



    # specify your configurations as a dict
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0

    }
    X_train_ext=np.hstack([X_train_leaves,X_train])
    X_test_ext=np.hstack([X_test_leaves,X_test])

    lgb_train=lgb.Dataset(X_train_leaves, y_train)
    lgb_eval=lgb.Dataset(X_test_leaves, y_test, reference=lgb_train)
    gbm = lgb.train(params,
                  lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5,
                   verbose_eval=False)
    y_pred_lgb2 =gbm.predict(X_test_leaves, num_iteration=gbm.best_iteration)
    lgb_auc2=roc_auc_score(y_test, y_pred_lgb2)

    print('GBDT + lightGBM auc : %.5f' % lgb_auc2)

    lgb_train=lgb.Dataset(X_train_ext, y_train)
    lgb_eval=lgb.Dataset(X_test_ext, y_test, reference=lgb_train)
    gbm = lgb.train(params,
                  lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5,
                   verbose_eval=False)
    y_pred_lgb3 =gbm.predict(X_test_ext, num_iteration=gbm.best_iteration)
    lgb_auc3=roc_auc_score(y_test, y_pred_lgb3)

    print('GBDT + original + lightGBM auc : %.5f' % lgb_auc3)


    f_time=time.clock()-start
    print('lightGBM time taken: %.2f'% f_time)'''