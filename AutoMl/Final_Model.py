import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
import gc
from sklearn.grid_search import GridSearchCV
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

file_name = "./data/train_preprocessed1.csv"
train_df1 = pd.read_csv(file_name, low_memory = False, index_col = False)

train_df1.drop(train_df1.columns[0], axis = 1, inplace = True)
file_name = "./data/train_preprocessed2.csv"
train_df2 = pd.read_csv(file_name, low_memory = False)

data_df1 = train_df1.drop(['Grant.Status'], axis = 1)
target_df1 = train_df1['Grant.Status']

data_df2 = train_df2.drop(['Grant.Status'], axis = 1)
target_df2 = train_df2['Grant.Status']

data1 = data_df1.values
target1 = target_df1.values
data2 = data_df2.values
target2 = target_df2.values

cnt = 0

max_depth_BO = np.zeros(205)
min_child_weight_BO = np.zeros(205)
colsample_bytree_BO = np.zeros(205)
subsample_BO = np.zeros(205)
gamma_BO = np.zeros(205)
auc_BO = np.zeros(205)

max_depth_BO2 = np.zeros(205)
min_child_weight_BO2 = np.zeros(205)
colsample_bytree_BO2 = np.zeros(205)
subsample_BO2 = np.zeros(205)
gamma_BO2 = np.zeros(205)
auc_BO2 = np.zeros(205)

max_depth_DE = np.zeros(1500)
min_child_weight_DE = np.zeros(1500)
colsample_bytree_DE = np.zeros(1500)
subsample_DE = np.zeros(1500)
gamma_DE = np.zeros(1500)
auc_DE = np.zeros(1500)

max_depth_DE2 = np.zeros(1500)
min_child_weight_DE2 = np.zeros(1500)
colsample_bytree_DE2 = np.zeros(1500)
subsample_DE2 = np.zeros(1500)
gamma_DE2 = np.zeros(1500)
auc_DE2 = np.zeros(1500)

def XGB_Train_Model(min_child_weight, max_depth, gamma, subsample, colsample_bytree) : 
    xgb_params = {
        #static parameters
        'n_trees' : 20,
        'eta' : 0.3,
        'objective' : 'reg:linear', 
        'eval_metric' : 'auc',
        'silent' : 1,
        
        #tuned parameters
        'max_depth' : int(max_depth),
        'subsample' : max(min(subsample, 1), 0),
        'min_child_weight' : int(min_child_weight),
        'gamma' : max(gamma, 0), 
        'colsample_bytree' : max(min(colsample_bytree, 1), 0)
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    
    kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
    results = cross_val_score(model, data2, target2, cv = kfold)
    auc = results.mean()*100
    return auc

xgb_clf = xgb.XGBClassifier(eval_metric = 'auc', n_trees = 20)

xgb_params = {
    'learning_rate' : [0.3],
    'min_child_weight' : np.arange(1, 20, 5),      # 4
    'max_depth' : np.arange(2, 10, 2),             # 4 
    'gamma' : np.arange(0, 10, 2.5),                 # 4
    'subsample' : np.arange(0.5, 1.0, 0.125),        # 4
    'colsample_bytree' : np.arange(0.1, 1.0, 0.3), # 3
    'objective' : ['reg:linear'],
    'silent' : [1],
}

GSCV = GridSearchCV(xgb_clf, xgb_params, cv = 5, scoring = 'roc_auc', n_jobs = 1, verbose = 1)

start_time = time.time()
GSCV.fit(data2, target2)
elapsed_time = time.time() - start_time
print("elapsed time : %s min %s sec".format(elapsed_time/60, elapsed_time%60))
best_parameters, score, _ = max(GSCV.grid_scores_, key=lambda x: x[1])
print('best parameters:', best_parameters)



XGB_Train_Model(best_parameters)


def LGB_Train_Model(gamma, max_depth, min_child_weight, colsample_bytree, subsample) :
    lgb_train = lgb.Dataset(data2, target2)
    
    lgb_params = {
        
    #static parameters
    'task': 'train',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'learning_rate' : 0.03,
    'reg_lambda' : 1.0,
    'num_leaves' : 1023,
        
    #tuned parameters
    'max_depth': int(max_depth),
    'min_child_weight' : int(min_child_weight),
    'colsample_bytree' : max(min(colsample_bytree, 1), 0),
    'subsample' : max(min(subsample, 1), 0),
    'gamma' : max(gamma, 0), 
    }

    model = lgb.LGBMClassifier(**lgb_params)
    
    kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
    results = cross_val_score(model, data2, target2, cv = kfold)
    auc = results.mean()*100
    return auc

lgb_clf = lgb.LGBMClassifier(task = 'train', metric = {'l2', 'auc'}, objective = 'regression', 
                            learning_rate = 0.03, reg_lambda = 1.0, num_leaves = 1023)

lgb_params = {
    'min_child_weight' : np.arange(1, 20, 5),      # 4
    'max_depth' : np.arange(2, 10, 2),             # 4 
    'gamma' : np.arange(0, 10, 2.5),                 # 4
    'subsample' : np.arange(0.5, 1.0, 0.125),        # 4
    'colsample_bytree' : np.arange(0.1, 1.0, 0.3), # 3
}

GSCV2 = GridSearchCV(lgb_clf, lgb_params, cv = 5, scoring = 'roc_auc', n_jobs = 1, verbose = 1)

start_time = time.time()
GSCV.fit(data, target)
elapsed_time = time.time() - start_time
print("elapsed time : %s min %s sec".format(elapsed_time/60, elapsed_time%60))
best_parameters, score, _ = max(GSCV2.grid_scores_, key=lambda x: x[1])
print('best parameters:', best_parameters)



XGB_Train_Model(best_parameters)



def XGB_Train_Model_BO(min_child_weight, max_depth, gamma, subsample, colsample_bytree) : 
    xgb_params = {
        #static parameters
        'n_trees' : 20,
        'eta' : 0.3,
        'objective' : 'reg:linear', 
        'eval_metric' : 'auc',
        'silent' : 1,
        
        #tuned parameters
        'max_depth' : int(max_depth),
        'subsample' : max(min(subsample, 1), 0),
        'min_child_weight' : int(min_child_weight),
        'gamma' : max(gamma, 0), 
        'colsample_bytree' : max(min(colsample_bytree, 1), 0)
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    
    kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
    results = cross_val_score(model, data2, target2, cv = kfold)
    auc = results.mean()*100
    
    ##############################plot parameter saving part#########################################
    global cnt
    global max_depth_BO, subsample_BO, min_child_weight_BO, gamma_BO, colsample_bytree_BO, auc_BO
    max_depth_BO[cnt]        = max_depth
    subsample_BO[cnt]        = subsample
    min_child_weight_BO[cnt] = min_child_weight
    gamma_BO[cnt]            = gamma
    subsample_BO[cnt]        = subsample
    auc_BO[cnt]              = auc
    cnt = cnt + 1
    ##############################plot parameter saving part#########################################    
    
    print("AUC : %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    return auc


xgb_params = {
    
    #Minimum sum of weights : to control overfitting
    'min_child_weight' : (1, 20), 
    
    #Maximum depth of a tree : to control overfitting
    'max_depth' : (2, 10),
    
    #minimum loss reduction required to make a split : makes algorithm conservative
    'gamma' : (0, 10), 
    
    #Fraction of observations to be randomly samples for each tree
    #Lower: prevent overfitting
    'subsample' : (0.5, 1),
    
    #Fraction of columns to be randomly samples for each tree
    'colsample_bytree' : (0.1, 1),
    
    }


xgb_bayesOPT = BayesianOptimization(XGB_Train_Model_BO, xgb_params)
start_time = time.time()
xgb_bayesOPT.maximize(init_points = 5, n_iter = 200)
elapsed_time = time.time() - start_time
print("elapsed time : %s min %s sec".format(elapsed_time/60, elapsed_time%60))
cnt = 0
xgbBO_data = {'min_child_weight' : min_child_weight_BO, 'max_depth' : max_depth_BO, 'gamma' : gamma_BO, 'subsample' : subsample_BO, 'colsample_bytree' : colsample_bytree_BO, 'auc' : auc_BO}
xgbBO_df = pd.DataFrame(data = xgbBO_data) 
xgbBO_df.to_csv("xgbBO.csv", sep = ',')

def LGB_Train_Model_BO(gamma, max_depth, min_child_weight, colsample_bytree, subsample) :
    lgb_train = lgb.Dataset(data2, target2)
    
    lgb_params = {
        
    #static parameters
    'task': 'train',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'learning_rate' : 0.03,
    'reg_lambda' : 1.0,
    'num_leaves' : 1023,
        
    #tuned parameters
    'max_depth': int(max_depth),
    'min_child_weight' : int(min_child_weight),
    'colsample_bytree' : max(min(colsample_bytree, 1), 0),
    'subsample' : max(min(subsample, 1), 0),
    'gamma' : max(gamma, 0), 
    }

    model = lgb.LGBMClassifier(**lgb_params)
    
    kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
    results = cross_val_score(model, data2, target2, cv = kfold)
    auc = results.mean()*100
    
    ##############################plot parameter saving part#########################################
    global cnt, optimizer
    global max_depth_BO2, subsample_BO2, min_child_weight_BO2, gamma_BO2, colsample_bytree_BO2, auc_BO2
    max_depth_BO2[cnt]        = max_depth
    subsample_BO2[cnt]        = subsample
    min_child_weight_BO2[cnt] = min_child_weight
    gamma_BO2[cnt]            = gamma
    subsample_BO2[cnt]        = subsample
    auc_BO2[cnt]              = auc
    cnt = cnt + 1
    ##############################plot parameter saving part#########################################
    print("AUC : %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    
    return auc


lgb_params = {
    'max_depth' : (2, 10), 
    'min_child_weight' : (1, 20), 
    'colsample_bytree' : (0.1, 1), 
    'subsample' : (0.5, 1),
    'gamma' : (0, 10)
}

lgb_bayesOPT = BayesianOptimization(LGB_Train_Model_BO, lgb_params)
start_time = time.time()
lgb_bayesOPT.maximize(init_points = 5, n_iter = 200)
elapsed_time = time.time() - start_time
print("elapsed time : %s min %s sec".format(elapsed_time/60, elapsed_time%60))
cnt = 0

lgbBO_data = {'min_child_weight' : min_child_weight_BO2, 'max_depth' : max_depth_BO2, 'gamma' : gamma_BO2, 'subsample' : subsample_BO2, 'colsample_bytree' : colsample_bytree_BO2, 'auc' : auc_BO2}
lgbBO_df = pd.DataFrame(data = lgbBO_data) 
lgbBO_df.to_csv("lgbBO.csv", sep = ',')

def XGB_Train_Model_DE(params) : 
    min_child_weight = params[0]
    max_depth = params[1]
    gamma = params[2]
    subsample = params[3] 
    colsample_bytree = params[4]
    xgb_params = {
        #static parameters
        'n_trees' : 20,
        'eta' : 0.3,
        'objective' : 'reg:linear', 
        'eval_metric' : 'auc',
        'silent' : 1,
        
        #tuned parameters
        'max_depth' : int(max_depth),
        'subsample' : max(min(subsample, 1), 0),
        'min_child_weight' : int(min_child_weight),
        'gamma' : max(gamma, 0), 
        'colsample_bytree' : max(min(colsample_bytree, 1), 0)
    }
    
    model = xgb.XGBClassifier(**xgb_params)
    
    kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
    results = cross_val_score(model, data2, target2, cv = kfold)
    auc = results.mean()*100
    
    ##############################plot parameter saving part#########################################
    global cnt
    global max_depth_DE, subsample_DE, min_child_weight_DE, gamma_DE, colsample_bytree_DE, auc_DE
    max_depth_DE[cnt]        = max_depth
    subsample_DE[cnt]        = subsample
    min_child_weight_DE[cnt] = min_child_weight
    gamma_DE[cnt]            = gamma
    subsample_DE[cnt]        = subsample
    auc_DE[cnt]              = auc
    cnt = cnt + 1
    ##############################plot parameter saving part#########################################    
    
    print("AUC : %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    return 100 - auc


bounds = [(1,20), (2, 10), (0, 10), (0.5, 1), (0.1, 1)]
start_time = time.time()
result = differential_evolution(XGB_Train_Model_DE, bounds, maxiter = 10, popsize = 30)
elapsed_time = time.time() - start_time
print("elapsed time : %s min %s sec".format(elapsed_time/60, elapsed_time%60))
cnt = 0

print("diff evolution using xgboost result : " + result.x + result.fun)
xgbDE_data = {'min_child_weight' : min_child_weight_DE, 'max_depth' : max_depth_DE, 'gamma' : gamma_DE, 'subsample' : subsample_DE, 'colsample_bytree' : colsample_bytree_DE, 'auc' : auc_DE}
xgbDE_df = pd.DataFrame(data = xgbDE_data) 
xgbDE_df.to_csv("xgbDE.csv", sep = ',')

def LGB_Train_Model_DE(params) :
    min_child_weight = params[0]
    max_depth = params[1]
    gamma = params[2]
    subsample = params[3] 
    colsample_bytree = params[4]
    lgb_train = lgb.Dataset(data2, target2)
    
    lgb_params = {
        
    #static parameters
    'task': 'train',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'learning_rate' : 0.03,
    'reg_lambda' : 1.0,
    'num_leaves' : 1023,
        
    #tuned parameters
    'max_depth': int(max_depth),
    'min_child_weight' : int(min_child_weight),
    'colsample_bytree' : max(min(colsample_bytree, 1), 0),
    'subsample' : max(min(subsample, 1), 0),
    'gamma' : max(gamma, 0), 
    }

    model = lgb.LGBMClassifier(**lgb_params)
    
    kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
    results = cross_val_score(model, data2, target2, cv = kfold)
    auc = results.mean()*100
    
    ##############################plot parameter saving part#########################################
    global cnt
    global max_depth_DE2, subsample_DE2, min_child_weight_DE2, gamma_DE2, colsample_bytree_DE2, auc_DE2
    max_depth_DE2[cnt]        = max_depth
    subsample_DE2[cnt]        = subsample
    min_child_weight_DE2[cnt] = min_child_weight
    gamma_DE2[cnt]            = gamma
    subsample_DE2[cnt]        = subsample
    auc_DE2[cnt]              = auc
    cnt = cnt + 1
    ##############################plot parameter saving part#########################################
    print("AUC : %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    return 100 - auc



#LGB Result (using DiffEvolution, Optimized Parameter)
bounds = [(1,20), (2, 10), (0, 10), (0.5, 1), (0.1, 1)]
start_time = time.time()
result = differential_evolution(LGB_Train_Model_DE, bounds, maxiter = 10, popsize = 30)
elapsed_time = time.time() - start_time
print("elapsed time : %s min %s sec".format(elapsed_time/60, elapsed_time%60))
cnt = 0

print("diff evolution using lightgbm result : " + result.x + result.fun)

lgbDE_data = {'min_child_weight' : min_child_weight_DE2, 'max_depth' : max_depth_DE2, 'gamma' : gamma_DE2, 'subsample' : subsample_DE2, 'colsample_bytree' : colsample_bytree_DE2, 'auc' : auc_DE2}
lgbDE_df = pd.DataFrame(data = lgbDE_data) 
lgbDE_df.to_csv("lgbDE.csv", sep = ',')
