import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import time

#load data

file_name = "./data/train_preprocessed2.csv"
train_df = pd.read_csv(file_name, low_memory = False)


#Setup data

array = train_df.values
data = array[:, 0:70]
target = array[:, 70]
seed = 7
test_size = 0.2

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = test_size, random_state = seed)



#Set XGBClassifier

xgb_clf = xgb.XGBClassifier(eval_metric = 'auc', n_trees = 250)

xgb_params = {
    'learning_rate' : np.arange(0.01, 0.20, 0.01),
    'min_child_weight' : np.arange(1, 20, 1), 
    'max_depth' : np.arange(2, 10, 1),
    'gamma' : np.arange(0, 10, 1), 
    'subsample' : np.arange(0.5, 1.0, 0.1),
    'colsample_bytree' : np.arange(0.1, 1.0, 0.1),
    'objective' : ['reg:linear'],
    'silent' : [1],
}


#Set GridSearchCV
GSCV = GridSearchCV(xgb_clf, xgb_params, cv=5, scoring = 'roc_auc', n_jobs = 1, verbose = 2)



#Running GridSearch
start_time = time.time()
GSCV.fit(data, target)
elapsed_time = time.time() - start_time

print("%s seconds elapsed"%elapsed_time)
best = rs.best_estimator_
print(best)






