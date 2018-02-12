from . import *
__all__ = ['BayesOpt']
#load data

file_name = "../data/train_preprocessed1.csv"
train_df1 = pd.read_csv(file_name, low_memory = False, index_col = False)
train_df1.drop(train_df1.columns[0], axis = 1, inplace = True)
file_name = "../data/train_preprocessed2.csv"
train_df2 = pd.read_csv(file_name, low_memory = False)

#divide target and data

data_df1 = train_df1.drop(['Grant.Status'], axis = 1)
target_df1 = train_df1['Grant.Status']

data_df2 = train_df2.drop(['Grant.Status'], axis = 1)
target_df2 = train_df2['Grant.Status']

data1 = data_df1.values
target1 = target_df1.values

data2 = data_df2.values
target2 = target_df2.values

# XGB Train Model using BayesOpt

def XGB_TrainingModel_forBayesOpt(preprocessing, min_child_weight, max_depth, gamma, subsample, colsample_bytree):
    xgb_params = {
        'n_trees': 20,
        'eta': 0.2,
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'objective': 'reg:linear',
        'eval_metric': 'auc',
        'silent': 1,
        'min_child_weight': int(min_child_weight),
        'gamma': max(gamma, 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0)
    }
    preprocessing = round(preprocessing)
    if preprocessing == 1:
        data = data1
        target = target1
    else:
        data = data2
        target = target2
    model = xgb.XGBClassifier(**xgb_params)

    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
    results = cross_val_score(model, data, target, cv=kfold)
    auc = results.mean() * 100
    print("AUC : %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    return auc


def LGB_TrainingModel_forBayesOpt(preprocessing, gamma, max_depth, min_child_weight, colsample_bytree, subsample):

    lgb_params = {

        # static parameters
        'task': 'train',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'learning_rate': 0.03,
        'reg_lambda': 1.0,
        'num_leaves': 1023,
        'verbose' : -1,

        # tuned parameters
        'max_depth': int(max_depth),
        'min_child_weight': int(min_child_weight),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'subsample': max(min(subsample, 1), 0),
        'min_split_gain': max(gamma, 0),
    }
    preprocessing = round(preprocessing)
    if preprocessing == 1:
        data = data1
        target = target1
    else:
        data = data2
        target = target2
    model = lgb.LGBMClassifier(**lgb_params)

    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
    results = cross_val_score(model, data, target, cv=kfold)
    auc = results.mean() * 100
    print("AUC : %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    return auc

def run(dataInput, TrainInput) :

    #Set parameter Range

    if dataInput == 3 :
        params = {
            'preprocessing': (1, 2),
            'min_child_weight': (1, 20),
            'max_depth': (2, 10),
            'gamma': (0, 10),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.1, 1)
        }
    elif dataInput == 2 :
        params = {
            'preprocessing': (2,2),
            'min_child_weight': (1, 20),
            'max_depth': (2, 10),
            'gamma': (0, 10),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.1, 1)
        }
    elif dataInput == 1 :
        params = {
            'preprocessing': (1,1),
            'min_child_weight': (1, 20),
            'max_depth': (2, 10),
            'gamma': (0, 10),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.1, 1)
        }

    else :
        print("wrong User dataInput")
        params = {
            'preprocessing': 2,
            'min_child_weight': (1, 20),
            'max_depth': (2, 10),
            'gamma': (0, 10),
            'subsample': (0.5, 1),
            'colsample_bytree': (0.1, 1)
        }

    print("Set number of Inital Points for optimizations(5~10 is recommended value)")
    n_init_points = int(input())
    print("Set number of Bayesian Iterations(50~100 is recommended value)")
    n_iters = int(input())

    if TrainInput == 1 :
        bayesOPT = BayesianOptimization(XGB_TrainingModel_forBayesOpt, params)
    else :
        bayesOPT = BayesianOptimization(LGB_TrainingModel_forBayesOpt, params)
    start_time = time.time()
    bayesOPT.maximize(init_points=n_init_points, n_iter=n_iters)
    elapsed_time = time.time() - start_time
    print("elapsed time : %s min %s sec" % (int(elapsed_time / 60), int(elapsed_time % 60)))
