from . import *
__all__ = ['GridSearch']
#
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


def XGB_Train_Model(preprocessing, min_child_weight, max_depth, gamma, subsample, colsample_bytree):
    xgb_params = {
        # static parameters
        'n_trees': 20,
        'eta': 0.3,
        'objective': 'reg:linear',
        'eval_metric': 'auc',
        'silent': 1,

        # tuned parameters
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'min_child_weight': int(min_child_weight),
        'gamma': max(gamma, 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0)
    }

    model = xgb.XGBClassifier(**xgb_params)
    if preprocessing == 1  :
        data = data1
        target = target1
    else :
        data = data2
        target = target2
    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
    results = cross_val_score(model, data, target, cv=kfold)
    auc = results.mean() * 100
    return auc


def LGB_Train_Model(preprocessing, min_child_weight, max_depth, gamma, subsample, colsample_bytree):


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

    model = lgb.LGBMClassifier(**lgb_params)
    if preprocessing == 1  :
        data = data1
        target = target1
    else :
        data = data2
        target = target2
    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
    results = cross_val_score(model, data, target, cv=kfold)
    auc = results.mean() * 100
    return auc

def run(dataInput, TrainInput) :

    #Set parameter Range
    xgb_params = {
        'min_child_weight': np.arange(1, 20, 5),  # 4
        'max_depth': np.arange(2, 10, 2),  # 4
        'gamma': np.arange(0, 10, 2.5),  # 4
        'subsample': np.arange(0.5, 1.0, 0.125),  # 4
        'colsample_bytree': np.arange(0.1, 1.0, 0.3)  # 3

    }
    lgb_params = {
        'min_child_weight': np.arange(1, 20, 5),  # 4
        'max_depth': np.arange(2, 10, 2),  # 4
        'min_split_gain': np.arange(0, 10, 2.5),  # 4
        'subsample': np.arange(0.5, 1.0, 0.125),  # 4
        'colsample_bytree': np.arange(0.1, 1.0, 0.3)  # 3

    }

    if dataInput == 1 :
        data = data1
        target = target1
    else :
        data = data2
        target = target2

    if TrainInput == 1 :
        model = xgb.XGBClassifier(eval_metric='auc', n_trees=20, learning_rate=0.3, objective='reg:linear', silent=1)
        GSCV = GridSearchCV(model, xgb_params, cv=5, scoring='roc_auc', n_jobs=1, verbose=1)
    else :
        model = lgb.LGBMClassifier(task='train', metric={'l2', 'auc'}, objective='regression', verbose = -1,
                                   learning_rate=0.03, reg_lambda=1.0, num_leaves=1023)
        GSCV = GridSearchCV(model, lgb_params, cv=5, scoring='roc_auc', n_jobs=1, verbose=1)
    start_time = time.time()

    GSCV.fit(data, target)
    elapsed_time = time.time() - start_time
    print("elapsed time : %s min %s sec" % (int(elapsed_time / 60), int(elapsed_time % 60)))
    best_parameters, score, _ = max(GSCV.grid_scores_, key=lambda x: x[1])
    print('best parameters:', best_parameters)

    print("Type in parameters for checking AUC scores")
    print("colsample_bytree : ")
    col_by = float(input())
    print("gamma : ")
    gamma = float(input())
    print("max_depth : ")
    max_dep = float(input())
    print("min_child_weight : ")
    min_ch = float(input())
    print("subsample : ")
    subsa = float(input())

    if TrainInput == 1 :
        result = XGB_Train_Model(dataInput, min_ch, max_dep, gamma, subsa, col_by)
    else :
        result = LGB_Train_Model(dataInput, min_ch, max_dep, gamma, subsa, col_by)

    print('Grid Search result(AUC) : '+ str(result))
    return
