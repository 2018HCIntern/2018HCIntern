from . import *
__all__ = ['DiffEv']

#Set Max Generation Count
max_generation = 100


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

BestAUCResult = np.zeros(100)
tmpAUCResult = np.zeros(1500)
itercnt = 0
iteration_range = 0


# XGB Train Model

def XGB_TrainingModel_forDiffEv(params):
    min_child_weight = params[0]
    max_depth = params[1]
    gamma = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    preprocessing = params[5]
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

    ##############################Result Saving Part#########################################
    global max_generation
    global BestAUCResult
    global tmpAUCResult
    global itercnt
    global iteration_range

    generation = int(itercnt / iteration_range)
    population = itercnt % iteration_range
    tmpAUCResult[population] = auc
    #print(itercnt)
    if ( ((iteration_range - population) == 1) & ( generation <= len(BestAUCResult))) :
        BestAUCResult[generation] = np.amax(tmpAUCResult)
        print("The Best AUC result for Generation %d is : %2f%%" %(generation, BestAUCResult[generation]))

    itercnt = itercnt + 1
    ##############################Result Saving Part#########################################
    #print("generation %d member %d result(AUC) : %2f%%" %(generation, population, auc))
    return 100 - auc

def LGB_TrainingModel_forDiffEv(params):
    min_child_weight = params[0]
    max_depth = params[1]
    gamma = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    preprocessing = params[5]
    lgb_params = {
        # static parameters
        'task': 'train',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'learning_rate': 0.03,
        'reg_lambda': 1.0,
        'num_leaves': 1023,
        'verbose': -1,

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

    ##############################Result Saving Part#########################################
    global max_generation
    global BestAUCResult
    global tmpAUCResult
    global itercnt
    global iteration_range

    generation = int(itercnt / iteration_range)
    population = itercnt % iteration_range
    tmpAUCResult[population] = auc

    if ( ((iteration_range - population) == 1) & ( generation < len(BestAUCResult))) :
        BestAUCResult[generation] = np.amax(tmpAUCResult)
        print("The Best AUC result for Generation %d is : %2f%%" %(generation, BestAUCResult[generation]))

    itercnt = itercnt + 1
    ##############################Result Saving Part#########################################

    return 100 - auc

def run(dataInput, TrainInput) :
    # Set parameter Range

    if dataInput == 3:
        bounds = [(1,2), (1, 20), (2, 10), (0, 10), (0.5, 1), (0.1, 1)]

    elif dataInput == 2:
        bounds = [(2, 2), (1, 20), (2, 10), (0, 10), (0.5, 1), (0.1, 1)]

    elif dataInput == 1:
        bounds = [(1, 1), (1, 20), (2, 10), (0, 10), (0.5, 1), (0.1, 1)]

    else:
        print("wrong User dataInput, we will use data 2")
        bounds = [(2, 2), (1, 20), (2, 10), (0, 10), (0.5, 1), (0.1, 1)]


    global max_generation
    global BestAUCResult
    global tmpAUCResult
    global itercnt
    global iteration_range

    print("Set generation number(max value :100)")
    generation_input = int(input())
    while generation_input > 100 :
        print("Input exceed max generation number(100), please type agian")
        generation_input = int(input())
    print("Set number of population members : (10~50 is recommended value)")
    popsize_input = int(input())

    itercnt = 0
    iteration_range = len(bounds) * popsize_input

    BestAUCResult = np.zeros(generation_input+1)
    tmpAUCResult = np.zeros(iteration_range)

    start_time = time.time()
    if TrainInput == 1 :
        result = differential_evolution(XGB_TrainingModel_forDiffEv, bounds, maxiter=generation_input, popsize=popsize_input)
    else :
        result = differential_evolution(LGB_TrainingModel_forDiffEv, bounds, maxiter=generation_input, popsize=popsize_input)

    elapsed_time = time.time() - start_time
    print("elapsed time : %s min %s sec" % (elapsed_time / 60, elapsed_time % 60))



