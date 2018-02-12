from code import *
def getDataInput() : 
  print("User Input for data")
  print("1 : use Data1,  2 : use Data2,  3 : include parameter in optimizer")

  UserInput = int(input())
  while ((UserInput != 1) & (UserInput != 2) & (UserInput != 3)) : 
    print("Wrong input value, type again")
    print("1 : use Data1,  2 : use Data2,  3 : include parameter in optimizer")
    UserInput = int(input())

  return UserInput

 
def getTrainInput() : 
  print("User Input for Training Model")
  print("1 : XGBoost,  2 : LightGBM")
  UserInput = int(input())
  while ((UserInput != 1) & (UserInput != 2)) : 
    print(str(UserInput))
    print("Wrong input value, type again")
    print("1 : XGBoost,  2 : LightGBM")
    UserInput = int(input())
  return UserInput


def getOptInput() : 
  print("User Input for Optimizer")
  print("1 : use default parameters,  2 : GridSearch,  3 : Bayesian Optimization,  4 : Differential Evolution")
  UserInput = int(input())
  while ((UserInput != 1) & (UserInput != 2) & (UserInput != 3) & (UserInput != 4)) : 
    print("Wrong input value, type again")
    print("1 : No Optimizer(use default parameters),  2 : GridSearch,  3 : Bayesian Optimization,  4 : Differential Evolution")
    UserInput = int(input())
  return UserInput


def loadData(dataInput) : 
  file_name = ""
  if dataInput == 1 : 
    file_name = "../data/train_preprocessed1.csv"
  else : 
    file_name = "../data/train_preprocessed2.csv"
  train_df = pd.read_csv(file_name, low_memory = False)
  return train_df 

def xgboostTrain(dataInput) : 
  train_df = loadData(dataInput)  
  #data, target preparation
  data_df = train_df.drop(['Grant.Status'], axis = 1)
  target_df = train_df['Grant.Status']
  data = data_df.values
  target = target_df.values
  model = xgb.XGBClassifier(eval_metric = 'auc')

  kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
  results = cross_val_score(model, data, target, cv = kfold)
  accuracy = results.mean()*100
  print("Using Default Parameters, XGB Score is ")
  print("AUC : %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
  return
 
def lightgbmTrain(dataInput) : 
  train_df = loadData(dataInput)  
  #data, target preparation
  data_df = train_df.drop(['Grant.Status'], axis = 1)
  target_df = train_df['Grant.Status']
  data = data_df.values
  target = target_df.values

  lgb_train = lgb.Dataset(data, target)
  lgb_params = {
    'task': 'train',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'max_depth' : 6,
    'learning_rate' : 0.03,
    'reg_lambda' : 1.0
  }
  model = lgb.LGBMClassifier(**lgb_params)
    
  kfold = KFold(n_splits = 5, random_state = 7, shuffle = True)
  results = cross_val_score(model, data, target, cv = kfold)
  auc = results.mean()*100
  print("Using Default Parameters, LGBM Score is ")
  print("AUC : %.2f%% (%.2f%%)" % (auc, results.std()*100))
  return


def mainRun() : 
  dataInput = getDataInput()
  trainInput = getTrainInput()
  optInput = getOptInput()

  if optInput == 4 :
    DiffEv.run(dataInput, trainInput)
  elif optInput == 3 :
    BayesOpt.run(dataInput, trainInput)
  elif optInput == 2 :
    GridSearch.run(dataInput, trainInput)
  else : 
    if dataInput == 3 : 
      print("If you are not using Optimizer, you need to select data 1 or 2")
      dataInput = int(input()) 
      while ((dataInput != 1) & (dataInput != 2)) : 
        print("Wrong input value, type again(1 or 2)")
        dataInput = int(input())
    else : 
      if trainInput == 1 : 
        xgboostTrain(dataInput)
      else : 
        lightgbmTrain(dataInput)
  print("Type 0 to exit, anything else to continue")
  exitInput = input()
  while (exitInput != "0") : 
    mainRun()
    print("Type 0 to exit, anything else to continue")
    exitInput = input()


