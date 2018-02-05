from GBDT import *
import pandas as pd
from sklearn.model_selection import train_test_split


example=pd.read_csv('example2.csv')
train_df, test_df = train_test_split(example, train_size = 0.75)
X_train = train_df.drop(['Grant.Status'], axis=1)
y_train = train_df['Grant.Status']
X_test = test_df.drop(['Grant.Status'], axis=1)
y_test = test_df['Grant.Status']

gbdt=GBDT(X_train,y_train,X_test,y_test)

X_train_leaves, X_test_leaves = gbdt.XGBdt()
clf = ClassifyingScore(X_train_leaves, y_train, X_test_leaves, y_test)

score = clf.LRscore()