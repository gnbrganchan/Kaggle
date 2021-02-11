import pickle
import base64
import random


import numpy as np
import pandas as pd
import sklearn.tree as skt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

# Parameters
FUDGE_FACTOR = 0.99
VERBOSE = False

DATA_FILE_LB = '/kaggle/input/generate-training-data-mine/training_data_mine.parquet'


TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = ['payout']

rand = 1024

def make_model(X_tr,y_tr,X_va,y_va):
    """Builds a decision tree model based on stored trainingd data"""
#     X_1, X_2, y_1, y_2 = train_test_split(X_tr, y_tr, test_size = 0.25, random_state = rand)
    X_1, X_2, y_1, y_2 = X_tr, X_va, y_tr, y_va
    

    model = lgb.LGBMRegressor(
        learning_rate = 0.0584,
        max_depth = 11,
        min_child_samples = 154,
        n_estimators = 2100,
        num_leaves = 182,
        reg_lambda = 89,
        subsample = 0.7666,
        first_metric_only = True
    )
    model.fit(X_1,y_1,
             eval_set=[(X_1,y_1),(X_2,y_2)],
             eval_metric='rmse',
             early_stopping_rounds=50)
    return model
##

## prepare data
data = pd.read_parquet(DATA_FILE_LB,columns=TRAIN_FEATS+TARGET_COL)

print(data.info())

data = data.query("n_pulls_self > 0 or n_pulls_opp > 1").reset_index().drop(["index"],axis=1)

print(data.info())


nsplits = 5
folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)
X_train = data[TRAIN_FEATS]
y_train = data[TARGET_COL]
##

# CV training
rmse = []
mae = []
i = 0
for tr_idx,va_idx in folds.split(X_train,y_train):
    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_va = y_train.iloc[va_idx]
    
    model = make_model(X_tr,y_tr,X_va,y_va)
    y_pred = model.predict(X_va,num_iteration = model.best_iteration_)

    rmse.append(np.sqrt(mean_squared_error(y_va, y_pred)))
    mae.append(mean_absolute_error(y_va, y_pred))
    i += 1
    
    #save model
    pickle.dump(model, open(f'model{i}.sav', 'wb'))
##
!tar cvfz main.py.tar.gz main.py model1.sav model2.sav model3.sav model4.sav model5.sav
