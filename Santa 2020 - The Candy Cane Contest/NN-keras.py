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
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

# Parameters
FUDGE_FACTOR = 0.99
VERBOSE = False
DATA_FILE = '/kaggle/input/sample-training-data/training_data_201223.parquet'

TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = 'payout'

rand = 1024
##

data = pd.read_parquet(DATA_FILE)
X = data[TRAIN_FEATS].sample(n=1000000,random_state=0).reset_index().drop("index",axis = 1)
y = data[TARGET_COL].sample(n=1000000,random_state=0).reset_index().drop("index",axis = 1)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers

SEED = 46

    

def make_model(X_train,y_train,X_valid,y_valid):
    """Builds a decision tree model based on stored trainingd data"""
    
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear'),
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=10000,
        epochs=1000,
        callbacks=[early_stopping],
    )
    return model
##

rmse = []
i = 0

nsplits = 5
folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)

# pretrainer = pretrain_model(X,y)

for tr_idx,va_idx in folds.split(X,y):
    X_tr = X.iloc[tr_idx].reset_index().drop("index",axis = 1)
    y_tr = y.iloc[tr_idx].reset_index().drop("index",axis = 1)
    X_va = X.iloc[va_idx].reset_index().drop("index",axis = 1)
    y_va = y.iloc[va_idx].reset_index().drop("index",axis = 1)
    
    model = make_model(X_tr,y_tr,X_va,y_va)
    y_pred = model.predict(X_va.values)

    rmse.append(np.sqrt(mean_squared_error(y_va, y_pred)))
    i += 1
    
    #save model
#     pickle.dump(model, open(f'model{i}.sav', 'wb'))
#     torch.save(model,f'model{i}.sav')
    model.save(f'model{i}')
##

!tar cvfz main.py.tar.gz main.py model1 model2 model3 model4 model5
