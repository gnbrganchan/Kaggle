!pip install ../input/pytorchtabnetpretraining/pytorch_tabnet-2.0.1-py3-none-any.whl

!cp ../input/pytorchtabnetpretraining/pytorch_tabnet-2.0.1-py3-none-any.whl pytorch_tabnet-2.0.1-py3-none-any.whl
!cp -r ../input/pytorchtabnetpretraining/pytorch_tabnet-2.0.1 pytorch_tabnet-2.0.1

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
DATA_FILE＿DT100 = '/kaggle/input/generate-training-data-dt/training_data_dt.parquet'
DATA_FILE＿DT200 = '/kaggle/input/generate-training-data-dt200/training_data_dt.parquet'
DATA_FILE＿DT_LGB_OTHER = '/kaggle/input/generate-training-data-dt-lgbm-other/training_data_dt_lgb_other.parquet'

TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = 'payout'

rand = 1024

data = pd.read_parquet(DATA_FILE)
X = data[TRAIN_FEATS].sample(n=1500000,random_state=0).reset_index().drop("index",axis = 1)
y = data[TARGET_COL].sample(n=1500000,random_state=0).reset_index().drop("index",axis = 1)
##

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor

SEED = 46

def pretrain_model(X,y):
    tabnet_params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                         n_independent=2, n_shared=2,
                         seed=SEED, lambda_sparse=1e-3, 
                         optimizer_fn=torch.optim.Adam, 
                         optimizer_params=dict(lr=2e-2),
                         mask_type="entmax",
                         scheduler_params=dict(mode="min",
                                               patience=5,
                                               min_lr=1e-5,
                                               factor=0.9,),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=10
                        )
    
    pretrainer = TabNetPretrainer(**tabnet_params)

    pretrainer.fit(
        X_train=X.values,
        eval_set=[X.values],
        max_epochs=200,
        patience=20, batch_size=30000, virtual_batch_size=3000,
        num_workers=1,drop_last=True)
    
    return pretrainer
    

def make_model(X_train,y_train,X_valid,y_valid,pretrainer):
    """Builds a decision tree model based on stored trainingd data"""
    
    tabnet_params = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                         n_independent=2, n_shared=2,
                         seed=SEED, lambda_sparse=1e-3,
                         optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=2e-2,
                                               weight_decay=1e-5
                                              ),
                         mask_type="entmax",
                         scheduler_params=dict(max_lr=0.05,
                                               steps_per_epoch=int(X_train.shape[0] / 256),
                                               epochs=200,
                                               is_batch_level=True
                                              ),
                         scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                         verbose=10,
#                          cat_idxs=cat_idxs, # comment out when Unsupervised
#                          cat_dims=cat_dims, # comment out when Unsupervised
#                          cat_emb_dim=1 # comment out when Unsupervised
                        )

    model = TabNetRegressor(**tabnet_params)

    model.fit(X_train=X_train.values,
              y_train=y_train.values,
              eval_set=[(X_valid.values, y_valid.values)],
              eval_name = ["valid"],
              eval_metric = ["rmse"],
              max_epochs=200,
              patience=20, batch_size=4096, virtual_batch_size=256,
              num_workers=0,drop_last=False,
              from_unsupervised=pretrainer # comment out when Unsupervised
             )
    return model
##

rmse = []
i = 0

nsplits = 5
folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)

pretrainer = pretrain_model(X,y)

for tr_idx,va_idx in folds.split(X,y):
    X_tr = X.iloc[tr_idx].reset_index().drop("index",axis = 1)
    y_tr = y.iloc[tr_idx].reset_index().drop("index",axis = 1)
    X_va = X.iloc[va_idx].reset_index().drop("index",axis = 1)
    y_va = y.iloc[va_idx].reset_index().drop("index",axis = 1)
    
    model = make_model(X_tr,y_tr,X_va,y_va,pretrainer)
    y_pred = model.predict(X_va.values)

    rmse.append(np.sqrt(mean_squared_error(y_va, y_pred)))
    i += 1
    
    #save model
#     pickle.dump(model, open(f'model{i}.sav', 'wb'))
#     torch.save(model,f'model{i}.sav')
    model.save_model(f'model{i}')
    
!tar cvfz main.py.tar.gz main.py model1.zip model2.zip model3.zip model4.zip model5.zip pytorch_tabnet-2.0.1-py3-none-any.whl pytorch_tabnet-2.0.1
