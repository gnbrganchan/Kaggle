from google.colab import drive
drive.mount('/content/drive',)

!pip install optuna
import optuna

path_data = "/content/drive/My Drive/input/siim-isic/"

import gc
import os
import random
import csv
import sys
import json
import datetime
import time
from contextlib import redirect_stdout

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from numba import jit
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from tqdm import tqdm
import logging

gc.collect()


X_origin = X_train.copy()
def objective(trial):
  # # PCA
  # n_pca = trial.suggest_int('n_pca', 1, 10)
  # X_train = X_origin.copy()
  # X_train_pca = X_train.drop(["sex", "age_approx", "anatom_site_general_challenge"], axis = 1).copy()
  # pca = PCA(n_components = n_pca)
  # pca.fit(X_train_pca)
  # X_train_pca = pca.transform(X_train_pca)
  # X_train_pca = pd.DataFrame(X_train_pca, columns=["PC{}".format(x + 1) for x in range(n_pca)])
  # X_train = X_train_pca.join(X_train.loc[:,["sex", "age_approx", "anatom_site_general_challenge"]])

  X_tr_s = pd.DataFrame()
  y_tr_s = pd.Series(dtype='float64')

  nsplits = 5
  # folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)
  folds = StratifiedKFold(n_splits=nsplits, shuffle = True, random_state = rand)

  # LGB
  params = {
    "objective": "binary",
    "boosting": trial.suggest_categorical('boosting',["gbdt","dart"]),
    "max_depth": trial.suggest_int('max_depth', 2, 20),
    "num_leaves": trial.suggest_int('num_leaves', 6, 200),
    "subsample": trial.suggest_uniform('subsample', 0.5, 1.0),
    "subsample_freq": 1,
    "bagging_seed": rand,
    "learning_rate": trial.suggest_uniform('learning_rate', 0.01, 0.2),
    "feature_fraction": trial.suggest_uniform('feature_fraction', 0.5, 1.0),
    "feature_seed": rand,
    "min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 2, 1000),
    "lambda_l1": trial.suggest_int('lambda_l1', 0, 200),
    "lambda_l2": trial.suggest_int('lambda_l2', 0, 200),
    "random_state": rand,
    "metric": "auc",
    "verbose" : -1
    }

  # # XGB
  # xgb_params = {
  #   "objective": "binary:logistic",
  #   "eta" : trial.suggest_categorical('eta',[0.05,0.1]),
  #   "max_depth" : trial.suggest_int('max_depth', 3, 9),
  #   "min_child_weight" : trial.suggest_loguniform('min_child_weight', 0.1, 10.0),
  #   "gamma" : trial.suggest_loguniform('gamma', 1e-8, 1.0),
  #   "colsample_bytree" : trial.suggest_discrete_uniform('colsample', 0.6, 0.95, 0.05),
  #   "subsample" : trial.suggest_discrete_uniform('subsample', 0.6, 0.95, 0.05),
  #   "alpha" : trial.suggest_loguniform('alpha', 1e-8, 1.0),
  #   "lambda" : trial.suggest_loguniform('lambda', 1e-6, 10.0),
  #   "random_state" : rand,
  #   "eval_metric": "auc",
  # }
  # esr = trial.suggest_categorical('esr',[100, 200])


  # # RF
  # rf_params = {
  #   "criterion" : trial.suggest_categorical('criterion',["gini","entropy"]),
  #   "max_depth" : trial.suggest_int('max_depth', 2, 20),
  #   "min_samples_split" : trial.suggest_int('min_samples_split', 2, 50),
  #   "max_leaf_nodes" : trial.suggest_int('max_leaf_nodes', 2, 500),
  #   "n_estimators" : trial.suggest_int('n_estimators', 1, 20) * 100
  # }

  for tr_idx,va_idx in folds.split(X_train,y_train):
      X_tr = X_train.iloc[tr_idx]
      y_tr = y_train.iloc[tr_idx]
      X_va = X_train.iloc[va_idx]
      y_va = y_train.iloc[va_idx]

      X_tr_s_1 = pd.DataFrame()
      y_tr_s = pd.concat([y_tr_s, y_va], axis = 0)
      
      X_tr_1, X_tr_2, y_tr_1, y_tr_2 = train_test_split(X_tr, y_tr, random_state = rand, stratify = y_tr)

      # LGB
      with redirect_stdout(open(os.devnull, 'w')):
        d_half_1 = lgb.Dataset(X_tr_1, label=y_tr_1, categorical_feature=categorical_features,  free_raw_data=False, silent = True)
        d_half_2 = lgb.Dataset(X_tr_2, label=y_tr_2, categorical_feature=categorical_features,  free_raw_data=False, silent = True)
        if params["boosting"] == "dart":
          model = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=[d_half_1,d_half_2], verbose_eval = False, categorical_feature=categorical_features)
        else:
          model = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=[d_half_1,d_half_2], verbose_eval = False,  early_stopping_rounds=100, categorical_feature=categorical_features)
        tr_pred = model.predict(X_va, num_iteration=model.best_iteration)

      # # XGB
      # with redirect_stdout(open(os.devnull, 'w')):
      #   d_1 = xgb.DMatrix(X_tr_1, label = y_tr_1)
      #   d_2 = xgb.DMatrix(X_tr_2, label = y_tr_2)
      #   d_val = xgb.DMatrix(X_va)
      #   wlist = [(d_1, 'train'), (d_2, 'eval')]
      #   model = xgb.train(xgb_params, dtrain = d_1, num_boost_round = 1000, evals = wlist, verbose_eval = 100, early_stopping_rounds = esr)
      #   tr_pred = model.predict(d_val, ntree_limit=model.best_ntree_limit)

      # # Random Forest
      # with redirect_stdout(open(os.devnull, 'w')):
      #   model = RandomForestClassifier(
      #       max_depth = rf_params["max_depth"],
      #       criterion = rf_params["criterion"],
      #       min_samples_split = rf_params["min_samples_split"],
      #       max_leaf_nodes = rf_params["max_leaf_nodes"],
      #       n_estimators = rf_params["n_estimators"], class_weight='balanced', random_state=rand)
      #   model.fit(X_tr, y_tr)
      #   tr_pred = model.predict_proba(X_va)[:,1] 


      X_tr_s_1["LGB"] = tr_pred
      X_tr_s = pd.concat([X_tr_s, X_tr_s_1], axis = 0)
      
      del model
      gc.collect()
      
  #     model = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=[d_half_2,d_half_1], verbose_eval=100, early_stopping_rounds=100)
  #     tr_pred = model.predict(X_va, num_iteration=model.best_iteration)
  #     val_score.append(roc_auc_score(y_va, tr_pred))
  #     pred += model.predict(X_test.drop(["image_name"],axis=1), num_iteration=model.best_iteration) / (nsplits*2)

  auc_score = roc_auc_score(y_tr_s, X_tr_s["LGB"])

  return -auc_score
  
  
  study = optuna.create_study(study_name="SIIM",
                            storage='sqlite:////content//drive//My Drive//SIIM_lgb.db',
                            load_if_exists=True)
study.optimize(objective, n_trials=2000,n_jobs=-1)

print(study.best_params)


study.trials_dataframe().to_csv("/content/drive/My Drive/study_history.csv")


#最適化したハイパーパラメータの確認
print('check!!!')
print('best_param:{}'.format(study.best_params))
print('====================')
 
#最適化後の目的関数値
print('best_value:{}'.format(study.best_value))
print('====================')
 
#最適な試行
print('best_trial:{}'.format(study.best_trial))
print('====================')
 
# # トライアルごとの結果を確認
# for i in study.trials:
#     print('param:{0}, eval_value:{1}'.format(i[5], i[2]))
# print('====================')


study.best_params
study.best_value
