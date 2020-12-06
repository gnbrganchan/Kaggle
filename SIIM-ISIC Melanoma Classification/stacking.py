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

path_data = "/content/drive/My Drive/input/siim-isic/"

train = pd.read_csv(path_data + "train.csv")
test = pd.read_csv(path_data + "test.csv")
sub = pd.read_csv(path_data + "sample_submission.csv")

train = pd.read_csv(path_data + "train.csv")
test = pd.read_csv(path_data + "test.csv")

# B7-512
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-512/pred_tr.csv").rename(columns = {"target" : "Price_B7_512"})
train = train.merge(pred_tr, how = "left")
# B7-384
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384/pred_tr.csv").rename(columns = {"target" : "Price_B7_384"})
train = train.merge(pred_tr, how = "left")
# B7-384-2018
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384-2018/pred_tr.csv").rename(columns = {"target" : "Price_B7_384_2018"})
train = train.merge(pred_tr, how = "left")
# B7-384-2019
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384-2019/pred_tr.csv").rename(columns = {"target" : "Price_B7_384_2019"})
train = train.merge(pred_tr, how = "left")
# B7-c5-384-2018-2019
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384-2018-2019/pred_tr.csv").rename(columns = {"target" : "Price_B7_cv_384_2018-2019"})
train = train.merge(pred_tr, how = "left")
# B6-cv5-384-2018
pred_tr = pd.read_csv(path_data + "effnet-b6-cv5-384-2018/pred_tr.csv").rename(columns = {"target" : "Price_B6_cv5_384_2018"})
train = train.merge(pred_tr, how = "left")
# B6-cv-384-2018-2019
pred_tr = pd.read_csv(path_data + "effnet-b6-cv-384-2018-2019/pred_tr.csv").rename(columns = {"target" : "Price_B6_cv_384_2018-2019"})
train = train.merge(pred_tr, how = "left")
# B6-cv5-384-2019
pred_tr = pd.read_csv(path_data + "effnet-b6-cv5-384-2019/pred_tr.csv").rename(columns = {"target" : "Price_B6_cv5_384_2019"})
train = train.merge(pred_tr, how = "left")
# B6-512
pred_tr = pd.read_csv(path_data + "effnet-b6-cv-512/pred_tr.csv").rename(columns = {"target" : "Price_B6_512"})
train = train.merge(pred_tr, how = "left")
# B5-384
pred_tr = pd.read_csv(path_data + "effnet-b5-cv-384/pred_tr.csv").rename(columns = {"target" : "Price_B5_384"})
train = train.merge(pred_tr, how = "left")
# B5
pred_tr = pd.read_csv(path_data + "effnet-b5-cv/pred_tr.csv").rename(columns = {"target" : "Price_B5"})
train = train.merge(pred_tr, how = "left")
# # B4-384
# pred_tr = pd.read_csv(path_data + "effnet-b4-cv-384/pred_tr.csv").rename(columns = {"target" : "Price_B4_384"})
# train = train.merge(pred_tr, how = "left")
# # B3
# pred_tr = pd.read_csv(path_data + "effnet-b3-cv/pred_tr.csv").rename(columns = {"target" : "Price_B3"})
# train = train.merge(pred_tr, how = "left")
# # B1
# pred_tr = pd.read_csv(path_data + "effnet-b1-cv/pred_tr.csv").rename(columns = {"target" : "Price_B1"})
# train = train.merge(pred_tr, how = "left")
# # DenseNet201
# pred_tr = pd.read_csv(path_data + "densenet201-cv/pred_tr.csv").rename(columns = {"target" : "Price_DN201"})
# train = train.merge(pred_tr, how = "left")
# # InceptionResNetV2
# pred_tr = pd.read_csv(path_data + "inceptionresnetv2-cv/pred_tr.csv").rename(columns = {"target" : "Price_IRN"})
# train = train.merge(pred_tr, how = "left")
# # Xception-384
# pred_tr = pd.read_csv(path_data + "xception-cv5-384/pred_tr.csv").rename(columns = {"target" : "Price_X_384"})
# train = train.merge(pred_tr, how = "left")
# # Xception
# pred_tr = pd.read_csv(path_data + "xception-cv/pred_tr.csv").rename(columns = {"target" : "Price_X"})
# train = train.merge(pred_tr, how = "left")
# # ResNet152V2
# pred_tr = pd.read_csv(path_data + "resnet152v2-cv/pred_tr.csv").rename(columns = {"target" : "Price_RN152"})
# train = train.merge(pred_tr, how = "left")
# # ResNet50V2
# pred_tr = pd.read_csv(path_data + "resnet50v2-cv/pred_tr.csv").rename(columns = {"target" : "Price_RN50"})
# train = train.merge(pred_tr, how = "left")

# B7-512
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-512/submission.csv").rename(columns = {"target" : "Price_B7_512"})
test = test.merge(pred_tr, how = "left")
# B7-384
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384/submission.csv").rename(columns = {"target" : "Price_B7_384"})
test = test.merge(pred_tr, how = "left")
# B7-384-2018
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384-2018/submission.csv").rename(columns = {"target" : "Price_B7_384_2018"})
test = test.merge(pred_tr, how = "left")
# B7-384-2019
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384-2019/submission.csv").rename(columns = {"target" : "Price_B7_384_2019"})
test = test.merge(pred_tr, how = "left")
# B7-c5-384-2018-2019
pred_tr = pd.read_csv(path_data + "effnet-b7-cv-384-2018-2019/submission.csv").rename(columns = {"target" : "Price_B7_cv_384_2018-2019"})
test = test.merge(pred_tr, how = "left")
# B6-cv5-384-2018
pred_tr = pd.read_csv(path_data + "effnet-b6-cv5-384-2018/submission.csv").rename(columns = {"target" : "Price_B6_cv5_384_2018"})
test = test.merge(pred_tr, how = "left")
# B6-cv5-384-2018-2019
pred_tr = pd.read_csv(path_data + "effnet-b6-cv-384-2018-2019/submission.csv").rename(columns = {"target" : "Price_B6_cv_384_2018-2019"})
test = test.merge(pred_tr, how = "left")
# B6-cv5-384-2019
pred_tr = pd.read_csv(path_data + "effnet-b6-cv5-384-2019/submission.csv").rename(columns = {"target" : "Price_B6_cv5_384_2019"})
test = test.merge(pred_tr, how = "left")
# B6-512
pred_tr = pd.read_csv(path_data + "effnet-b6-cv-512/submission.csv").rename(columns = {"target" : "Price_B6_512"})
test = test.merge(pred_tr, how = "left")
# B5-384
pred_tr = pd.read_csv(path_data + "effnet-b5-cv-384/submission.csv").rename(columns = {"target" : "Price_B5_384"})
test = test.merge(pred_tr, how = "left")
# B5
pred_tr = pd.read_csv(path_data + "effnet-b5-cv/submission.csv").rename(columns = {"target" : "Price_B5"})
test = test.merge(pred_tr, how = "left")
# # B4-384
# pred_tr = pd.read_csv(path_data + "effnet-b4-cv-384/submission.csv").rename(columns = {"target" : "Price_B4_384"})
# test = test.merge(pred_tr, how = "left")
# # B3
# pred_tr = pd.read_csv(path_data + "effnet-b3-cv/submission.csv").rename(columns = {"target" : "Price_B3"})
# test = test.merge(pred_tr, how = "left")
# # B1
# pred_tr = pd.read_csv(path_data + "effnet-b1-cv/submission.csv").rename(columns = {"target" : "Price_B1"})
# test = test.merge(pred_tr, how = "left")
# # DenseNet201
# pred_tr = pd.read_csv(path_data + "densenet201-cv/submission.csv").rename(columns = {"target" : "Price_DN201"})
# test = test.merge(pred_tr, how = "left")
# # InceptionResNetV2
# pred_tr = pd.read_csv(path_data + "inceptionresnetv2-cv/submission.csv").rename(columns = {"target" : "Price_IRN"})
# test = test.merge(pred_tr, how = "left")
# # Xception-384
# pred_tr = pd.read_csv(path_data + "xception-cv5-384/submission.csv").rename(columns = {"target" : "Price_X_384"})
# test = test.merge(pred_tr, how = "left")
# # Xception
# pred_tr = pd.read_csv(path_data + "xception-cv/submission.csv").rename(columns = {"target" : "Price_X"})
# test = test.merge(pred_tr, how = "left")
# # ResNet152V2
# pred_tr = pd.read_csv(path_data + "resnet152v2-cv/submission.csv").rename(columns = {"target" : "Price_RN152"})
# test = test.merge(pred_tr, how = "left")
# # ResNet50V2
# pred_tr = pd.read_csv(path_data + "resnet50v2-cv/submission.csv").rename(columns = {"target" : "Price_RN50"})
# test = test.merge(pred_tr, how = "left")

train = train[~train.sex.isna()]
train.fillna({"anatom_site_general_challenge" : "NA"}, inplace = True)
test.fillna({"anatom_site_general_challenge" : "NA"}, inplace = True)

le = LabelEncoder()
le.fit_transform(pd.concat((train.sex,test.sex),axis = 0).unique())
train.sex = le.transform(train.sex)
test.sex = le.transform(test.sex)
le = LabelEncoder()
le.fit_transform(pd.concat((train.anatom_site_general_challenge,test.anatom_site_general_challenge),axis = 0).unique())
train.anatom_site_general_challenge = le.transform(train.anatom_site_general_challenge)
test.anatom_site_general_challenge = le.transform(test.anatom_site_general_challenge)

train = train[~train.isna().any(axis=1)]
y_train = train["target"]
X_train = train.drop(["image_name", "patient_id", "diagnosis", "benign_malignant", "target"], axis = 1)
X_test = test.drop(["patient_id"], axis = 1)

categorical_features = ["sex", "anatom_site_general_challenge"]
rand = 1024
X_tr_s = pd.DataFrame()
y_tr_s = pd.Series()

nsplits = 5
# folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)
folds = StratifiedKFold(n_splits=nsplits, shuffle = True, random_state = rand)

lgb_params = {
    "objective": "binary",
    "boosting": "dart",
    "max_depth": 2,
    "num_leaves": 108,
    "subsample": 0.7444639757342139,
    "subsample_freq": 1,
    "bagging_seed": rand,
    "learning_rate": 0.040529143447705585,
    "feature_fraction": 0.9425000816411454,
    "feature_seed": rand,
    "min_data_in_leaf": 105,
    "lambda_l1": 0,
    "lambda_l2": 67,
    "random_state": rand,
    "metric": "auc"
}
xgb_params = {
    "objective": "binary:logistic",
    "eta" : 0.05,
    "max_depth" : 3,
    "min_child_weight" : 0.1468470923616862,
    "gamma" : 1.988631240268601e-07,
    "colsample_bytree" : 0.85,
    "subsample" : 0.65,
    "alpha" : 0.003079217000310264,
    "lambda" : 0.03112985702049758,
    "random_state" : rand,
    "eval_metric": "auc",
}
xgb_params_s = {
    "objective": "binary:logistic",
    "eta" : 0.1,
    "max_depth" : 2,
    "min_child_weight" : 6.763060350980329,
    "gamma" : 0.26470815797119057,
    "colsample_bytree" : 0.65,
    "subsample" : 1.0,
    "alpha" : 1.0131473490847693e-08,
    "lambda" : 0.5390634169142206,
    "random_state" : rand,
    "eval_metric": "auc",
}
rf_params = {
    "criterion" : "entropy",
    "max_depth" : 4,
    "max_leaf_nodes" : 32,
    "min_samples_split" : 42,
    "n_estimators" : 1500
}
rf_params_s = {
    "criterion" : "entropy",
    "max_depth" : 3,
    "max_leaf_nodes" : 410,
    "min_samples_split" : 20,
    "n_estimators" : 200
}

X_tr_s = pd.DataFrame()
y_tr_s = pd.Series()
X_te_s = pd.DataFrame()

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_mean = np.zeros(len(X_test))
pred_rmean = np.zeros(len(X_test))
pred_rmin = np.zeros(len(X_test))
pred_rmax = np.zeros(len(X_test))
pred_lr = np.zeros(len(X_test))
pred_rf = np.zeros(len(X_test))
pred_kn1 = np.zeros(len(X_test))
pred_kn5 = np.zeros(len(X_test))
pred_dt = np.zeros(len(X_test))

df_fimp = pd.DataFrame()
for tr_idx,va_idx in folds.split(X_train,y_train):
    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_va = y_train.iloc[va_idx]
    
    X_tr_s_1 = pd.DataFrame()
    y_tr_s = pd.concat([y_tr_s, y_va], axis = 0)
    
    # X_tr_1, X_tr_2, y_tr_1, y_tr_2 = train_test_split(X_tr, y_tr, test_size = 0.25, random_state = rand)
    X_tr_1, X_tr_2, y_tr_1, y_tr_2 = train_test_split(X_tr, y_tr, random_state = rand, stratify = y_tr)

    # LGB
    d_1 = lgb.Dataset(X_tr_1, label=y_tr_1, categorical_feature=categorical_features, free_raw_data=False)
    d_2 = lgb.Dataset(X_tr_2, label=y_tr_2, categorical_feature=categorical_features, free_raw_data=False)
    
    model = lgb.train(lgb_params, train_set=d_1, num_boost_round=1000, valid_sets=[d_1,d_2], verbose_eval=100, early_stopping_rounds=100)
    tr_pred = model.predict(X_va, num_iteration=model.best_iteration)
    pred_lgb += model.predict(X_test.drop(["image_name"],axis=1), num_iteration=model.best_iteration) / (nsplits)
    X_tr_s_1["LGB"] = tr_pred
    
    df_fimp_1 = pd.DataFrame()
    df_fimp_1["feature"] = X_train.columns.values
    df_fimp_1["importance"] = model.feature_importance()
    
    df_fimp = pd.concat([df_fimp, df_fimp_1], axis=0)
    
    # XGB
    d_1 = xgb.DMatrix(X_tr_1, label = y_tr_1)
    d_2 = xgb.DMatrix(X_tr_2, label = y_tr_2)
    d_val = xgb.DMatrix(X_va)
    d_test = xgb.DMatrix(X_test.drop(["image_name"],axis=1))

    wlist = [(d_1, 'train'), (d_2, 'eval')]

    model = xgb.train(xgb_params, dtrain = d_1, num_boost_round = 2000, evals = wlist, verbose_eval = 100, early_stopping_rounds = 100)
    tr_pred = model.predict(d_val, ntree_limit=model.best_ntree_limit)
    pred_xgb += model.predict(d_test, ntree_limit=model.best_ntree_limit) / (nsplits) 
    X_tr_s_1["XGB"] = tr_pred
    
    # Random Forest
    model = RandomForestClassifier(criterion = rf_params["criterion"],
                                   max_leaf_nodes = rf_params["max_leaf_nodes"],
                                   min_samples_split = rf_params["min_samples_split"],
                                   max_depth=rf_params["max_depth"],
                                   n_estimators=rf_params["n_estimators"], class_weight='balanced', random_state=rand)
    model.fit(X_tr, y_tr)
    tr_pred = model.predict_proba(X_va)[:,1]
    pred_rf += model.predict_proba(X_test.drop(["image_name"], axis = 1))[:,1] / (nsplits)  
    X_tr_s_1["RF"] = tr_pred
    
#     # Decision Tree
#     model = DecisionTreeClassifier(max_depth = 7, class_weight='balanced')
#     model.fit(X_tr, y_tr)
#     tr_pred = model.predict_proba(X_va)[:,1]
#     pred_dt += model.predict_proba(X_test.drop(["image_name"], axis = 1))[:,1] / (nsplits)
#     X_tr_s_1["DT"] = tr_pred
        
    # Logistics Regression
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_tr, y_tr)
    tr_pred = model.predict_proba(X_va)[:,1]
    pred_lr += model.predict_proba(X_test.drop(["image_name"], axis = 1))[:,1] / (nsplits)
    X_tr_s_1["LR"] = tr_pred
    
#     # K neighbor1
#     model = CondensedNearestNeighbour(random_state = rand)
#     X_tr_1, y_tr_1 = model.fit_resample(X_tr, y_tr)
#     model = KNeighborsClassifier(n_neighbors = 1)
#     model.fit(X_tr_1, y_tr_1)
#     tr_pred = model.predict_proba(X_va)[:,1]
#     pred_kn1 += model.predict(X_test.drop(["image_name"], axis = 1)) / (nsplits)
#     X_tr_s_1["KN1"] = tr_pred
    
#     # K neighbor5
#     model = KNeighborsClassifier(n_neighbors = 5)
#     model.fit(X_tr_1, y_tr_1)
#     tr_pred = model.predict_proba(X_va)[:,1]
#     pred_kn5 += model.predict(X_test.drop(["image_name"], axis = 1)) / (nsplits)
#     X_tr_s_1["KN5"] = tr_pred
    
    # Mean
    tr_pred = X_va.drop(["sex", "age_approx", "anatom_site_general_challenge"], axis = 1).mean(axis = 1).reset_index().loc[:,0]
    pred_mean += X_test.drop(["image_name", "sex", "age_approx", "anatom_site_general_challenge"], axis = 1).mean(axis = 1).reset_index().loc[:,0] / (nsplits)
    X_tr_s_1["Mean"] = tr_pred
    
    # RankMean
    tr_pred = (X_va.drop(["sex", "age_approx", "anatom_site_general_challenge"], axis = 1).rank()/len(X_va)).mean(axis = 1).reset_index().loc[:,0]
    pred_rmean += (X_test.drop(["image_name", "sex", "age_approx", "anatom_site_general_challenge"], axis = 1).rank()/len(X_test)).mean(axis = 1).reset_index().loc[:,0] / (nsplits)
    X_tr_s_1["RMean"] = tr_pred

    # RankMin
    tr_pred = (X_va.drop(["sex", "age_approx", "anatom_site_general_challenge"], axis = 1).rank()/len(X_va)).min(axis = 1).reset_index().loc[:,0]
    pred_rmin += (X_test.drop(["image_name", "sex", "age_approx", "anatom_site_general_challenge"], axis = 1).rank()/len(X_test)).min(axis = 1).reset_index().loc[:,0] / (nsplits)
    X_tr_s_1["RMin"] = tr_pred

    # RankMax
    tr_pred = (X_va.drop(["sex", "age_approx", "anatom_site_general_challenge"], axis = 1).rank()/len(X_va)).max(axis = 1).reset_index().loc[:,0]
    pred_rmax += (X_test.drop(["image_name", "sex", "age_approx", "anatom_site_general_challenge"], axis = 1).rank()/len(X_test)).max(axis = 1).reset_index().loc[:,0] / (nsplits)
    X_tr_s_1["RMax"] = tr_pred
    
    del model
    gc.collect()
    
    X_tr_s = pd.concat([X_tr_s, X_tr_s_1], axis = 0)
    
    X_tr_s = X_tr_s.reset_index().drop(["index"], axis = 1)
y_tr_s = y_tr_s.reset_index().drop(["index"], axis = 1)

X_te_s["image_name"] = X_test["image_name"]
X_te_s["LGB"] = pred_lgb
X_te_s["XGB"] = pred_xgb
X_te_s["RF"] = pred_rf
# X_te_s["DT"] = pred_dt
X_te_s["LR"] = pred_lr
# X_te_s["KN1"] = pred_kn1
# X_te_s["KN5"] = pred_kn5
X_te_s["Mean"] = pred_mean
X_te_s["RMean"] = pred_rmean
X_te_s["RMin"] = pred_rmin
X_te_s["RMax"] = pred_rmax

print(X_tr_s.apply(lambda x : roc_auc_score(y_tr_s,x)))

plt.figure(figsize=(14, 7))
sns.barplot(x="importance", y="feature", data=df_fimp.sort_values(by="importance", ascending=False))
plt.title("LightGBM Feature Importance")
plt.tight_layout()

# xgb for prediction
X_tr_ss = pd.DataFrame()
y_tr_ss = pd.Series()
pred_xgb_s = np.zeros(len(X_test))
pred_rf_s = np.zeros(len(X_test))
fimp = pd.DataFrame()
for tr_idx,va_idx in folds.split(X_tr_s,y_tr_s):
    X_tr = X_tr_s.iloc[tr_idx]
    y_tr = y_tr_s.iloc[tr_idx]
    X_va = X_tr_s.iloc[va_idx]
    y_va = y_tr_s.iloc[va_idx]
    
    X_tr_ss_1 = pd.DataFrame()
    y_tr_ss = pd.concat([y_tr_ss, y_va], axis = 0)
    
    X_tr_1, X_tr_2, y_tr_1, y_tr_2 = train_test_split(X_tr, y_tr, test_size = 0.25, random_state = rand)
    
    # XGB
    d_1 = xgb.DMatrix(X_tr_1, label = y_tr_1)
    d_2 = xgb.DMatrix(X_tr_2, label = y_tr_2)
    d_val = xgb.DMatrix(X_va)
    d_test = xgb.DMatrix(X_te_s.drop(["image_name"],axis=1))

    wlist = [(d_1, 'train'), (d_2, 'eval')]

    model = xgb.train(xgb_params_s, dtrain = d_1, num_boost_round = 2000, evals = wlist, verbose_eval = 100, early_stopping_rounds = 100)
    tr_pred = model.predict(d_val, ntree_limit=model.best_ntree_limit)
    pred_xgb_s += model.predict(d_test, ntree_limit=model.best_ntree_limit) / (nsplits)
    X_tr_ss_1["XGB"] = tr_pred
    
    fimp_1 = pd.DataFrame()
    fimp_1["feature"] = model.get_fscore().keys()
    fimp_1["importance"] = model.get_fscore().values()    
    fimp = pd.concat([fimp, fimp_1], axis=0)

    # Random Forest
    model = RandomForestClassifier(criterion = rf_params_s["criterion"],
                                   max_leaf_nodes = rf_params_s["max_leaf_nodes"],
                                   min_samples_split = rf_params_s["min_samples_split"],
                                   max_depth=rf_params_s["max_depth"],
                                   n_estimators=rf_params_s["n_estimators"], class_weight='balanced', random_state=rand)
    model.fit(X_tr, y_tr)
    tr_pred = model.predict_proba(X_va)[:,1]
    pred_rf_s += model.predict_proba(X_te_s.drop(["image_name"], axis = 1))[:,1] / (nsplits)  
    X_tr_ss_1["RF"] = tr_pred

    del model
    gc.collect();gc.collect();

    X_tr_ss = pd.concat([X_tr_ss, X_tr_ss_1], axis = 0)
    
  plt.figure(figsize=(14, 7))
sns.barplot(x="importance", y="feature", data=fimp.sort_values(by="importance", ascending=False))
plt.title("XGBM Feature Importance")
plt.tight_layout()

X_tr_ss["XGB+RF"] = (X_tr_ss["XGB"] + X_tr_ss["RF"])/2
print(X_tr_ss.apply(lambda x : roc_auc_score(y_tr_ss,x)))

X_test["target"] = pred_xgb
submit = sub.drop(["target"],axis=1).copy()
submit = submit.merge(X_test.loc[:,["image_name", "target"]] , on = "image_name", how = "left")
submit.to_csv("/content/drive/My Drive/submission_xgb.csv", index = False)
submit.head(3)
