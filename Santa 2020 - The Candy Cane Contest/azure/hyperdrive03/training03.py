import argparse
import os
import sys
from azureml.core import Run
import pandas as pd
import subprocess

# Set regularization parameter
parser = argparse.ArgumentParser()
parser.add_argument('--max_depth', type=float, dest='max_depth')
parser.add_argument('--num_leaves', type=float, dest='num_leaves')
parser.add_argument('--subsample', type=float, dest='subsample')
parser.add_argument('--learning_rate', type=float, dest='learning_rate')
parser.add_argument('--min_data_in_leaf', type=float, dest='min_data_in_leaf')
parser.add_argument('--lambda_l1', type=float, dest='lambda_l1')
parser.add_argument('--lambda_l2', type=float, dest='lambda_l2')
parser.add_argument('--n_estimators', type=float, dest='n_estimators')


args = parser.parse_args()
max_depth = int(args.max_depth)
num_leaves = int(args.num_leaves)
subsample = args.subsample
learning_rate = args.learning_rate
min_data_in_leaf = int(args.min_data_in_leaf)
lambda_l1 = args.lambda_l1
lambda_l2 = args.lambda_l2
n_estimators = int(args.n_estimators)

# Get the experiment run context
run = Run.get_context()
run.log('version',sys.version)

from contextlib import redirect_stdout

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
DATA_FILE_LB = 'training_data_mine.parquet'

TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = 'payout'

rand = 1024

 # LGB
lgb_params = {
# "objective": "binary",
# "boosting": trial.suggest_categorical('boosting',["gbdt","dart"]),
"max_depth": max_depth,
"num_leaves": num_leaves,
"subsample": subsample,
# "subsample_freq": 1,
# "bagging_seed": rand,
"learning_rate": learning_rate,
# "feature_fraction": trial.suggest_uniform('feature_fraction', 0.5, 1.0),
# "feature_seed": rand,
"min_data_in_leaf": min_data_in_leaf,
"lambda_l1": lambda_l1,
"lambda_l2": lambda_l2,
# "random_state": rand,
"metric": "rmse"
}


def make_model(X,Y):
    """Builds a decision tree model based on stored trainingd data"""
    X_1, X_2, y_1, y_2 = train_test_split(X, Y, test_size = 0.25, random_state = rand)

    with redirect_stdout(open(os.devnull, 'w')):
        model = lgb.LGBMRegressor(
            learning_rate = lgb_params["learning_rate"],
            max_depth = lgb_params["max_depth"],
            min_child_samples = lgb_params["min_data_in_leaf"],
            n_estimators = n_estimators,
            num_leaves = lgb_params["num_leaves"],
            reg_alpha = lgb_params["lambda_l1"],
            reg_lambda = lgb_params["lambda_l2"],
            subsample = lgb_params["subsample"]
        )

        model.fit(X_1,y_1,
                eval_set=[(X_1,y_1),(X_2,y_2)],
                eval_metric='rmse',
                early_stopping_rounds=50)
    return model

# prepare data
data = pd.read_parquet(DATA_FILE_LB)
data = data.query("n_pulls_self > 0 or n_pulls_opp > 1")

nsplits = 5
folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)
X_train = data[TRAIN_FEATS]
y_train = data[TARGET_COL]
y_train = np.log1p(y_train)

# CV training
rmse = []
mae = []
i = 0
for tr_idx,va_idx in folds.split(X_train,y_train):
    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_va = y_train.iloc[va_idx]
    
    model = make_model(X_tr,y_tr)
    y_pred = model.predict(X_va,num_iteration = model.best_iteration_)
    rmse.append(np.sqrt(mean_squared_error(y_va, y_pred)))
    mae.append(mean_absolute_error(y_va,y_pred))
    i += 1

run.log("rmse",np.mean(rmse))
run.log("mae",np.mean(mae))

run.complete()