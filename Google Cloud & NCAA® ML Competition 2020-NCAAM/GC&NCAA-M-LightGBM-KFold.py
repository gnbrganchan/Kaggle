import gc
import os
import random
import csv
import sys
import json
import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from numba import jit
#最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_rows', 300)
#最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_columns', 300)

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm

plt.style.use("seaborn")
sns.set(font_scale=1)

teams = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv")
tourney_compact = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
regular_compact = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")
second_compact = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSecondaryTourneyCompactResults.csv")
rank = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MMasseyOrdinals.csv")
sample_submission = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv")

teams_1ago = teams.copy()
teams_1ago.Season = teams_1ago.Season + 1
teams_1ago.Seed = teams_1ago.Seed.apply(lambda x: x[1:3]).astype(int)

t = teams.loc[:,["Season","TeamID"]]
t = t.merge(t,on="Season",how="outer")
t = t[t.TeamID_x < t.TeamID_y]
t = t.rename(columns={"TeamID_x":"TeamID_Low","TeamID_y":"TeamID_High"})

train_pred = tourney_compact.merge(teams,left_on=["Season","WTeamID"], right_on=["Season", "TeamID"],how="left")
train_pred = train_pred.merge(teams,left_on=["Season","LTeamID"], right_on=["Season", "TeamID"],how="left")

train_pred["TeamID_Low"] = train_pred.loc[:,["WTeamID","LTeamID"]].min(axis=1)
train_pred["TeamID_High"] = train_pred.loc[:,["WTeamID","LTeamID"]].max(axis=1)
train_pred.loc[train_pred.TeamID_Low == train_pred.WTeamID, "Seed_Low"] = train_pred[train_pred.TeamID_Low == train_pred.WTeamID].Seed_x.apply(lambda x: x[1:3]).astype(int)
train_pred.loc[train_pred.TeamID_Low == train_pred.LTeamID, "Seed_Low"] = train_pred[train_pred.TeamID_Low == train_pred.LTeamID].Seed_y.apply(lambda x: x[1:3]).astype(int)
train_pred.loc[train_pred.TeamID_High == train_pred.WTeamID, "Seed_High"] = train_pred[train_pred.TeamID_High == train_pred.WTeamID].Seed_x.apply(lambda x: x[1:3]).astype(int)
train_pred.loc[train_pred.TeamID_High == train_pred.LTeamID, "Seed_High"] = train_pred[train_pred.TeamID_High == train_pred.LTeamID].Seed_y.apply(lambda x: x[1:3]).astype(int)
train_pred.loc[train_pred.TeamID_High == train_pred.WTeamID, "target"] = 0
train_pred.loc[train_pred.TeamID_High == train_pred.LTeamID, "target"] = 1
train_pred["Seed_dif"] = train_pred.Seed_Low - train_pred.Seed_High
train_pred["Seed_div"] = train_pred.Seed_Low / train_pred.Seed_High

train_pred = train_pred.loc[:,["TeamID_Low","TeamID_High","Season","Seed_Low","Seed_High","Seed_dif","Seed_div","target"]]
train_pred = train_pred.merge(teams_1ago,left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
train_pred = train_pred.merge(teams_1ago,left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
train_pred.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
train_pred = train_pred.rename(columns={"Seed_x":"Seed_1ago_Low","Seed_y":"Seed_1ago_High"})
train_pred = train_pred[train_pred.Season >= 1986]
train_pred["Seed_prog_Low"] = train_pred.Seed_Low - train_pred.Seed_1ago_Low
train_pred["Seed_prog_High"] = train_pred.Seed_High - train_pred.Seed_1ago_High

test = t.copy().merge(teams,left_on=["Season","TeamID_Low"], right_on=["Season", "TeamID"],how="left")
test = test.merge(teams,left_on=["Season","TeamID_High"], right_on=["Season", "TeamID"],how="left")
test.loc[:,"Seed_x"] = test.Seed_x.apply(lambda x: x[1:3]).astype(int)
test.loc[:,"Seed_y"] = test.Seed_y.apply(lambda x: x[1:3]).astype(int)
test = test.rename(columns={"Seed_x":"Seed_Low","Seed_y":"Seed_High"})
test["Seed_dif"] = test.Seed_Low - test.Seed_High
test["Seed_div"] = test.Seed_Low / test.Seed_High
test= test.loc[:,["TeamID_Low","TeamID_High","Season","Seed_Low","Seed_dif","Seed_div","Seed_High"]]
test = test.merge(teams_1ago,left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
test = test.merge(teams_1ago,left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
test.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
test = test.rename(columns={"Seed_x":"Seed_1ago_Low","Seed_y":"Seed_1ago_High"})
test["Seed_prog_Low"] = test.Seed_Low - test.Seed_1ago_Low
test["Seed_prog_High"] = test.Seed_High - test.Seed_1ago_High

train_pred = train_pred.merge(regular_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_Low"],right_on=["Season","WTeamID"],how="left")
train_pred = train_pred.merge(regular_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_High"],right_on=["Season","WTeamID"],how="left")
train_pred = train_pred.rename(columns={"DayNum_x":"Reg_Win_Low","DayNum_y":"Reg_Win_High"})
train_pred = train_pred.merge(regular_compact.groupby(["Season","LTeamID"]).count().reset_index().loc[:,["Season","LTeamID","DayNum"]],left_on=["Season","TeamID_Low"],right_on=["Season","LTeamID"],how="left")
train_pred = train_pred.merge(regular_compact.groupby(["Season","LTeamID"]).count().reset_index().loc[:,["Season","LTeamID","DayNum"]],left_on=["Season","TeamID_High"],right_on=["Season","LTeamID"],how="left")
train_pred = train_pred.rename(columns={"DayNum_x":"Reg_Lose_Low","DayNum_y":"Reg_Lose_High"})
train_pred.drop(["WTeamID_x","WTeamID_y","LTeamID_x","LTeamID_y"],axis=1,inplace=True)
train_pred["Reg_WinRate_Low"] = train_pred.Reg_Win_Low/(train_pred.Reg_Win_Low+train_pred.Reg_Lose_Low)
train_pred["Reg_WinRate_High"] = train_pred.Reg_Win_High/(train_pred.Reg_Win_High+train_pred.Reg_Lose_High)
train_pred["Reg_WinRate_div"] = train_pred.Reg_WinRate_Low / train_pred.Reg_WinRate_High

train_pred = train_pred.merge(regular_compact.groupby(["Season","WTeamID"]).sum().reset_index().loc[:,["Season","WTeamID","WScore"]],left_on=["Season","TeamID_Low"],right_on=["Season","WTeamID"],how="left")
train_pred = train_pred.merge(regular_compact.groupby(["Season","WTeamID"]).sum().reset_index().loc[:,["Season","WTeamID","WScore"]],left_on=["Season","TeamID_High"],right_on=["Season","WTeamID"],how="left")
train_pred = train_pred.rename(columns={"WScore_x":"Reg_WScore_Low","WScore_y":"Reg_WScore_High"})
train_pred = train_pred.merge(regular_compact.groupby(["Season","LTeamID"]).sum().reset_index().loc[:,["Season","LTeamID","LScore"]],left_on=["Season","TeamID_Low"],right_on=["Season","LTeamID"],how="left")
train_pred = train_pred.merge(regular_compact.groupby(["Season","LTeamID"]).sum().reset_index().loc[:,["Season","LTeamID","LScore"]],left_on=["Season","TeamID_High"],right_on=["Season","LTeamID"],how="left")
train_pred = train_pred.rename(columns={"LScore_x":"Reg_LScore_Low","LScore_y":"Reg_LScore_High"})
train_pred.drop(["WTeamID_x","WTeamID_y","LTeamID_x","LTeamID_y"],axis=1,inplace=True)

train_pred["Reg_Score_Low"] = (train_pred.Reg_WScore_Low + train_pred.Reg_LScore_Low) / (train_pred.Reg_Win_Low + train_pred.Reg_Lose_Low)
train_pred["Reg_Score_High"] = (train_pred.Reg_WScore_High + train_pred.Reg_LScore_High) / (train_pred.Reg_Win_High + train_pred.Reg_Lose_High)
train_pred["Reg_Score_div"] = train_pred.Reg_Score_Low / train_pred.Reg_Score_High

test = test.merge(regular_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_Low"],right_on=["Season","WTeamID"],how="left")
test = test.merge(regular_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_High"],right_on=["Season","WTeamID"],how="left")
test = test.rename(columns={"DayNum_x":"Reg_Win_Low","DayNum_y":"Reg_Win_High"})
test = test.merge(regular_compact.groupby(["Season","LTeamID"]).count().reset_index().loc[:,["Season","LTeamID","DayNum"]],left_on=["Season","TeamID_Low"],right_on=["Season","LTeamID"],how="left")
test = test.merge(regular_compact.groupby(["Season","LTeamID"]).count().reset_index().loc[:,["Season","LTeamID","DayNum"]],left_on=["Season","TeamID_High"],right_on=["Season","LTeamID"],how="left")
test = test.rename(columns={"DayNum_x":"Reg_Lose_Low","DayNum_y":"Reg_Lose_High"})
test.drop(["WTeamID_x","WTeamID_y","LTeamID_x","LTeamID_y"],axis=1,inplace=True)
test["Reg_WinRate_Low"] = test.Reg_Win_Low/(test.Reg_Win_Low+test.Reg_Lose_Low)
test["Reg_WinRate_High"] = test.Reg_Win_High/(test.Reg_Win_High+test.Reg_Lose_High)
test["Reg_WinRate_div"] = test.Reg_WinRate_Low / test.Reg_WinRate_High

test = test.merge(regular_compact.groupby(["Season","WTeamID"]).mean().reset_index().loc[:,["Season","WTeamID","WScore"]],left_on=["Season","TeamID_Low"],right_on=["Season","WTeamID"],how="left")
test = test.merge(regular_compact.groupby(["Season","WTeamID"]).mean().reset_index().loc[:,["Season","WTeamID","WScore"]],left_on=["Season","TeamID_High"],right_on=["Season","WTeamID"],how="left")
test = test.rename(columns={"WScore_x":"Reg_WScore_Low","WScore_y":"Reg_WScore_High"})
test = test.merge(regular_compact.groupby(["Season","LTeamID"]).mean().reset_index().loc[:,["Season","LTeamID","LScore"]],left_on=["Season","TeamID_Low"],right_on=["Season","LTeamID"],how="left")
test = test.merge(regular_compact.groupby(["Season","LTeamID"]).mean().reset_index().loc[:,["Season","LTeamID","LScore"]],left_on=["Season","TeamID_High"],right_on=["Season","LTeamID"],how="left")
test = test.rename(columns={"LScore_x":"Reg_LScore_Low","LScore_y":"Reg_LScore_High"})
test.drop(["WTeamID_x","WTeamID_y","LTeamID_x","LTeamID_y"],axis=1,inplace=True)

test["Reg_Score_Low"] = (test.Reg_WScore_Low + test.Reg_LScore_Low) // (test.Reg_Win_Low + test.Reg_Lose_Low)
test["Reg_Score_High"] = (test.Reg_WScore_High + test.Reg_LScore_High) // (test.Reg_Win_High + test.Reg_Lose_High)
test["Reg_Score_div"] = test.Reg_Score_Low / test.Reg_Score_High

train_pred = train_pred.merge(rank.groupby(["Season","TeamID"]).mean().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
train_pred = train_pred.merge(rank.groupby(["Season","TeamID"]).mean().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
train_pred.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
train_pred = train_pred.rename(columns={"OrdinalRank_x":"Rank_Low","OrdinalRank_y":"Rank_High"})
train_pred = train_pred.merge(rank.groupby(["Season","TeamID"]).max().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
train_pred = train_pred.merge(rank.groupby(["Season","TeamID"]).max().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
train_pred.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
train_pred = train_pred.rename(columns={"OrdinalRank_x":"maxRank_Low","OrdinalRank_y":"maxRank_High"})
train_pred = train_pred.merge(rank.groupby(["Season","TeamID"]).min().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
train_pred = train_pred.merge(rank.groupby(["Season","TeamID"]).min().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
train_pred.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
train_pred = train_pred.rename(columns={"OrdinalRank_x":"minRank_Low","OrdinalRank_y":"minRank_High"})
train_pred["Rank_div"] = train_pred.Rank_Low / train_pred.Rank_High
train_pred["minRank_div"] = train_pred.minRank_Low / train_pred.minRank_High
train_pred["maxRank_div"] = train_pred.maxRank_Low / train_pred.maxRank_High

test = test.merge(rank.groupby(["Season","TeamID"]).mean().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
test = test.merge(rank.groupby(["Season","TeamID"]).mean().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
test.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
test = test.rename(columns={"OrdinalRank_x":"Rank_Low","OrdinalRank_y":"Rank_High"})
test = test.merge(rank.groupby(["Season","TeamID"]).max().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
test = test.merge(rank.groupby(["Season","TeamID"]).max().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
test.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
test = test.rename(columns={"OrdinalRank_x":"maxRank_Low","OrdinalRank_y":"maxRank_High"})
test = test.merge(rank.groupby(["Season","TeamID"]).min().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_Low"],right_on=["Season","TeamID"],how="left")
test = test.merge(rank.groupby(["Season","TeamID"]).min().reset_index().loc[:,["Season","TeamID","OrdinalRank"]],left_on=["Season","TeamID_High"],right_on=["Season","TeamID"],how="left")
test.drop(["TeamID_x","TeamID_y"],axis=1,inplace=True)
test = test.rename(columns={"OrdinalRank_x":"minRank_Low","OrdinalRank_y":"minRank_High"})
test["Rank_div"] = test.Rank_Low / test.Rank_High
test["minRank_div"] = test.minRank_Low / test.minRank_High
test["maxRank_div"] = test.maxRank_Low / test.maxRank_High

train_pred = train_pred.merge(second_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_Low"],right_on=["Season","WTeamID"],how="left")
train_pred = train_pred.merge(second_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_High"],right_on=["Season","WTeamID"],how="left")
train_pred = train_pred.rename(columns={"DayNum_x":"Sec_Win_Low","DayNum_y":"Sec_Win_High"})
train_pred.drop(["WTeamID_x","WTeamID_y"],axis=1,inplace=True)

test = test.merge(second_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_Low"],right_on=["Season","WTeamID"],how="left")
test = test.merge(second_compact.groupby(["Season","WTeamID"]).count().reset_index().loc[:,["Season","WTeamID","DayNum"]],left_on=["Season","TeamID_High"],right_on=["Season","WTeamID"],how="left")
test = test.rename(columns={"DayNum_x":"Sec_Win_Low","DayNum_y":"Sec_Win_High"})
test.drop(["WTeamID_x","WTeamID_y"],axis=1,inplace=True)

rand = 1024
params = {
"objective": "binary",
"boosting": "dart",
"max_depth": 3,
"num_leaves": 41,
"subsample": 0.9212067639630617,
"subsample_freq": 1,
"bagging_seed": rand,
"learning_rate": 0.03565146873857785,
#"feature_fraction": 0.8,
#"feature_seed": rand,
# "min_data_in_leaf": 200,
"lambda_l1": 6,
"lambda_l2": 15,
"random_state": rand,
"metric": "binary_logloss"
}
sub = pd.DataFrame()
df_fimp = pd.DataFrame()
select_cols = ["Seed_Low","Seed_High","Seed_div","Reg_Win_Low","Reg_Win_High","Reg_Lose_Low","Reg_Lose_High",
            "Reg_WinRate_Low","Reg_WinRate_High","Rank_Low","Rank_High","Sec_Win_Low","Sec_Win_High",
            "maxRank_Low","maxRank_High","minRank_Low","minRank_High","Reg_WinRate_div","Rank_div"]#,
            #"Reg_LScore_Low","Reg_LScore_High","Reg_WScore_Low","Reg_WScore_High"]
for season in range(2015,2020):
    train = train_pred.loc[(train_pred.Season <= season-1)]# & (train_pred.Season >= season-20)]
    X_train = train.copy()
    X_train = X_train.loc[:,select_cols]

    y_train = train.target
    X_test = test.loc[test.Season == season]
    X_te = X_test.copy()
    X_te = X_te.loc[:,select_cols]
    
    train_seasons = train.Season.unique()
    n_split = 3
    pred = np.zeros(len(X_test))
    folds = KFold(n_splits=n_split)
    for tr_idx,va_idx in folds.split(X_train,y_train):
        y_tr = y_train.iloc[tr_idx]
        X_tr = X_train.iloc[tr_idx]
        y_val = y_train.iloc[va_idx]
        X_val = X_train.iloc[va_idx]

        d_tr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        d_val = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        model = lgb.train(params, train_set=d_tr, num_boost_round=2696, valid_sets=[d_tr,d_val], verbose_eval=100, early_stopping_rounds=50)

        pred += model.predict(X_te, num_iteration=model.best_iteration) / n_split
        
        df_fimp_1 = pd.DataFrame()
        df_fimp_1["feature"] = X_tr.columns.values
        df_fimp_1["importance"] = model.feature_importance()

        df_fimp = pd.concat([df_fimp, df_fimp_1], axis=0)
    X_test["Pred"] = pred
    sub = sub.append(X_test)

plt.hist(sub.Pred)

plt.figure(figsize=(14, 14))
sns.barplot(x="importance", y="feature", data=df_fimp.sort_values(by="importance", ascending=False))
plt.title("LightGBM Feature Importance")
plt.tight_layout()

score = train_pred[(train_pred.Season >= 2015) & (train_pred.Season <= 2019)].copy()
score = score.merge(sub,on=["Season","TeamID_Low","TeamID_High"],how="left")
print(score.loc[:,["target","Pred"]].head(100))

print(log_loss(score.target,score.Pred))

sub["ID"] = sub["Season"].astype(str).str.cat(sub["TeamID_Low"].astype(str),sep="_").str.cat(sub["TeamID_High"].astype(str),sep="_")
sub = sub.loc[:,["ID","Pred"]]

sub.to_csv("submission.csv",index=False)
