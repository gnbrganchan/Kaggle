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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm

plt.style.use("seaborn")
sns.set(font_scale=1)

sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
test = pd.read_csv("../input/data-science-bowl-2019/test.csv")
train = pd.read_csv("../input/data-science-bowl-2019/train.csv")
train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
# test_labels = pd.read_csv("../input/dsb-each-attemption-result-in-test-data/results_in_test_agg.csv")

specs = specs.rename(columns={'info': 'information'})
specs["information"] = specs["information"].str[:40]
train["timestamp"] = pd.to_datetime(train['timestamp'])
test["timestamp"] = pd.to_datetime(test['timestamp'])

test_labels = prepare(test,True)
test_labels = agg(test_labels)
test_labels = test_labels.reset_index()

keep_id = train_labels[['installation_id']].drop_duplicates()
train = pd.merge(train, keep_id, on="installation_id", how="inner")
train = pd.merge(train,specs.loc[:,["event_id","information"]],on="event_id", how="left")
test = pd.merge(test,specs.loc[:,["event_id","information"]],on="event_id", how="left")

le_id = LabelEncoder()
le_id.fit_transform(pd.concat((train.installation_id,test.installation_id),axis=0).unique())
train.installation_id= le_id.transform(train.installation_id)
train_labels.installation_id= le_id.transform(train_labels.installation_id)
test.installation_id= le_id.transform(test.installation_id)
test_labels.installation_id= le_id.transform(test_labels.installation_id)

le = LabelEncoder()
le.fit_transform(pd.concat((train.game_session,test.game_session),axis=0).unique())
train.game_session= le.transform(train.game_session)
train_labels.game_session= le.transform(train_labels.game_session)
test.game_session= le.transform(test.game_session)
test_labels.game_session= le.transform(test_labels.game_session)

le_info = LabelEncoder()
le_info.fit_transform(specs.information)
train.information = le_info.transform(train.information)
test.information = le_info.transform(test.information)

df_info = pd.DataFrame({'info':le_info.classes_, 'code':le_info.transform(le_info.classes_)}).head(200)
asslist = train_labels.title.unique()
infolist = range(df_info.code.max()+1)
codelist = train.event_code.unique()
typlist = train.type.unique()#["Clip", "Game", "Activity", "Assessment"]
desc_in_infolist = ["GreatJob", "WhoaSoCool", "Amazing", "GoLower","Dot_Wow", "SoHigh", "NEXTSTONE", "NOT_THAT_HEAVY","RIGHTANSWER1","water_success"]
# print(asslist)

def features(X_df,df,stime):
    cnz = np.count_nonzero
    grp = df.groupby("game_session")
    typ = df.type
    evcode = df.event_code
    evcount1 = df.event_count == 1
    info = df.information
    X_df["sum_max_count"] = grp["event_count"].max().sum()
    X_df["mean_max_gametime"] = grp["game_time"].max().mean()#//(60*15)
    X_df["std_max_gametime"] = grp["game_time"].max().std()#//(60*15)
    X_df["sum_max_gametime"] = grp["game_time"].max().sum()#//(60*15)
    X_df["unique_game_session"] = df.game_session.nunique()
    X_df["unique_title"] = df.title.nunique()

    for i in typlist:
        X_df["count_{}".format(i)] = cnz((typ == i) & (evcount1))
    for i in codelist:
        X_df["count_{}".format(i)] = cnz(evcode == i)
    for i in infolist:
        X_df["count_info_{}".format(i)] = cnz(info == i)
        
    ### last session ###
    X_df["time_from_last_session"] = (stime - df.timestamp.max()).seconds#//(60*15)#10minuts
    
    df_last = df[df.timestamp == df.timestamp.max()].copy()
#    if len(df_last) == 0:
#        X_df["last_session_title"] = "None"
#    else:
#        X_df["last_session_title"] = df_last.title.max()
    if len(df_last) == 0:
        X_df["last_session_type"] = "None"
    else:
        X_df["last_session_type"] = df_last.type.max()
    X_df["last_session_max_count"] = df_last["event_count"].max()
    X_df["last_session_max_gametime"] = df_last["game_time"].max()#//(60*15)
    
#     ### last2 session ###
#     last_gs = df_last.game_session.max()
#     df_last2 = df[df.game_session != last_gs].copy()
#     df_last2 = df_last2[df_last2.timestamp == df_last2.timestamp.max()].copy()
#     X_df["time_from_last2_session"] = (stime - df_last2.timestamp.max()).total_seconds()
# #     if len(df_last2) == 0:
# #         X_df["last2_session_title"] = "None"
# #     else:
# #         X_df["last2_session_title"] = df_last2.title.max()
#     X_df["last2_session_max_count"] = df_last2["event_count"].max()
#     X_df["last2_session_max_gametime"] = df_last2["game_time"].max()
    
#     ### last3 session ###
#     last2_gs = df_last2.game_session.max()
#     df_last3 = df_last2[df_last2.game_session != last_gs].copy()
#     df_last3 = df_last3[df_last3.timestamp == df_last3.timestamp.max()].copy()
#     X_df["time_from_last3_session"] = (stime - df_last3.timestamp.max()).total_seconds()
# #     if len(df_last3) == 0:
# #         X_df["last3_session_title"] = "None"
# #     else:
# #         X_df["last3_session_title"] = df_last3.title.max()
#     X_df["last3_session_max_count"] = df_last3["event_count"].max()
#     X_df["last3_session_max_gametime"] = df_last3["game_time"].max()
    
    ### assessment ###
    df_ass = df[(typ=="Assessment") & (((evcode==4110) & (df.title=="Bird Measurer (Assessment)")) | ((evcode==4100) & (df.title!="Bird Measurer (Assessment)")))].copy()
    X_df["count_correct"] = df_ass.event_data.str.contains('"correct":true').sum()
    X_df["count_incorrect"] = df_ass.event_data.str.contains('"correct":false').sum()
    if X_df.count_correct.max()+X_df.count_incorrect.max() == 0:
        X_df["rate_correct"] = -1
    else:
        X_df["rate_correct"] = X_df.count_correct.max()/(X_df.count_correct.max()+X_df.count_incorrect.max())
    for i in asslist:
        X_df["count_correct_{}".format(i)] = df_ass[df_ass.title == i].event_data.str.contains('"correct":true').sum()
        X_df["count_incorrect_{}".format(i)] = df_ass[df_ass.title == i].event_data.str.contains('"correct":false').sum()
        if X_df["count_correct_{}".format(i)].max()+X_df["count_incorrect_{}".format(i)].max() == 0:
            X_df["rate_correct_{}".format(i)] = -1
        else:
            X_df["rate_correct_{}".format(i)] = X_df["count_correct_{}".format(i)].max()/(X_df["count_correct_{}".format(i)].max()+X_df["count_incorrect_{}".format(i)].max())
    
    ### last assessment ###
    last_ass_gs = df_ass[df_ass.timestamp == df_ass.timestamp.max()].game_session.max()
    df_last_ass = df_ass[df_ass.game_session == last_ass_gs].copy()
    if len(df_last_ass) == 0:
        X_df["last_ass_title"] = "None"
    else:
        X_df["last_ass_title"] = df_last_ass.title.max()
    X_df["last_count_correct"] = df_last_ass.event_data.str.contains('"correct":true').sum()
    X_df["last_count_incorrect"] = df_last_ass.event_data.str.contains('"correct":false').sum()
    X_df["time_from_last_ass"] = (stime - df_last_ass.timestamp.max()).seconds#//(60*15) #10分単位
    
#     ### last 7 day ###
#     df_7days = df[(stime - df.timestamp).apply(lambda x: x.days < 7)].copy()
#     if len(df_7days) == 0:
#         return X_df
#     X_df["7days_sum_max_count"] = df_7days.groupby("game_session")["event_count"].max().sum()
#     X_df["7days_mean_max_gametime"] = df_7days.groupby("game_session")["game_time"].max().mean()#//(60*15)
#     X_df["7days_std_max_gametime"] = df_7days.groupby("game_session")["game_time"].max().std()#//(60*15)
#     X_df["7days_sum_max_gametime"] = df_7days.groupby("game_session")["game_time"].max().sum()#//(60*15)
#     X_df["7days_unique_game_session"] = df_7days.game_session.nunique()
#     X_df["7days_unique_title"] = df_7days.title.nunique()
#     for i in typlist:
#         X_df["7days_count_{}".format(i)] = cnz((df_7days.type == i) & (df_7days.event_count == 1))
    
#     ### last 1 day ###
#     df_today = df[(stime - df.timestamp).apply(lambda x: x.days < 1)].copy()
#     if len(df_today) == 0:
#         return X_df
#     X_df["today_sum_max_count"] = df_today.groupby("game_session")["event_count"].max().sum()
#     X_df["today_mean_max_gametime"] = df_today.groupby("game_session")["game_time"].max().mean()#//(60*15)
#     X_df["today_std_max_gametime"] = df_today.groupby("game_session")["game_time"].max().std()#//(60*15)
#     X_df["today_sum_max_gametime"] = df_today.groupby("game_session")["game_time"].max().sum()#//(60*15)
#     X_df["today_unique_game_session"] = df_today.game_session.nunique()
#     X_df["today_unique_title"] = df_today.title.nunique()
#     for i in typlist:
#         X_df["today_count_{}".format(i)] = cnz((df_today.type == i) & (df_today.event_count == 1))

#     for i in desc_in_infolist:
#         X_df["count_{}".format(i)] = df.event_data.str.contains(i).sum()
#     hour = stime.hour
#     X_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
#     X_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
#    df_today = df[(df.timestamp.dt.month == stime.month) & (df.timestamp.dt.day == stime.day)].copy()
#    X_df["today_sum_max_count"] = df_today.groupby("game_session")["event_count"].max().sum()
#    X_df["today_mean_max_gametime"] = df_today.groupby("game_session")["game_time"].max().mean()
#    X_df["today_sum_max_gametime"] = df_today.groupby("game_session")["game_time"].max().sum()
#    X_df["today_duration_gametime"] = df_today.groupby("game_session")["game_time"].max().sum()/len(df_today)
#    X_df["today_unique_game_session"] = df_today.game_session.nunique()
#    for i in typlist:
#        X_df["today_count_{}".format(i)] = cnz((df_today.type == i) & (df_today.event_count == 1))

    return X_df

def make_dat(X_df,df):
    df_out = pd.DataFrame()
    ids = X_df.installation_id.unique()
    for id in tqdm(ids):
        df_id = df[df.installation_id == id]
        X_df_id = X_df[X_df.installation_id == id]
        sessions = X_df_id.game_session.unique()
        for gs in sessions:
            X_df_id_gs = X_df_id[X_df_id.game_session == gs]
            stime = df_id[df_id.game_session == gs].iloc[0].timestamp
            dfgs = features(X_df_id_gs.copy(),df_id[(df_id.timestamp < stime) & ((stime - df_id.timestamp).apply(lambda x: x.days <= 21))].copy(),stime)
            df_out = df_out.append(dfgs)         
    return df_out
    
print(len(train_labels))
X_train = train_labels.merge(train[((train.type=="Assessment") &  (train.event_count==1))].loc[:,["game_session","installation_id","world","timestamp"]],on=["installation_id","game_session"],how="left")
X_train.drop(["timestamp"],axis=1,inplace=True)
y_train = X_train.accuracy_group

X_train = make_dat(X_train, train)
print(X_train.columns)

X_test = test[(test.type=="Assessment") & (test.event_count==1)].loc[:,["game_session","installation_id","title","world","timestamp"]].copy()
# X_test = X_test.head(100) # for develop
X_test = make_dat(X_test,test)
print(X_test.columns)

def stract_hists(feature, train=X_train, test=X_test, adjust=False, plot=False):
    n_bins = 10
    train_data = train[feature]
    test_data = test[feature]
    if adjust:
        test_data *= train_data.mean() / test_data.mean()
    perc_90 = np.percentile(train_data, 95)
    train_data = np.clip(train_data, 0, perc_90)
    test_data = np.clip(test_data, 0, perc_90)
    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)
    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)
    msre = mean_squared_error(train_hist, test_hist)
    if plot:
        print(msre)
        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)
        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)
        plt.show()
    return msre
# stract_hists('Magma Peak - Level 1_2000', adjust=False, plot=True)

counter = 0
to_remove = []
features = X_train.columns
for feat_a in features:
    for feat_b in features:
        if type(X_train[feat_a].max()) is str or type(X_train[feat_b].max()) is str:
            continue
        if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
            c = np.corrcoef(X_train[feat_a], X_train[feat_b])[0][1]
            if c > 0.9:
                counter += 1
                to_remove.append(feat_b)
                print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
                
to_exclude = [] 
# ajusted_test = X_test.copy()
for feature in ajusted_test.columns:
    if type(X_test[feature].max()) is str or type(X_test[feature].max()) == pd._libs.tslibs.timestamps.Timestamp:
        continue
    if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']:
        data = X_train[feature]
        train_mean = data.mean()
        data = X_test[feature] 
        test_mean = data.mean()
        try:
            error = stract_hists(feature, adjust=True)
            ajust_factor = train_mean / test_mean
            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:
                to_exclude.append(feature)
                print(feature, train_mean, test_mean, error)
            else:
                X_test[feature] *= ajust_factor
        except:
            to_exclude.append(feature)
            print(feature, train_mean, test_mean)
            
 print(to_remove)
to_remain = ["accuracy","accuracy_group","num_correct","num_incorrect"]
features = [x for x in X_train.columns if x not in (to_remove + to_exclude) or x in to_remain]
X_train = X_train[features]
# features = [x for x in X_test.columns if x not in (to_remove + to_exclude) or x in to_remain]
# X_test = X_test[features]

436.0
le = LabelEncoder()
X_train.title= le.fit_transform(X_train.title)
X_test.title= le.transform(X_test.title)

#le = LabelEncoder()
#le.fit_transform(pd.concat((train.title,X_train.last_session_title),axis=0).unique())
#X_train.last_session_title= le.transform(X_train.last_session_title)
#X_test.last_session_title= le.transform(X_test.last_session_title)

le = LabelEncoder()
le.fit_transform(pd.concat((train.type,X_train.last_session_type),axis=0).unique())
X_train.last_session_type= le.transform(X_train.last_session_type)
X_test.last_session_type= le.transform(X_test.last_session_type)

# le = LabelEncoder()
# le.fit_transform(pd.concat((train.title,X_train.last2_session_title),axis=0).unique())
# X_train.last2_session_title= le.transform(X_train.last2_session_title)
# X_test.last2_session_title= le.transform(X_test.last2_session_title)

le = LabelEncoder()
le.fit_transform(pd.concat((train_labels.title,X_train.last_ass_title),axis=0).unique())
X_train.last_ass_title= le.transform(X_train.last_ass_title)
X_test.last_ass_title= le.transform(X_test.last_ass_title)

le = LabelEncoder()
X_train.world= le.fit_transform(X_train.world)
X_test.world= le.transform(X_test.world)

X_train.drop(["installation_id","game_session"],axis=1,inplace=True)
X_te = X_test.copy()
X_te.drop(["installation_id","game_session","timestamp"],axis=1,inplace=True)
#X_te.drop(["game_session","timestamp"],axis=1,inplace=True)

X_train_origin = X_train.copy()
y_train_origin = y_train.copy()
train_labels_origin = train_labels.copy()

val = X_test.merge(test_labels.loc[:,["installation_id","game_session","accuracy_group"]],on=["installation_id","game_session"],how="left").copy()
val = val[~np.isnan(val.accuracy_group)]
val_x = val.copy()
val_x.drop(["installation_id","game_session","timestamp","accuracy_group"],axis=1,inplace=True)
print(val_x.columns)
print(val.accuracy_group)

categorical_features = ["title","world","last_session_type","last_ass_title"]

nsplits = 5
nrand = 10

df_first = pd.DataFrame()
df_fimp = pd.DataFrame()
init = True
X_train = X_train_origin.copy()
X_train.drop(["num_correct","num_incorrect","accuracy","accuracy_group"],axis=1,inplace=True)
y_train = y_train_origin
#X_train = val_x.copy()
#y_train = val.accuracy_group
#X_train = X_train.append(val_x)
#y_train = y_train.append(val.accuracy_group)

for rand in range(nrand):
    random.seed(rand)
#     indexes = X_train_origin.index.isin(X_train_origin[X_train_origin.accuracy_group==3].sample(frac=len(X_train_origin[X_train_origin.accuracy_group==0])/len(X_train_origin[X_train_origin.accuracy_group==3]),random_state=rand).index)
#     X_train = X_train_origin[~indexes].copy()
#     y_train = y_train_origin[~indexes].copy()
#     train_labels = train_labels_origin[~indexes].copy()
#     X_train.drop(["num_correct","num_incorrect","accuracy","accuracy_group"],axis=1,inplace=True)

    folds = KFold(n_splits=nsplits, shuffle=True, random_state=rand)
    params = {
    "objective": "regression",
    "boosting": "gbdt",
    "max_depth": 15,
    "num_leaves": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "bagging_seed": rand,
    "learning_rate": 0.025,
    "feature_fraction": 0.5,
    "feature_seed": rand,
    "min_data_in_leaf": 200,
    "lambda_l1": 20,
    "lambda_l2": 2,
    "random_state": rand,
    "metric": "rmse"
    }
    pred = np.zeros(len(X_te))
    tpred = np.zeros(len(X_train))
    for tr_idx,va_idx in folds.split(X_train,y_train):
        X_half_1 = X_train.iloc[tr_idx]
        y_half_1 = y_train.iloc[tr_idx]
        X_half_2 = val_x#X_train.iloc[va_idx]
        y_half_2 = val.accuracy_group#y_train.iloc[va_idx]

        d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, categorical_feature=categorical_features, free_raw_data=False)
        d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, categorical_feature=categorical_features, free_raw_data=False)

        model_half = lgb.train(params, train_set=d_half_1, num_boost_round=2000, valid_sets=[d_half_1,d_half_2], verbose_eval=100, early_stopping_rounds=50)

        tpred += model_half.predict(X_train, num_iteration=model_half.best_iteration) / nsplits
        pred += model_half.predict(X_te, num_iteration=model_half.best_iteration) / nsplits

        df_fimp_1 = pd.DataFrame()
        df_fimp_1["feature"] = X_train.columns.values
        df_fimp_1["importance"] = model_half.feature_importance()

        df_fimp = pd.concat([df_fimp, df_fimp_1], axis=0)
        del model_half
        gc.collect()
        
    df_first["pred_{}".format(rand)] = pred

    if init:
        pred_total = pred
        tpred_total = tpred
        tlabel = train_labels.accuracy_group
        init = False
    else:
#         pred_total = np.concatenate([pred_total,pred])
        pred_total += pred
        tpred_total = np.concatenate([tpred_total,tpred])
#         tpred_total += tpred
        tlabel = np.concatenate([tlabel,train_labels.accuracy_group])
pred_total /= nrand

plt.figure(figsize=(14, 14))
sns.barplot(x="importance", y="feature", data=df_fimp.sort_values(by="importance", ascending=False))
plt.title("LightGBM Feature Importance")
plt.tight_layout()

p = pd.DataFrame({"lab":tlabel, "pred":tpred_total,"pclass": np.trunc(tpred_total*100)})
sns.distplot(p[p.lab==0].pred, kde=False, rug=False, bins=100) 

# X_train = X_train_origin[~X_train_origin.index.isin(X_train_origin[X_train_origin.accuracy_group==3].sample(frac=len(X_train_origin[X_train_origin.accuracy_group==0])/len(X_train_origin[X_train_origin.accuracy_group==3]),random_state=0).index)].copy()
X_train = X_train_origin.copy()
psort = pred_total.copy()
# psort = tpred_total.copy()
psort.sort()
X_train_val = X_train.append(val)
th1 = psort[int(np.count_nonzero(X_train.accuracy_group<1)/len(X_train)*len(psort))]
th2 = psort[int(np.count_nonzero(X_train.accuracy_group<2)/len(X_train)*len(psort))]
th3 = psort[int(np.count_nonzero(X_train.accuracy_group<3)/len(X_train)*len(psort))]
#th1 = psort[int(np.count_nonzero(val.accuracy_group<1)/len(val)*len(psort))]
#th2 = psort[int(np.count_nonzero(val.accuracy_group<2)/len(val)*len(psort))]
#th3 = psort[int(np.count_nonzero(val.accuracy_group<3)/len(val)*len(psort))]
# th1 = psort[int(np.count_nonzero(X_train_val.accuracy_group<1)/len(X_train_val)*len(psort))]
# th2 = psort[int(np.count_nonzero(X_train_val.accuracy_group<2)/len(X_train_val)*len(psort))]
# th3 = psort[int(np.count_nonzero(X_train_val.accuracy_group<3)/len(X_train_val)*len(psort))]
print(th1,th2,th3)


X_te_agg = X_test.copy()
X_te_agg["accuracy_group"] = 0
X_te_agg.loc[pred_total > th1,"accuracy_group"] = 1
X_te_agg.loc[pred_total > th2,"accuracy_group"] = 2
X_te_agg.loc[pred_total > th3,"accuracy_group"] = 3
X_te_agg = X_te_agg.merge(test_labels.loc[:,["installation_id","game_session","accuracy_group"]],on=["installation_id","game_session"],how="left")
eval = X_te_agg[~X_te_agg.accuracy_group_y.isna()]
print(quadratic_weighted_kappa(eval.reset_index().accuracy_group_x.astype("int"),eval.reset_index().accuracy_group_y.astype("int")))

X_te_agg["accuracy_group"] = np.where(X_te_agg.accuracy_group_y.isna(),X_te_agg.accuracy_group_x,X_te_agg.accuracy_group_y)
X_te_agg.drop(["accuracy_group_x","accuracy_group_y"],axis=1,inplace=True)
# X_te_agg = X_te_agg.merge(test.loc[(test.type=="Assessment") & (test.event_count==1),["installation_id","game_session","timestamp"]],on=["installation_id","game_session"],how="left")
#print(X_te_agg.head(20))

NameError: name 'pred_total' is not defined
X_te_ans = X_te_agg.iloc[X_te_agg.groupby("installation_id")["timestamp"].idxmax()][["installation_id","accuracy_group"]].astype({"accuracy_group": int}).reset_index()[["installation_id","accuracy_group"]]
# X_te_ans = X_te_agg.groupby("installation_id")["accuracy_group"].last().astype({"accuracy_group": int}).reset_index()
X_te_ans.installation_id = le_id.inverse_transform(X_te_ans.installation_id)
X_te_ans.to_csv("submission.csv",index=False)
print(len(X_te_ans))
print(X_te_ans.head(10))

for i in range(4):
    print(len(X_te_ans[X_te_ans.accuracy_group==i]))
