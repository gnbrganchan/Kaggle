import os
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt
import seaborn as sns

import lightgbm as lgb

from dataclasses import dataclass

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import json
#最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_rows', 100)
#最大表示列数の指定（ここでは50列を指定）
pd.set_option('display.max_columns', 50)

# parameter
rand = 1024
lgb_params = {
    'objective': 'regression',
    'boosting_type': 'gbdt', #default=gbdt
#     'max_depth': 10,
    'n_estimators': 50000, #default=100*num_class
    'learning_rate': 0.04, #default=0.1
    'num_leaves': 300, #default=31
    'colsample_bytree': 0.8, #default=1.0 (% of cols selected)
    'subsample': 0.7, #default=1.0 (% of rows selected)
    'subsample_freq': 2, #default=0 (perform bagging every kth iteration)
    'bagging_seed': 42, #default=3
    'reg_alpha': 0, #default=0.0 (L1 regularization)
    'reg_lambda': 0, #default=0.0 (L2 regularization)
    'random_state': 42, #default=None
    'metric': 'RMSE',
    'n_jobs': -1, #default=0
}

def interp(df,path_id):
    indexes = df[df["x"].isnull()].index
    df_predict_backward = pd.DataFrame([],columns=["x","y"],index=indexes)
    df_predict_forward = pd.DataFrame([],columns=["x","y"],index=indexes)
    for idx in range(len(indexes)-1):
        df_points = df[(df["timestamp"] > indexes[idx]) & (df["timestamp"] < indexes[idx+1])]
        if len(df_points) > 1:
            lr_x = LinearRegression()
            lr_x.fit(df_points["timestamp"].values.reshape(-1,1),df_points["x"])
            lr_y = LinearRegression()
            lr_y.fit(df_points["timestamp"].values.reshape(-1,1),df_points["y"])
            df_predict_backward.loc[indexes[idx],"x"] = lr_x.predict(indexes[idx].reshape(-1,1))[0]
            df_predict_backward.loc[indexes[idx],"y"] = lr_y.predict(indexes[idx].reshape(-1,1))[0]
            df_predict_forward.loc[indexes[idx+1],"x"] = lr_x.predict(indexes[idx+1].reshape(-1,1))[0]
            df_predict_forward.loc[indexes[idx+1],"y"] = lr_y.predict(indexes[idx+1].reshape(-1,1))[0]
    df_predict = pd.concat([df_predict_forward,df_predict_backward]).groupby("timestamp").sum()/pd.concat([df_predict_forward,df_predict_backward]).groupby("timestamp").count()
    df_predict["path_id"] = path_id
    return df_predict
  
  def interp2(df,df_rel,path_id):
    df = df.reset_index(drop=True)
    n_near = 10
    df = pd.merge(df,df_rel[["path_id","timestamp","rel_x","rel_y"]],how="left",on=["path_id","timestamp"])
    for shift_len in range(1,n_near+1):
        df[f"x_prev{shift_len}"] = df.shift(shift_len)["x"]
        df[f"y_prev{shift_len}"] = df.shift(shift_len)["y"]
        df[f"x_next{shift_len}"] = df.shift(-shift_len)["x"]
        df[f"y_next{shift_len}"] = df.shift(-shift_len)["y"]        
        df[f"x_prev{shift_len}+rel"] = df[f"x_prev{shift_len}"]+df["rel_x"]-df.shift(shift_len)["rel_x"]
        df[f"y_prev{shift_len}+rel"] = df[f"y_prev{shift_len}"]+df["rel_y"]-df.shift(shift_len)["rel_y"]
        df[f"x_next{shift_len}+rel"] = df[f"x_next{shift_len}"]+df["rel_x"]-df.shift(-shift_len)["rel_x"]
        df[f"y_next{shift_len}+rel"] = df[f"y_next{shift_len}"]+df["rel_y"]-df.shift(-shift_len)["rel_y"]
    xcols = sum([[f"x_prev{i}+rel",f"x_next{i}+rel"]for i in range(1,n_near+1)],[])
    ycols = sum([[f"y_prev{i}+rel",f"y_next{i}+rel"]for i in range(1,n_near+1)],[])
    df["x"] = df[xcols].mean(axis=1)
    df["y"] = df[ycols].mean(axis=1)
#     df["x"] = df[xcols].median(skipna=True,axis=1)
#     df["y"] = df[ycols].median(skipna=True,axis=1)

    df_predict = df[["timestamp","x","y"]].copy()
    df_predict.index = df_predict.timestamp
    df_predict = df_predict[~df_predict.isnull().any(axis=1)]
    df_predict["path_id"] = path_id
    return df_predict
  
  def interp3(df,df_rel):
    df = df.reset_index(drop=True)
    n_near = 10
    df = pd.merge(df,df_rel[["path_id","timestamp","rel_x","rel_y"]],how="left",on=["path_id","timestamp"])
    for shift_len in range(1,n_near+1):
        df[f"x_prev{shift_len}"] = df.groupby("path_id").shift(shift_len)["x"]
        df[f"y_prev{shift_len}"] = df.groupby("path_id").shift(shift_len)["y"]
        df[f"x_next{shift_len}"] = df.groupby("path_id").shift(-shift_len)["x"]
        df[f"y_next{shift_len}"] = df.groupby("path_id").shift(-shift_len)["y"]        
        df[f"x_prev{shift_len}+rel"] = df[f"x_prev{shift_len}"]+df["rel_x"]-df.groupby("path_id").shift(shift_len)["rel_x"]
        df[f"y_prev{shift_len}+rel"] = df[f"y_prev{shift_len}"]+df["rel_y"]-df.groupby("path_id").shift(shift_len)["rel_y"]
        df[f"x_next{shift_len}+rel"] = df[f"x_next{shift_len}"]+df["rel_x"]-df.groupby("path_id").shift(-shift_len)["rel_x"]
        df[f"y_next{shift_len}+rel"] = df[f"y_next{shift_len}"]+df["rel_y"]-df.groupby("path_id").shift(-shift_len)["rel_y"]
    xcols = sum([[f"x_prev{i}+rel",f"x_next{i}+rel"]for i in range(1,n_near+1)],[])
    ycols = sum([[f"y_prev{i}+rel",f"y_next{i}+rel"]for i in range(1,n_near+1)],[])
    df["x"] = df[xcols].mean(axis=1)
    df["y"] = df[ycols].mean(axis=1)
#     df["x"] = df[xcols].median(skipna=True,axis=1)
#     df["y"] = df[ycols].median(skipna=True,axis=1)

    df_predict = df[["floor","path_id","timestamp","x","y","waypoint_x","waypoint_y"]].copy()
    df_predict["x_diff"] = np.abs(df_predict["x"]-df_predict["waypoint_x"])
    df_predict["y_diff"] = np.abs(df_predict["y"]-df_predict["waypoint_y"])
#     df_predict.index = df_predict.timestamp
#     df_predict = df_predict[df_predict.isnull().any(axis=1)]
    return df_predict

if not os.path.exists("train"): os.mkdir("train")
if not os.path.exists("trainp"): os.mkdir("trainp")
if not os.path.exists("test"): os.mkdir("test")    
  
  
df_sample = pd.read_csv("../input/predict-location-post01/submission.csv")
df_submit = df_sample.drop(["x","y"], axis = 1).copy()
df_sample["site_id"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[0])
df_sample["path_id"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[1])
df_sample["timestamp"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[2]).astype(float)

oof_x_all = []
oof_y_all = []
x_true = []
y_true = []
df_pred = pd.DataFrame()
df_out_all = pd.DataFrame()
est_score = 0.0
EST = 0.0

list_site = df_sample["site_id"].unique()
for site_id in tqdm(list_site):
    df_train_wp = pd.read_parquet(f"../input/martwp/train/{site_id}.parquet")
    df_sample_site = df_sample.query("site_id == @site_id")
    # read train and test mart
    df_train_pred = pd.read_csv(f"../input/predict-location-post01/train/{site_id}.csv")
    df_test_pred = pd.read_csv(f"../input/predict-location-post01/test/{site_id}.csv")
    df_test_pred = df_test_pred[df_test_pred.site_id.isnull()]
    df_train_rel = pd.read_parquet(f"../input/mart31/train/{site_id}.parquet").drop(["waypoint_x","waypoint_y","floor"],axis=1)
    df_test_rel = pd.read_parquet(f"../input/mart31/test/{site_id}.parquet")
    df_train_41 = pd.read_parquet(f"../input/mart41te01/train/{site_id}.parquet")
    df_test_41 = pd.read_parquet(f"../input/mart41te01/test/{site_id}.parquet")
    df_train_42 = pd.read_parquet(f"../input/mart42te01/train/{site_id}.parquet")
    df_test_42 = pd.read_parquet(f"../input/mart42te01/test/{site_id}.parquet")
    df_train_r = pd.read_parquet(f"../input/mart11-12-r/train/{site_id}.parquet")
    df_test_r = pd.read_parquet(f"../input/mart11-12-r/test/{site_id}.parquet")
    df_train_leak = pd.read_csv(f"../input/martpath/train/{site_id}.csv")
    df_test_leak = pd.read_csv(f"../input/martpath/test/{site_id}.csv")

    df_train = pd.merge(df_train_pred,df_train_rel,how="inner",on=["path_id","timestamp"])
    df_test = pd.merge(df_test_pred,df_test_rel,how="inner",on=["path_id","timestamp"])
    
    wificols = sum([[f"strong_wifi{i}_x",f"strong_wifi{i}_y",f"strong_wifi_val{i}"] for i in range(5)],[])
    ibeaconcols = sum([[f"strong_ibeacon{i}_x",f"strong_ibeacon{i}_y",f"strong_ibeacon_val{i}"] for i in range(1)],[])

    df_train = pd.merge(df_train,df_train_41[["path_id","timestamp"]+wificols],how="left",on=["path_id","timestamp"])
#     df_train = pd.merge(df_train,df_train_42[["path_id","timestamp"]+ibeaconcols],how="left",on=["path_id","timestamp"])
#     df_train = pd.merge(df_train,df_train_r[["path_id","timestamp","n_wifi","n_ibeacon"]],how="left",on=["path_id","timestamp"])
#     df_train = pd.merge(df_train,df_train_leak[["path_id","device"]],how="left",on=["path_id"])
    df_test = pd.merge(df_test,df_test_41[["path_id","timestamp"]+wificols],how="left",on=["path_id","timestamp"])  
#     df_test = pd.merge(df_test,df_test_42[["path_id","timestamp"]+ibeaconcols],how="left",on=["path_id","timestamp"])
#     df_test = pd.merge(df_test,df_test_r[["path_id","timestamp","n_wifi","n_ibeacon"]],how="left",on=["path_id","timestamp"])
#     df_test = pd.merge(df_test,df_test_leak[["path_id","device"]],how="left",on=["path_id"])

#     break
    # make train mart
    for shift_len in range(1,2):
        df_train[f"waypoint_x_pred_prev{shift_len}"] = df_train.groupby("path_id").shift(shift_len)["x"]
        df_train[f"waypoint_y_pred_prev{shift_len}"] = df_train.groupby("path_id").shift(shift_len)["y"]
        df_train[f"waypoint_x_pred_next{shift_len}"] = df_train.groupby("path_id").shift(-shift_len)["x"]
        df_train[f"waypoint_y_pred_next{shift_len}"] = df_train.groupby("path_id").shift(-shift_len)["y"]
        df_train[f"waypoint_x_pred_prev{shift_len}+rel"] = df_train[f"waypoint_x_pred_prev{shift_len}"]+df_train["rel_x"]-df_train.groupby("path_id").shift(shift_len)["rel_x"]
        df_train[f"waypoint_y_pred_prev{shift_len}+rel"] = df_train[f"waypoint_y_pred_prev{shift_len}"]+df_train["rel_y"]-df_train.groupby("path_id").shift(shift_len)["rel_y"]
        df_train[f"waypoint_x_pred_next{shift_len}+rel"] = df_train[f"waypoint_x_pred_next{shift_len}"]+df_train["rel_x"]-df_train.groupby("path_id").shift(-shift_len)["rel_x"]
        df_train[f"waypoint_y_pred_next{shift_len}+rel"] = df_train[f"waypoint_y_pred_next{shift_len}"]+df_train["rel_y"]-df_train.groupby("path_id").shift(-shift_len)["rel_y"]
        df_train.drop([f"waypoint_x_pred_prev{shift_len}",f"waypoint_y_pred_prev{shift_len}",f"waypoint_x_pred_next{shift_len}",f"waypoint_y_pred_next{shift_len}"],axis=1,inplace=True)

    df_train.drop(["rel_x","rel_y"],axis=1,inplace=True)

    # make test mart
    for shift_len in range(1,2):
        df_test[f"waypoint_x_pred_prev{shift_len}"] = df_test.groupby("path_id").shift(shift_len)["x"]
        df_test[f"waypoint_y_pred_prev{shift_len}"] = df_test.groupby("path_id").shift(shift_len)["y"]
        df_test[f"waypoint_x_pred_next{shift_len}"] = df_test.groupby("path_id").shift(-shift_len)["x"]
        df_test[f"waypoint_y_pred_next{shift_len}"] = df_test.groupby("path_id").shift(-shift_len)["y"]        
        df_test[f"waypoint_x_pred_prev{shift_len}+rel"] = df_test[f"waypoint_x_pred_prev{shift_len}"]+df_test["rel_x"]-df_test.groupby("path_id").shift(shift_len)["rel_x"]
        df_test[f"waypoint_y_pred_prev{shift_len}+rel"] = df_test[f"waypoint_y_pred_prev{shift_len}"]+df_test["rel_y"]-df_test.groupby("path_id").shift(shift_len)["rel_y"]
        df_test[f"waypoint_x_pred_next{shift_len}+rel"] = df_test[f"waypoint_x_pred_next{shift_len}"]+df_test["rel_x"]-df_test.groupby("path_id").shift(-shift_len)["rel_x"]
        df_test[f"waypoint_y_pred_next{shift_len}+rel"] = df_test[f"waypoint_y_pred_next{shift_len}"]+df_test["rel_y"]-df_test.groupby("path_id").shift(-shift_len)["rel_y"]
        df_test.drop([f"waypoint_x_pred_prev{shift_len}",f"waypoint_y_pred_prev{shift_len}",f"waypoint_x_pred_next{shift_len}",f"waypoint_y_pred_next{shift_len}"],axis=1,inplace=True)
    df_test.drop(["rel_x","rel_y"],axis=1,inplace=True)
    df_test = df_test
    
    df_kfold = df_train.groupby("path_id")["floor","waypoint_x","waypoint_y"].mean().reset_index()
    y1_train = df_train["waypoint_x"]
    y2_train = df_train["waypoint_y"]
    X_train = df_train.drop(["timestamp","waypoint_x","waypoint_y","path_id","x_diff","y_diff","floor"], axis = 1)
    X_test = df_test.drop(["site_path_timestamp","site_id","timestamp","path_id","floor"], axis = 1)
#     X_train["floor_pred"] = df_train["floor"]
#     X_test["floor_pred"] = df_test["floor"]
    
    groups = df_train["path_id"]
    floors = df_train["floor"]
    loc_labels = KMeans(n_clusters = 5, random_state=rand).fit_predict(df_kfold[["waypoint_x","waypoint_y"]])
    cat_features = []
    
    # initialize oof and pred
    oof_x = np.zeros(len(X_train))
    oof_y = np.zeros(len(X_train))
    pred_x = np.zeros(len(X_test))
    pred_y = np.zeros(len(X_test))

    # make fold
    nsplits = 40
    nsplits = min(nsplits,np.unique(loc_labels,return_counts=True)[1].min())
    folds = StratifiedKFold(n_splits = nsplits, shuffle = True, random_state = rand)

    # CV training and prediction
    df_fimp = pd.DataFrame()
    for tr_idx_p,va_idx_p in folds.split(df_kfold,loc_labels):
        tr_id, va_id = df_kfold.loc[tr_idx_p,"path_id"], df_kfold.loc[va_idx_p,"path_id"]
        tr_idx = df_train[df_train["path_id"].isin(tr_id)].index
        va_idx = df_train[df_train["path_id"].isin(va_id)].index

        X_tr = X_train.iloc[tr_idx].reset_index().drop("index",axis = 1)
        y1_tr = y1_train.iloc[tr_idx].reset_index().drop("index",axis = 1)
        y2_tr = y2_train.iloc[tr_idx].reset_index().drop("index",axis = 1)
        X_va = X_train.iloc[va_idx].reset_index().drop("index",axis = 1)
        y1_va = y1_train.iloc[va_idx].reset_index().drop("index",axis = 1)
        y2_va = y2_train.iloc[va_idx].reset_index().drop("index",axis = 1)
        
        model1 = lgb.LGBMRegressor(**lgb_params)
        model2 = lgb.LGBMRegressor(**lgb_params)
        # train and predict x
        model1.fit(X_tr, y1_tr,
                eval_set=[(X_va, y1_va)],
                verbose=False,
                early_stopping_rounds=20,
                categorical_feature=cat_features
                )
        oof_x[va_idx] = model1.predict(X_va)
        pred_x += model1.predict(X_test) / nsplits

        # train and predict y
        model2.fit(X_tr, y2_tr,
                eval_set=[(X_va, y2_va)],
                verbose=False,
                early_stopping_rounds=20,
                categorical_feature=cat_features
                )
        oof_y[va_idx] = model2.predict(X_va)
        pred_y += model2.predict(X_test) / nsplits
        df_fimp_1 = pd.DataFrame()
        df_fimp_1["feature"] = X_train.columns.values
        df_fimp_1["importance"] = model1.feature_importances_
        df_fimp = pd.concat([df_fimp, df_fimp_1], axis=0)

    # add oof data
    oof_x_all = np.append(oof_x_all,oof_x)
    oof_y_all = np.append(oof_y_all,oof_y)
    x_true = np.append(x_true, y1_train)
    y_true = np.append(y_true, y2_train)
    
    # interpolate train
    df_trainp_all = pd.DataFrame({"path_id":df_train["path_id"],"timestamp":df_train["timestamp"],"x":oof_x,"y":oof_y})
    df_trainp0 = pd.merge(df_train_wp,df_trainp_all,how="outer",on=["path_id","timestamp"]).sort_values(['path_id','timestamp'])
    df_trainp0.index = df_trainp0["timestamp"]
    for path_id in df_trainp0.path_id.unique():
#         df_trainp_path = interp(df_trainp0[(df_trainp0["path_id"] == path_id)].copy(),path_id)
        df_trainp_path = interp2(df_trainp0[(df_trainp0["path_id"] == path_id)].copy(),df_train_rel,path_id)
#         df_trainp_path["x"] = 0.5*df_trainp_path["x"] + 0.5*df_trainp_path1["x"]
#         df_trainp_path["y"] = 0.5*df_trainp_path["y"] + 0.5*df_trainp_path1["y"]
        df_trainp0.loc[(df_trainp0.index.isin(df_trainp_path.index)) & (df_trainp0.path_id.isin(df_trainp_path.path_id)),["x","y"]] = df_trainp_path.loc[:,["x","y"]]
        df_trainp0.loc[(df_trainp0["path_id"] == path_id),["floor"]] = df_trainp0.loc[(df_trainp0["path_id"] == path_id),["floor"]].mean()[0]
    df_trainp0 = df_trainp0[~df_trainp0.isnull().any(axis=1)]
    df_trainp0["x_diff"] = np.abs(df_trainp0["x"]-df_trainp0["waypoint_x"])
    df_trainp0["y_diff"] = np.abs(df_trainp0["y"]-df_trainp0["waypoint_y"])

    # interpolate test
    df_pred_all = pd.DataFrame({"path_id":df_test["path_id"],"timestamp":df_test["timestamp"],"x":pred_x,"y":pred_y})
    df_pred0 = pd.merge(df_sample_site.drop(["x","y"],axis=1),df_pred_all,how="outer",on=["path_id","timestamp"]).sort_values(['path_id','timestamp'])
    df_pred0.index = df_pred0["timestamp"]
    for path_id in df_pred0.path_id.unique():
#         df_pred_path = interp(df_pred0[(df_pred0["path_id"] == path_id)].copy(),path_id)
        df_pred_path = interp2(df_pred0[(df_pred0["path_id"] == path_id)].copy(),df_test_rel,path_id)
#         df_pred_path["x"] = 0.5*df_pred_path["x"] + 0.5*df_pred_path1["x"]
#         df_pred_path["y"] = 0.5*df_pred_path["y"] + 0.5*df_pred_path1["y"]
        df_pred0.loc[(df_pred0.index.isin(df_pred_path.index)) & (df_pred0.path_id.isin(df_pred_path.path_id)),["x","y"]] = df_pred_path.loc[:,["x","y"]]
        df_pred0.loc[(df_pred0["path_id"] == path_id),["floor"]] = df_pred0.loc[(df_pred0["path_id"] == path_id),["floor"]].mean()[0]
    df_pred = df_pred.append(df_pred0[["site_path_timestamp","x","y"]])

    # output train prediction
    df_out = pd.DataFrame({
        "floor": df_train.floor, "path_id": df_train.path_id,"timestamp": df_train.timestamp,
        "x": oof_x,"y": oof_y,
        "waypoint_x": df_train.waypoint_x, "waypoint_y": df_train.waypoint_y,
        "x_diff": np.abs(oof_x-df_train.waypoint_x), "y_diff": np.abs(oof_y-df_train.waypoint_y)
    })
#     df_out = interp3(df_out,df_train_rel)
    df_out_all = df_out_all.append(df_out)
    print(f"{site_id} nfloor:{df_train.floor.nunique()} npath:{df_train.path_id.nunique()} ntrain:{len(df_train)} ntest:{len(df_pred0.dropna())} %test:{100*len(df_pred0.dropna())/len(df_submit):.2f} xy:{np.sqrt((df_out.x-df_out.waypoint_x)**2 + (df_out.y-df_out.waypoint_y)**2).mean():.4f} CV:{np.sqrt((df_trainp0.waypoint_x-df_trainp0.x)**2 + (df_trainp0.waypoint_y-df_trainp0.y)**2).mean():.4f}")
    est_score += np.sqrt((df_out.x-df_out.waypoint_x)**2 + (df_out.y-df_out.waypoint_y)**2).mean()*len(df_pred0.dropna())/len(df_submit)
    EST += np.sqrt((df_trainp0.waypoint_x-df_trainp0.x)**2 + (df_trainp0.waypoint_y-df_trainp0.y)**2).mean()*len(df_pred0.dropna())/len(df_submit)
    df_out.to_csv(f"train/{site_id}.csv", index = False)
    df_trainp0.to_csv(f"trainp/{site_id}.csv", index = False)
    df_pred0.to_csv(f"test/{site_id}.csv",index = False)
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 10))
    df_train.plot.scatter(ax=axes[0],x="waypoint_x",y="waypoint_y",xlim=[df_train.waypoint_x.min(),df_train.waypoint_x.max()],ylim=[df_train.waypoint_y.min(),df_train.waypoint_y.max()])
    df_pred0.plot.scatter(ax=axes[1],x="x",y="y",xlim=[df_train.waypoint_x.min(),df_train.waypoint_x.max()],ylim=[df_train.waypoint_y.min(),df_train.waypoint_y.max()])
    
#     break

# score
print(np.sqrt((oof_x_all-x_true)**2 + (oof_y_all-y_true)**2).mean())
print(f"est_score:{est_score}, EST:{EST}")

# output
df_submit = df_submit.merge(df_pred, how = "left", on = "site_path_timestamp")
df_submit.to_csv(f"submission.csv", index = False)

plt.figure(figsize=(14, 13))
sns.barplot(x="importance", y="feature", data=df_fimp.sort_values(by="importance", ascending=False).head(100))
plt.title("LightGBM Feature Importance")
plt.tight_layout()
