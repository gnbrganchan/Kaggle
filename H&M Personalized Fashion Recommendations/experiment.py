import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import datetime
import itertools
import os
from contextlib import redirect_stdout
from tqdm.notebook import tqdm
#カラム内の文字数。デフォルトは50だった
pd.set_option("display.max_colwidth", 100)

#行数
pd.set_option("display.max_rows", 101)

#列数
pd.set_option("display.max_rows", 30)

rand = 64
lgb_params = {}
lgb_params["class"] = {
    "objective": "binary",
    "boosting": "gbdt",
    "max_depth": -1,
    "num_leaves": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "bagging_seed": rand,
    "learning_rate": 0.02,
    "feature_fraction": 0.6,
    "min_data_in_leaf": 100,
    "lambda_l1": 0,
    "lambda_l2": 0,
    "random_state": rand,
    "metric": "auc",#"binary_logloss",
    "verbose": -1
}

lgb_params["rank"] = {
    "objective": "lambdarank",
    "boosting": "gbdt",
    "max_depth": -1,
    "num_leaves": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "bagging_seed": rand,
    "learning_rate": 0.02,
    "feature_fraction": 0.6,
    "min_data_in_leaf": 100,
    "lambda_l1": 0,
    "lambda_l2": 0,
    "random_state": rand,
    "metric": "map",
    "eval_at": 12,
    "verbose": -1
}

path = "/content/drive/MyDrive/kaggle/HM/"

tran_dtypes = {"t_dat":"str",
               "customer_id":"str",
               "article_id":"int",
               "product_code":"int",
               "price":"float",
               "sales_channel_id":"int"}
art_dtypes = {"article_id":"int",
              "product_code":"int",
              "product_type_no":"int",
              "graphical_appearance_no":"int",
              "colour_group_code":"int",
              "department_no":"int",
              "index_code":"str",
              "index_group_no":"int",
              "section_no":"int",
              "garment_group_no":"int"}
cust_dtypes = {"customer_id":"str"}

obj = "class" # "class" or "rank"
N = 15000
N_div = 50
n_iter = 5
idx_file = "exp14"
len_hist = 366
n_round = 4000
n_splits = 1
tmp_top = 200
tr_set = [1,8,15,22]
len_tr = 7
nobuy = 30

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    
def read_data(day_oldest):
    df_art = pd.read_csv(path+"input/articles.csv",dtype=art_dtypes)
    le = LabelEncoder()
    le.fit(df_art["index_code"].unique())
    df_art["index_code"] = le.transform(df_art["index_code"])
#     display(df_art["index_code"].unique())

    df_cust = pd.read_csv(path+"input/customers.csv",dtype=cust_dtypes)
    df_cust["age"] = df_cust["age"].fillna(df_cust["age"].mean())
    df_cust[["FN","Active"]] = df_cust[["FN","Active"]].fillna(0)
    df_cust["club_member_status"] = df_cust["club_member_status"].apply(lambda x:1 if x == "ACTIVE" else 0)
    df_cust["fashion_news_frequency"] = df_cust["fashion_news_frequency"].apply(lambda x:0 if x == "NONE" else 1)

    df_trans = pd.read_csv(path+"input/transactions_train.csv",dtype=tran_dtypes)
    df_trans["t_dat"] = pd.to_datetime(df_trans["t_dat"],format="%Y-%m-%d")
    df_trans = df_trans.query(f"t_dat >= '{day_oldest}'").copy()
    df_trans = df_trans.drop_duplicates(["customer_id","article_id","t_dat"])
    df_trans = df_trans.merge(df_art[["article_id","product_code","product_type_no","graphical_appearance_no","colour_group_code","department_no","index_code","index_group_no","section_no","garment_group_no"]],how="left",on="article_id")
    df_trans = df_trans.merge(df_cust[["customer_id","age"]],how="left",on="customer_id")
    
    dict_vec = {}
    vec_art = np.load(path+"input/articles.npy")
    df_vec = pd.concat([df_art["article_id"],pd.DataFrame(vec_art)],axis=1)
    for i in range(len(vec_art)):
        dict_vec[df_art["article_id"][i]] = vec_art[i]
    del vec_art,df_vec

    df_art = df_art.set_index("article_id")
    df_cust = df_cust.set_index("customer_id")

    return df_trans,df_art,df_cust,dict_vec
    
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
df_trans = pd.read_csv(path+"input/transactions_train.csv",dtype=tran_dtypes)
df_trans["t_dat"] = pd.to_datetime(df_trans["t_dat"],format="%Y-%m-%d")
df_trans = df_trans.drop_duplicates(["customer_id","article_id","t_dat"])

day_end_valtmp = df_trans["t_dat"].max()
day_start_valtmp = day_end_valtmp - datetime.timedelta(days=6)  
df_trans_val_1 = df_trans.query("(t_dat >= @day_start_valtmp) and (t_dat <= @day_end_valtmp)").copy()
df_trans_val_1["article_id"] = df_trans_val_1["article_id"].astype(str).str.zfill(10)
df_agg_val_1 = df_trans_val_1.groupby("customer_id")["article_id"].apply(list).reset_index()
df_agg_val_1 = df_agg_val_1[df_agg_val_1["article_id"].apply(len) != 0]

del df_trans, df_trans_val_1
gc.collect()

def feat_store(df_trans,l_cust,ds,de,dsr,der,dsh,deh):
    feat ={}
    dsm = der - datetime.timedelta(days=30)
    df_trans_yesterday = df_trans.query("(t_dat == @der)")
    df_trans_recent = df_trans.query("(t_dat >= @dsr) and (t_dat <= @der)")
    df_trans_month = df_trans.query("(t_dat >= @dsm) and (t_dat <= @der)")
    df_trans_hist = df_trans.query("(t_dat >= @dsh) and (t_dat <= @deh)")
    
    # article

    feat["art_buy_hist"] = df_trans_hist.groupby(["article_id"])["t_dat"].agg(art_buy_hist="count")
    feat["art_buy_month"] = df_trans_month.groupby(["article_id"])["t_dat"].agg(art_buy_month="count")
    feat["art_buy_recent"] = df_trans_recent.groupby(["article_id"])["t_dat"].agg(art_buy_recent="count")
    feat["art_buy_recent_10"] = df_trans_recent.query("age < 20").groupby(["article_id"])["t_dat"].agg(art_buy_recent_10="count")
    feat["art_buy_recent_20"] = df_trans_recent.query("age >= 20 and age < 30").groupby(["article_id"])["t_dat"].agg(art_buy_recent_20="count")
    feat["art_buy_recent_30"] = df_trans_recent.query("age >= 30 and age < 40").groupby(["article_id"])["t_dat"].agg(art_buy_recent_30="count")
    feat["art_buy_recent_40"] = df_trans_recent.query("age >= 40 and age < 50").groupby(["article_id"])["t_dat"].agg(art_buy_recent_40="count")
    feat["art_buy_recent_50"] = df_trans_recent.query("age >= 50 and age < 60").groupby(["article_id"])["t_dat"].agg(art_buy_recent_50="count")
    feat["art_buy_recent_60"] = df_trans_recent.query("age >= 60").groupby(["article_id"])["t_dat"].agg(art_buy_recent_60="count")


    feat["art_buy_yesterday"] = df_trans_yesterday.groupby(["article_id"])["t_dat"].agg(art_buy_yesterday="count")
    df_buy1 = df_trans_hist.groupby("article_id")["customer_id"].nunique().reset_index().rename(columns={"customer_id":"cnt_buy1"})
    df_buy2 = df_trans_hist[df_trans_hist.duplicated(["customer_id","article_id"])].copy()
    df_buy2 = df_buy2.drop_duplicates(["customer_id","article_id"])
    df_buy2 = df_buy2.groupby("article_id")["article_id"].agg(cnt_buy2='count').reset_index()
    df_buy = pd.merge(df_buy1,df_buy2,how="left",on="article_id").fillna(0)
    df_buy["rebuy_rate"] = df_buy["cnt_buy2"]/df_buy["cnt_buy1"]
    feat["rebuy_rate"] = df_buy[["article_id","rebuy_rate"]]

    df_trans_yesterday = df_trans_yesterday.query("(customer_id in @l_cust)")
    df_trans_recent = df_trans_recent.query("(customer_id in @l_cust)")
    df_trans_hist = df_trans_hist.query("(customer_id in @l_cust)")
    
    # customer * article
    feat["rate_sales_channel_hist"] = df_trans_hist.groupby(["customer_id"])["sales_channel_id"].agg(rate_sales_channel_hist="mean")
    feat["rate_sales_channel_month"] = df_trans_month.groupby(["customer_id"])["sales_channel_id"].agg(rate_sales_channel_month="mean")
    feat["rate_sales_channel_recent"] = df_trans_recent.groupby(["customer_id"])["sales_channel_id"].agg(rate_sales_channel_recent="mean")
    feat["n_buy_hist"] = df_trans_hist.groupby(["customer_id","article_id"])["t_dat"].agg(n_buy_hist="count")
    feat["n_buy_month"] = df_trans_month.groupby(["customer_id","article_id"])["t_dat"].agg(n_buy_month="count")
    feat["n_buy_recent"] = df_trans_recent.groupby(["customer_id","article_id"])["t_dat"].agg(n_buy_recent="count")
    feat["n_buy_month_ch1"] = df_trans_month.query("sales_channel_id == 1").groupby(["customer_id","article_id"])["t_dat"].agg(n_buy_month_ch1="count")
    feat["n_buy_month_ch2"] = df_trans_month.query("sales_channel_id == 2").groupby(["customer_id","article_id"])["t_dat"].agg(n_buy_month_ch2="count")
    feat["n_buy_recent_ch1"] = df_trans_recent.query("sales_channel_id == 1").groupby(["customer_id","article_id"])["t_dat"].agg(n_buy_recent_ch1="count")
    feat["n_buy_recent_ch2"] = df_trans_recent.query("sales_channel_id == 2").groupby(["customer_id","article_id"])["t_dat"].agg(n_buy_recent_ch2="count")
    feat["days_after_buy"] = df_trans_hist.groupby(["customer_id","article_id"])["t_dat"].agg(days_after_buy=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_all"] = df_trans_hist.groupby(["customer_id"])["t_dat"].agg(n_buy_hist_all="count")
    feat["n_buy_month_all"] = df_trans_month.groupby(["customer_id"])["t_dat"].agg(n_buy_month_all="count")
    feat["n_buy_recent_all"] = df_trans_recent.groupby(["customer_id"])["t_dat"].agg(n_buy_recent_all="count")
    feat["days_after_buy_all"] = df_trans_hist.groupby(["customer_id"])["t_dat"].agg(days_after_buy_all=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_prod"] = df_trans_hist.groupby(["customer_id","product_code"])["t_dat"].agg(n_buy_hist_prod="count")
    feat["n_buy_month_prod"] = df_trans_month.groupby(["customer_id","product_code"])["t_dat"].agg(n_buy_month_prod="count")
    feat["n_buy_recent_prod"] = df_trans_recent.groupby(["customer_id","product_code"])["t_dat"].agg(n_buy_recent_prod="count")
    feat["days_after_buy_prod"] = df_trans_hist.groupby(["customer_id","product_code"])["t_dat"].agg(days_after_buy_prod=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_ptype"] = df_trans_hist.groupby(["customer_id","product_type_no"])["t_dat"].agg(n_buy_hist_ptype="count")
    feat["n_buy_month_ptype"] = df_trans_month.groupby(["customer_id","product_type_no"])["t_dat"].agg(n_buy_month_ptype="count")
    feat["n_buy_recent_ptype"] = df_trans_recent.groupby(["customer_id","product_type_no"])["t_dat"].agg(n_buy_recent_ptype="count")
    feat["days_after_buy_ptype"] = df_trans_hist.groupby(["customer_id","product_type_no"])["t_dat"].agg(days_after_buy_ptype=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_graph"] = df_trans_hist.groupby(["customer_id","graphical_appearance_no"])["t_dat"].agg(n_buy_hist_graph="count")
    feat["n_buy_month_graph"] = df_trans_month.groupby(["customer_id","graphical_appearance_no"])["t_dat"].agg(n_buy_month_graph="count")
    feat["n_buy_recent_graph"] = df_trans_recent.groupby(["customer_id","graphical_appearance_no"])["t_dat"].agg(n_buy_recent_graph="count")
    feat["days_after_buy_graph"] = df_trans_hist.groupby(["customer_id","graphical_appearance_no"])["t_dat"].agg(days_after_buy_graph=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_col"] = df_trans_hist.groupby(["customer_id","colour_group_code"])["t_dat"].agg(n_buy_hist_col="count")
    feat["n_buy_month_col"] = df_trans_month.groupby(["customer_id","colour_group_code"])["t_dat"].agg(n_buy_month_col="count")
    feat["n_buy_recent_col"] = df_trans_recent.groupby(["customer_id","colour_group_code"])["t_dat"].agg(n_buy_recent_col="count")
    feat["days_after_buy_col"] = df_trans_hist.groupby(["customer_id","colour_group_code"])["t_dat"].agg(days_after_buy_col=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_dep"] = df_trans_hist.groupby(["customer_id","department_no"])["t_dat"].agg(n_buy_hist_dep="count")
    feat["n_buy_month_dep"] = df_trans_month.groupby(["customer_id","department_no"])["t_dat"].agg(n_buy_month_dep="count")
    feat["n_buy_recent_dep"] = df_trans_recent.groupby(["customer_id","department_no"])["t_dat"].agg(n_buy_recent_dep="count")
    feat["days_after_buy_dep"] = df_trans_hist.groupby(["customer_id","department_no"])["t_dat"].agg(days_after_buy_dep=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_idx"] = df_trans_hist.groupby(["customer_id","index_code"])["t_dat"].agg(n_buy_hist_idx="count")
    feat["n_buy_month_idx"] = df_trans_month.groupby(["customer_id","index_code"])["t_dat"].agg(n_buy_month_idx="count")
    feat["n_buy_recent_idx"] = df_trans_recent.groupby(["customer_id","index_code"])["t_dat"].agg(n_buy_recent_idx="count")
    feat["days_after_buy_idx"] = df_trans_hist.groupby(["customer_id","index_code"])["t_dat"].agg(days_after_buy_idx=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_idxg"] = df_trans_hist.groupby(["customer_id","index_group_no"])["t_dat"].agg(n_buy_hist_idxg="count")
    feat["n_buy_month_idxg"] = df_trans_month.groupby(["customer_id","index_group_no"])["t_dat"].agg(n_buy_month_idxg="count")
    feat["n_buy_recent_idxg"] = df_trans_recent.groupby(["customer_id","index_group_no"])["t_dat"].agg(n_buy_recent_idxg="count")
    feat["days_after_buy_idxg"] = df_trans_hist.groupby(["customer_id","index_group_no"])["t_dat"].agg(days_after_buy_idxg=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_sec"] = df_trans_hist.groupby(["customer_id","section_no"])["t_dat"].agg(n_buy_hist_sec="count")
    feat["n_buy_month_sec"] = df_trans_month.groupby(["customer_id","section_no"])["t_dat"].agg(n_buy_month_sec="count")
    feat["n_buy_recent_sec"] = df_trans_recent.groupby(["customer_id","section_no"])["t_dat"].agg(n_buy_recent_sec="count")
    feat["days_after_buy_sec"] = df_trans_hist.groupby(["customer_id","section_no"])["t_dat"].agg(days_after_buy_sec=lambda x:(ds - max(x)).days)
    feat["n_buy_hist_garm"] = df_trans_hist.groupby(["customer_id","garment_group_no"])["t_dat"].agg(n_buy_hist_garm="count")
    feat["n_buy_month_garm"] = df_trans_month.groupby(["customer_id","garment_group_no"])["t_dat"].agg(n_buy_month_garm="count")
    feat["n_buy_recent_garm"] = df_trans_recent.groupby(["customer_id","garment_group_no"])["t_dat"].agg(n_buy_recent_garm="count")
    feat["days_after_buy_garm"] = df_trans_hist.groupby(["customer_id","garment_group_no"])["t_dat"].agg(days_after_buy_garm=lambda x:(ds - max(x)).days)
    feat["art_id_recent"] = df_trans_recent.groupby("customer_id")["article_id"].apply(list).rename("art_id_recent")
    feat["art_id_month"] = df_trans_month.groupby("customer_id")["article_id"].apply(list).rename("art_id_month")
    feat["art_id_yesterday"] = df_trans_yesterday.groupby("customer_id")["article_id"].apply(list).rename("art_id_yesterday")
    feat["n_buy_hist_index_0"] = df_trans_hist.query("index_code == 0").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_0="count")
    feat["n_buy_recent_index_0"] = df_trans_recent.query("index_code == 0").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_0="count")
    feat["n_buy_hist_index_1"] = df_trans_hist.query("index_code == 1").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_1="count")
    feat["n_buy_recent_index_1"] = df_trans_recent.query("index_code == 1").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_1="count")
    feat["n_buy_hist_index_2"] = df_trans_hist.query("index_code == 2").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_2="count")
    feat["n_buy_recent_index_2"] = df_trans_recent.query("index_code == 2").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_2="count")
    feat["n_buy_hist_index_3"] = df_trans_hist.query("index_code == 3").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_3="count")
    feat["n_buy_recent_index_3"] = df_trans_recent.query("index_code == 3").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_3="count")
    feat["n_buy_hist_index_4"] = df_trans_hist.query("index_code == 4").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_4="count")
    feat["n_buy_recent_index_4"] = df_trans_recent.query("index_code == 4").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_4="count")
    feat["n_buy_hist_index_5"] = df_trans_hist.query("index_code == 5").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_5="count")
    feat["n_buy_recent_index_5"] = df_trans_recent.query("index_code == 5").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_5="count")
    feat["n_buy_hist_index_6"] = df_trans_hist.query("index_code == 6").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_6="count")
    feat["n_buy_recent_index_6"] = df_trans_recent.query("index_code == 6").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_6="count")
    feat["n_buy_hist_index_7"] = df_trans_hist.query("index_code == 7").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_7="count")
    feat["n_buy_recent_index_7"] = df_trans_recent.query("index_code == 7").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_7="count")
    feat["n_buy_hist_index_8"] = df_trans_hist.query("index_code == 8").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_8="count")
    feat["n_buy_recent_index_8"] = df_trans_recent.query("index_code == 8").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_8="count")
    feat["n_buy_hist_index_9"] = df_trans_hist.query("index_code == 9").groupby(["customer_id"])["t_dat"].agg(n_buy_hist_index_9="count")
    feat["n_buy_recent_index_9"] = df_trans_recent.query("index_code == 9").groupby(["customer_id"])["t_dat"].agg(n_buy_recent_index_9="count")


    del df_trans_yesterday, df_trans_recent, df_trans_month, df_trans_hist, df_buy1, df_buy2, df_buy
    gc.collect()

    return feat
    
def add_feat(df,ds,de,dsr,der,dsh,deh,feat,dict_vec):
    # rate_sales_channel
    df = df.merge(feat["rate_sales_channel_hist"],how="left",left_on=["customer_id"], right_index=True)
    df = df.merge(feat["rate_sales_channel_month"],how="left",left_on=["customer_id"], right_index=True)
    df = df.merge(feat["rate_sales_channel_recent"],how="left",left_on=["customer_id"], right_index=True)  
    # art_buy
    df = df.merge(feat["art_buy_hist"],how="left",left_on=["article_id"], right_index=True)
    df = df.merge(feat["art_buy_month"],how="left",left_on=["article_id"], right_index=True)
    df = df.merge(feat["art_buy_recent"],how="left",left_on=["article_id"], right_index=True) 
    # art_buy_age
    for age in [10,20,30,40,50,60]:
        df = df.merge(feat[f"art_buy_recent_{age}"],how="left",left_on=["article_id"], right_index=True) 
    # art_buy_yesterday
    df = df.merge(feat["art_buy_yesterday"],how="left",left_on=["article_id"], right_index=True)  
    # n_buy
    df = df.merge(feat["n_buy_hist"],how="left",left_on=["customer_id","article_id"], right_index=True)
    df = df.merge(feat["n_buy_month"],how="left",left_on=["customer_id","article_id"], right_index=True)
    df = df.merge(feat["n_buy_month_ch1"],how="left",left_on=["customer_id","article_id"], right_index=True)
    df = df.merge(feat["n_buy_month_ch2"],how="left",left_on=["customer_id","article_id"], right_index=True)
    df = df.merge(feat["n_buy_recent"],how="left",left_on=["customer_id","article_id"], right_index=True)
    df = df.merge(feat["n_buy_recent_ch1"],how="left",left_on=["customer_id","article_id"], right_index=True)
    df = df.merge(feat["n_buy_recent_ch2"],how="left",left_on=["customer_id","article_id"], right_index=True)
    # days_after_buy
    df = df.merge(feat["days_after_buy"],how="left",left_on=["customer_id","article_id"], right_index=True)
    # n_buy_all
    df = df.merge(feat["n_buy_hist_all"],how="left",left_on=["customer_id"], right_index=True)
    df = df.merge(feat["n_buy_month_all"],how="left",left_on=["customer_id"], right_index=True)
    df = df.merge(feat["n_buy_recent_all"],how="left",left_on=["customer_id"], right_index=True)
    # days_after_buy_all
    df = df.merge(feat["days_after_buy_all"],how="left",left_on=["customer_id"], right_index=True)
    # n_buy_prod
    df = df.merge(feat["n_buy_hist_prod"],how="left",left_on=["customer_id","product_code"], right_index=True)
    df = df.merge(feat["n_buy_month_prod"],how="left",left_on=["customer_id","product_code"], right_index=True)
    df = df.merge(feat["n_buy_recent_prod"],how="left",left_on=["customer_id","product_code"], right_index=True)
    # days_after_buy_prod
    df = df.merge(feat["days_after_buy_prod"],how="left",left_on=["customer_id","product_code"], right_index=True)
    # n_buy_ptype
    df = df.merge(feat["n_buy_hist_ptype"],how="left",left_on=["customer_id","product_type_no"], right_index=True)
    df = df.merge(feat["n_buy_month_ptype"],how="left",left_on=["customer_id","product_type_no"], right_index=True)
    df = df.merge(feat["n_buy_recent_ptype"],how="left",left_on=["customer_id","product_type_no"], right_index=True)
    # days_after_buy_ptype
    df = df.merge(feat["days_after_buy_ptype"],how="left",left_on=["customer_id","product_type_no"], right_index=True)
    # n_buy_graph
    df = df.merge(feat["n_buy_hist_graph"],how="left",left_on=["customer_id","graphical_appearance_no"], right_index=True)
    df = df.merge(feat["n_buy_month_graph"],how="left",left_on=["customer_id","graphical_appearance_no"], right_index=True)
    df = df.merge(feat["n_buy_recent_graph"],how="left",left_on=["customer_id","graphical_appearance_no"], right_index=True)
    # days_after_buy_graph
    df = df.merge(feat["days_after_buy_graph"],how="left",left_on=["customer_id","graphical_appearance_no"], right_index=True)
    # n_buy_col
    df = df.merge(feat["n_buy_hist_col"],how="left",left_on=["customer_id","colour_group_code"], right_index=True)
    df = df.merge(feat["n_buy_month_col"],how="left",left_on=["customer_id","colour_group_code"], right_index=True)
    df = df.merge(feat["n_buy_recent_col"],how="left",left_on=["customer_id","colour_group_code"], right_index=True)
    # days_after_buy_col
    df = df.merge(feat["days_after_buy_col"],how="left",left_on=["customer_id","colour_group_code"], right_index=True)
    # n_buy_dep
    df = df.merge(feat["n_buy_hist_dep"],how="left",left_on=["customer_id","department_no"], right_index=True)
    df = df.merge(feat["n_buy_month_dep"],how="left",left_on=["customer_id","department_no"], right_index=True)
    df = df.merge(feat["n_buy_recent_dep"],how="left",left_on=["customer_id","department_no"], right_index=True)
    # days_after_buy_dep
    df = df.merge(feat["days_after_buy_dep"],how="left",left_on=["customer_id","department_no"], right_index=True)
    # n_buy_idx
    df = df.merge(feat["n_buy_hist_idx"],how="left",left_on=["customer_id","index_code"], right_index=True)
    df = df.merge(feat["n_buy_month_idx"],how="left",left_on=["customer_id","index_code"], right_index=True)
    df = df.merge(feat["n_buy_recent_idx"],how="left",left_on=["customer_id","index_code"], right_index=True)
    # days_after_buy_idx
    df = df.merge(feat["days_after_buy_idx"],how="left",left_on=["customer_id","index_code"], right_index=True)
    # n_buy_idxg
    df = df.merge(feat["n_buy_hist_idxg"],how="left",left_on=["customer_id","index_group_no"], right_index=True)
    df = df.merge(feat["n_buy_month_idxg"],how="left",left_on=["customer_id","index_group_no"], right_index=True)
    df = df.merge(feat["n_buy_recent_idxg"],how="left",left_on=["customer_id","index_group_no"], right_index=True)
    # days_after_buy_idxg
    df = df.merge(feat["days_after_buy_idxg"],how="left",left_on=["customer_id","index_group_no"], right_index=True)
    # n_buy_sec
    df = df.merge(feat["n_buy_hist_sec"],how="left",left_on=["customer_id","section_no"], right_index=True)
    df = df.merge(feat["n_buy_month_sec"],how="left",left_on=["customer_id","section_no"], right_index=True)
    df = df.merge(feat["n_buy_recent_sec"],how="left",left_on=["customer_id","section_no"], right_index=True)
    # days_after_buy_sec
    df = df.merge(feat["days_after_buy_sec"],how="left",left_on=["customer_id","section_no"], right_index=True)
    # n_buy_garm
    df = df.merge(feat["n_buy_hist_garm"],how="left",left_on=["customer_id","garment_group_no"], right_index=True)
    df = df.merge(feat["n_buy_month_garm"],how="left",left_on=["customer_id","garment_group_no"], right_index=True)
    df = df.merge(feat["n_buy_recent_garm"],how="left",left_on=["customer_id","garment_group_no"], right_index=True)
    # days_after_buy_garm
    df = df.merge(feat["days_after_buy_garm"],how="left",left_on=["customer_id","garment_group_no"], right_index=True)
    for i in range(10):
        # n_buy_hist_idx
        df = df.merge(feat[f"n_buy_hist_index_{i}"],how="left",left_on=["customer_id"], right_index=True)
        # n_buy_recent_idx
        df = df.merge(feat[f"n_buy_recent_index_{i}"],how="left",left_on=["customer_id"], right_index=True)
    # rebuy_rate
    df = df.merge(feat["rebuy_rate"],how="left",on="article_id")
    # sim_article_recent
    df = df.merge(feat["art_id_recent"],how="left",left_on="customer_id", right_index = True)
    sim_max,sim_sum,sim_mean = [],[],[]
    tmp = df[["article_id","art_id_recent"]].values
    for i in range(len(df)):
      if not isinstance(tmp[i][1],list):
        sim_max.append(0);sim_sum.append(0);sim_mean.append(0)
      else:
        list_sim = [cos_sim(dict_vec[tmp[i][0]],dict_vec[x]) for x in tmp[i][1]]
        sim_max.append(max(list_sim))
        sim_sum.append(sum(list_sim))
        sim_mean.append(np.mean(list_sim))
    df["sim_max_recent"] = sim_max
    df["sim_sum_recent"] = sim_sum
    df["sim_mean_recent"] = sim_mean
    df = df.drop(["art_id_recent"], axis = 1)
    # sim_article_month
    df = df.merge(feat["art_id_month"],how="left",left_on="customer_id", right_index = True)
    sim_max,sim_sum,sim_mean = [],[],[]
    tmp = df[["article_id","art_id_month"]].values
    for i in range(len(df)):
      if not isinstance(tmp[i][1],list):
        sim_max.append(0);sim_sum.append(0);sim_mean.append(0)
      else:
        list_sim = [cos_sim(dict_vec[tmp[i][0]],dict_vec[x]) for x in tmp[i][1]]
        sim_max.append(max(list_sim))
        sim_sum.append(sum(list_sim))
        sim_mean.append(np.mean(list_sim))
    df["sim_max_month"] = sim_max
    df["sim_sum_month"] = sim_sum
    df["sim_mean_month"] = sim_mean
    df = df.drop(["art_id_month"], axis = 1)

    # 欠損値埋め
    cols = ["n_buy_hist","n_buy_month","n_buy_recent",
            "n_buy_month_ch1","n_buy_month_ch2","n_buy_recent_ch1","n_buy_recent_ch2",
            "n_buy_hist_all","n_buy_month_all","n_buy_recent_all",
            "n_buy_hist_prod","n_buy_month_prod","n_buy_recent_prod",
            "n_buy_hist_ptype","n_buy_month_ptype","n_buy_recent_ptype",
            "n_buy_hist_graph","n_buy_month_graph","n_buy_recent_graph",
            "n_buy_hist_col","n_buy_month_col","n_buy_recent_col",
            "n_buy_hist_dep","n_buy_month_dep","n_buy_recent_dep",
            "n_buy_hist_idx","n_buy_month_idx","n_buy_recent_idx",
            "n_buy_hist_idxg","n_buy_month_idxg","n_buy_recent_idxg",
            "n_buy_hist_sec","n_buy_month_sec","n_buy_recent_sec",
            "n_buy_hist_garm","n_buy_month_garm","n_buy_recent_garm",
            "art_buy_hist","art_buy_month","art_buy_recent","art_buy_yesterday",
            "rebuy_rate", "sim_max_month", "sim_sum_month", "sim_mean_month", "sim_max_recent", "sim_sum_recent", "sim_mean_recent",
            "art_buy_recent_10","art_buy_recent_20","art_buy_recent_30","art_buy_recent_40","art_buy_recent_50","art_buy_recent_60",
            "n_buy_hist_index_0","n_buy_hist_index_1","n_buy_hist_index_2","n_buy_hist_index_3","n_buy_hist_index_4","n_buy_hist_index_5","n_buy_hist_index_6","n_buy_hist_index_7","n_buy_hist_index_8","n_buy_hist_index_9",
            "n_buy_recent_index_0","n_buy_recent_index_1","n_buy_recent_index_2","n_buy_recent_index_3","n_buy_recent_index_4","n_buy_recent_index_5","n_buy_recent_index_6","n_buy_recent_index_7","n_buy_recent_index_8","n_buy_recent_index_9"]
    df[cols] = df[cols].fillna(0)

    df[["days_after_buy","days_after_buy_all","days_after_buy_prod","days_after_buy_ptype","days_after_buy_graph","days_after_buy_col","days_after_buy_dep","days_after_buy_idx",
      "days_after_buy_idxg","days_after_buy_sec","days_after_buy_garm"]] = \
    df[["days_after_buy","days_after_buy_all","days_after_buy_prod","days_after_buy_ptype","days_after_buy_graph","days_after_buy_col","days_after_buy_dep","days_after_buy_idx",
      "days_after_buy_idxg","days_after_buy_sec","days_after_buy_garm"]].fillna(10+len_hist)

    df[["rate_sales_channel_hist","rate_sales_channel_month","rate_sales_channel_recent"]] = df[["rate_sales_channel_hist","rate_sales_channel_month","rate_sales_channel_recent"]].fillna(1.5)

    return df
    
def recommend_train(day_start_val):
    day_start = [day_start_val - datetime.timedelta(days=i-1+len_tr) for i in tr_set]
    day_end = [day_start_val - datetime.timedelta(days=i) for i in tr_set]
    day_start_rec = [x - datetime.timedelta(days=7) for x in day_start]
    day_end_rec = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_hist = [x - datetime.timedelta(days=len_hist) for x in day_start]
    day_end_hist = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_rec_test = day_start_val - datetime.timedelta(days=7)
    day_end_rec_test = day_start_val - datetime.timedelta(days=1)
    day_start_hist_test = day_start_val - datetime.timedelta(days=1+len_hist)
    day_end_hist_test = day_start_val - datetime.timedelta(days=1)
    day_end_val = day_start_val + datetime.timedelta(days=6)

    df_trans, df_art, df_cust, dict_vec = read_data(day_oldest = day_start_hist[-1])

    q_date = ""
    for i in range(len(day_start)):
      if i == 0: q_date = f"((t_dat >= '{day_start[0]}') and (t_dat <= '{day_end[0]}'))"
      else: q_date = q_date + f" or ((t_dat >= '{day_start[i]}') and (t_dat <= '{day_end[i]}'))"
    top_art_all = df_trans.query(q_date).groupby("article_id")["t_dat"].count().sort_values(ascending = False).index[:N].tolist()

    list_df_buy = []
    list_list_cust =[]
    for i in range(len(day_start)):
      list_df_buy.append(df_trans.query(f"(t_dat >= '{day_start[i]}') and (t_dat <= '{day_end[i]}') and (article_id in @top_art_all)").drop_duplicates(["customer_id","article_id"])[["customer_id","article_id"]].copy())
      list_df_buy[i]["target"] = 1
      list_list_cust.append(list_df_buy[i]["customer_id"].unique().tolist()) 
    for iter_train in tqdm(range(n_iter)):
      list_df_nobuy = []
      list_train =[]
      for i in range(len(day_start)):
        list_df_nobuy.append(pd.concat([pd.DataFrame({"customer_id":x,"article_id":random.sample(top_art_all,nobuy)}) for x in list_list_cust[i]]))
        list_df_nobuy[i]["target"] = 0
        list_train.append(pd.concat([list_df_buy[i],list_df_nobuy[i]]).drop_duplicates(["customer_id","article_id"]))
      del list_df_nobuy
      display(list_train[0]["target"].value_counts())

      df_train = pd.DataFrame()
      for i in tqdm(range(len(day_start))):
        feat = feat_store(df_trans,list_list_cust[i],day_start[i],day_end[i],day_start_rec[i],day_end_rec[i],day_start_hist[i],day_end_hist[i])
        list_train[i] = list_train[i].merge(df_art[["product_code","product_type_no","graphical_appearance_no","colour_group_code","department_no","index_code","index_group_no","section_no","garment_group_no"]],how="left",left_on="article_id",right_index=True)
        list_train[i] = list_train[i].merge(df_cust[["age","FN","Active","club_member_status","fashion_news_frequency"]],how="left",left_on="customer_id",right_index=True)
        list_train[i]["week"] = i
        df_train = df_train.append(add_feat(list_train[i],day_start[i],day_end[i],day_start_rec[i],day_end_rec[i],day_start_hist[i],day_end_hist[i],feat,dict_vec))
        del feat
      del list_train
      gc.collect()

      df_train = df_train.sort_values(["customer_id","week"]).reset_index(drop = True)

      X_train = df_train.drop(["product_code","product_type_no","department_no","target"],axis=1)
      y_train = df_train["target"]
      del df_train

      list_model = []
      if n_splits == 0:
        tr_g = X_train.groupby(["customer_id","week"])["article_id"].count().values
        X_train = X_train.drop(["customer_id","week"], axis = 1)
        if obj == "rank":
          d_tr = lgb.Dataset(X_train, label=y_train,  free_raw_data=False, group = tr_g)
        else:
          d_tr = lgb.Dataset(X_train, label=y_train,  free_raw_data=False, group = tr_g)
        list_model.append(lgb.train(lgb_params[obj], train_set=d_tr, num_boost_round=n_round, valid_sets=[d_tr], verbose_eval=500))
        del  d_tr
      elif n_splits == 1:
        tr_idx, va_idx = next(GroupShuffleSplit().split(X_train, y_train, X_train["customer_id"]))
        X_tr, X_va, y_tr, y_va = X_train.iloc[tr_idx], X_train.iloc[va_idx], y_train.iloc[tr_idx], y_train.iloc[va_idx]
        if obj == "rank":
          tr_g = X_tr.groupby(["customer_id","week"])["article_id"].count().values
          va_g = X_va.groupby(["customer_id","week"])["article_id"].count().values
        X_tr = X_tr.drop(["customer_id","week"], axis = 1)
        X_va = X_va.drop(["customer_id","week"], axis = 1)
        if obj == "rank":
          d_tr = lgb.Dataset(X_tr, label=y_tr,  free_raw_data=False, group = tr_g)
          d_va = lgb.Dataset(X_va, label=y_va,  free_raw_data=False, group = va_g)
        else:
          d_tr = lgb.Dataset(X_tr, label=y_tr,  free_raw_data=False)
          d_va = lgb.Dataset(X_va, label=y_va,  free_raw_data=False)          
        list_model.append(lgb.train(lgb_params[obj], train_set=d_tr, num_boost_round=n_round, valid_sets=[d_tr,d_va], verbose_eval=500, early_stopping_rounds=100))
        del X_tr, X_va, y_tr, y_va, d_tr, d_va
      else:
        folds = GroupKFold(n_splits = n_splits)
        for tr_idx,va_idx in folds.split(X_train,y_train,X_train["customer_id"]):
          X_tr, X_va, y_tr, y_va = X_train.iloc[tr_idx], X_train.iloc[va_idx], y_train.iloc[tr_idx], y_train.iloc[va_idx] 
          if obj == "rank":
            tr_g = X_tr.groupby(["customer_id","week"])["article_id"].count().values
            va_g = X_va.groupby(["customer_id","week"])["article_id"].count().values
          X_tr = X_tr.drop(["customer_id","week"], axis = 1)
          X_va = X_va.drop(["customer_id","week"], axis = 1)
          if obj == "rank":
            d_tr = lgb.Dataset(X_tr, label=y_tr,  free_raw_data=False, group = tr_g)
            d_va = lgb.Dataset(X_va, label=y_va,  free_raw_data=False, group = va_g)
          else:
            d_tr = lgb.Dataset(X_tr, label=y_tr,  free_raw_data=False)
            d_va = lgb.Dataset(X_va, label=y_va,  free_raw_data=False)            
          list_model.append(lgb.train(lgb_params[obj], train_set=d_tr, num_boost_round=n_round, valid_sets=[d_tr,d_va], verbose_eval=500, early_stopping_rounds=100))
          del X_tr, X_va, y_tr, y_va, d_tr, d_va
      
      os.makedirs(path+f"pkl/{iter_train}/", exist_ok=True)
      pd.to_pickle(list_model,path+f"pkl/{iter_train}/models_{idx_file}_{day_start_val.date()}.pkl")
      del X_train, y_train
      gc.collect()
    del df_trans, df_art, df_cust
    gc.collect()
    return 0
    
def recommend_pred(series_cust,day_start_val,submit=False,sub_no = 0):
    day_start = [day_start_val - datetime.timedelta(days=i-1+len_tr) for i in tr_set]
    day_end = [day_start_val - datetime.timedelta(days=i) for i in tr_set]
    day_start_rec = [x - datetime.timedelta(days=7) for x in day_start]
    day_end_rec = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_hist = [x - datetime.timedelta(days=len_hist) for x in day_start]
    day_end_hist = [x - datetime.timedelta(days=1) for x in day_start]
    day_start_rec_test = day_start_val - datetime.timedelta(days=7)
    day_end_rec_test = day_start_val - datetime.timedelta(days=1)
    day_start_hist_test = day_start_val - datetime.timedelta(days=1+len_hist)
    day_end_hist_test = day_start_val - datetime.timedelta(days=1)

    day_end_val = day_start_val + datetime.timedelta(days=6)

    df_trans, df_art, df_cust,  dict_vec = read_data(day_oldest = day_start_hist[-1])
    
    q_date = ""
    for i in range(len(day_start)):
      if i == 0: q_date = f"((t_dat >= '{day_start[0]}') and (t_dat <= '{day_end[0]}'))"
      else: q_date = q_date + f" or ((t_dat >= '{day_start[i]}') and (t_dat <= '{day_end[i]}'))"
    top_art_all = df_trans.query(q_date).groupby("article_id")["t_dat"].count().sort_values(ascending = False).index[:N].tolist()

    list_sl = list(range(0,N,N_div))
    if list_sl[-1] != N:list_sl.append(N)
    df_ans = pd.DataFrame()

    feat = feat_store(df_trans,series_cust.tolist(),day_start_val,day_end_val,day_start_rec_test,day_end_rec_test,day_start_hist_test,day_end_hist_test)
    del df_trans

    for iter_art in tqdm(range(len(list_sl)-1)):
      top_art = top_art_all[list_sl[iter_art]:list_sl[iter_art+1]]
      df_test = cudf.from_pandas(pd.DataFrame(itertools.product(series_cust.tolist(),top_art),columns=["customer_id","article_id"]))
      df_test = df_test.merge(df_art[["product_code","product_type_no","graphical_appearance_no","colour_group_code","department_no","index_code","index_group_no","section_no","garment_group_no"]],how="left",left_on="article_id",right_index=True)
      df_test = df_test.merge(df_cust[["age","FN","Active","club_member_status","fashion_news_frequency"]],how="left",left_on="customer_id",right_index=True)

      df_test = add_feat(df_test,day_start_val,day_end_val,day_start_rec_test,day_end_rec_test,day_start_hist_test,day_end_hist_test,feat,dict_vec)

      df_pred = df_test[["customer_id","article_id"]].copy()
      df_test = df_test.drop(["customer_id","product_code","product_type_no","department_no"],axis=1)
      pred = np.zeros(len(df_pred))
      for iter_train in range(n_iter):
        list_model = pd.read_pickle(path+f"pkl/{iter_train}/models_{idx_file}_{day_start_val.date()}.pkl")
        for i in range(max(1,n_splits)):
          list_model[i].save_model(path+f"lgbm_{idx_file}.model")
          if obj == "rank":
            with redirect_stdout(open(os.devnull, 'w')):
              fm = ForestInference.load(filename=path+f"lgbm_{idx_file}.model",model_type='lightgbm')
            pred += fm.predict(df_test) / (max(1,n_splits) * n_iter)
          else:
            with redirect_stdout(open(os.devnull, 'w')):
              fm = ForestInference.load(filename=path+f"lgbm_{idx_file}.model",output_class=True,model_type='lightgbm')
            pred += fm.predict_proba(df_test)[:,1] / (max(1,n_splits) * n_iter)      
      df_pred["pred"] = pred

      df_ans = df_ans.append(df_pred)
      df_ans = df_ans.sort_values(["customer_id","pred"],ascending = False)
      df_ans = df_ans.groupby("customer_id").head(tmp_top)
      del list_model,df_test,df_pred,pred
      gc.collect()
        
    df_ans = df_ans.groupby(["customer_id","article_id"])["pred"].sum().reset_index().sort_values(["customer_id","pred"],ascending = False)
    df_ans["pred"] = df_ans["pred"]/n_iter
    df_ans["article_id"] = df_ans["article_id"].astype(str).str.zfill(10)
    if submit: df_ans.groupby("customer_id").head(tmp_top).to_csv(path+f"output/div{idx_file}/oof_{idx_file}_{day_start_val.date()}_{sub_no}.csv",index = False)
    else: df_ans.groupby("customer_id").head(tmp_top).to_csv(path+f"output/oof_{idx_file}_{day_start_val.date()}.csv",index = False)
    df_ans = df_ans.groupby("customer_id").head(12)
    mapk_val = mapk(df_agg_val_1["article_id"].tolist(),df_ans.groupby("customer_id")["article_id"].apply(list).tolist())
    print(f"mapk:{mapk_val} ")
    df_ans = df_ans.groupby("customer_id")["article_id"].apply(list).reset_index()
    df_ans = df_ans.rename({'article_id':'pred'},axis=1)
    gc.collect()
    return df_ans
    
recommend_train(day_start_valtmp)

recommend_pred(df_agg_val_1["customer_id"],day_start_valtmp)

df_sub = pd.read_csv(path+"input/sample_submission.csv")

recommend_train(day_end_valtmp+datetime.timedelta(days=1))

import time
size_block = 30000
list_slice = list(range(0,len(df_sub),size_block))
if list_slice[-1] != len(df_sub):list_slice.append(len(df_sub))
os.makedirs(path+f"output/div{idx_file}",exist_ok = True)
for i in tqdm(reversed(range(len(list_slice)-1))):
  time.sleep(1)
  if not os.path.exists(path+f"output/div{idx_file}/submission_{i}.csv"):
    df_sub_0 = df_sub[list_slice[i]:list_slice[i+1]].copy()
    df_ans = recommend_pred(df_sub_0["customer_id"],day_end_valtmp+datetime.timedelta(days=1),submit=True,sub_no = i)
    df_ans["prediction"] = df_ans["pred"].apply(lambda x:' '.join(x))
    df_ans[["customer_id","prediction"]].to_csv(path+f"output/div{idx_file}/submission_{i}.csv",index = False)
    del df_sub_0,df_ans
    gc.collect()
