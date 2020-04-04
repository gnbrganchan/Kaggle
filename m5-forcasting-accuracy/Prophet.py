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
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from tqdm import tqdm
import logging

logging.disable(logging.FATAL)

plt.style.use("seaborn")
sns.set(font_scale=1)

df = pd.read_csv("/content/drive/My Drive/input/m5-forecasting-accuracy/sales_train_validation.csv")
price = pd.read_csv("/content/drive/My Drive/input/m5-forecasting-accuracy/sell_prices.csv")
calendar = pd.read_csv("/content/drive/My Drive/input/m5-forecasting-accuracy/calendar.csv")
submission = pd.read_csv("/content/drive/My Drive/input/m5-forecasting-accuracy/sample_submission.csv")

# price.drop([""], axis = 1, inplace = True)
calendar.drop(["weekday", "wday", "month", "year", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"], axis = 1, inplace = True)

WRMSSE = 0.0
if os.path.exists("/content/drive/My Drive/sub.csv"):
    print("Read existing file.")
    sub = pd.read_csv("/content/drive/My Drive/sub.csv")
    res = pd.read_csv("/content/drive/My Drive/res.csv")
    for i in range(len(res)):
        WRMSSE += res.weight[i] * res.rmsse[i]
else:
    print("Make a new file.")
    sub = pd.DataFrame()
    res = pd.DataFrame()
    ids = df.id.unique()

df = pd.melt(df,id_vars = df.columns[df.columns.str.endswith("id")],value_vars = df.columns[df.columns.str.startswith("d_")])
df = df.rename(columns = {"value" : "sales"})

df = df.merge(calendar, left_on  = "variable", right_on = "d",how = "left")
del calendar
gc.collect()
# print("a")
df = df.merge(price,on = ["store_id", "item_id", "wm_yr_wk"], how = "left")
del price
gc.collect()
# print("b")
df = df.dropna(subset = ["sell_price"])
df['date'] = pd.to_datetime(df['date'])

gc.collect()

df = df.set_index("id")

def RMSSE(pred, act, train):
    one_day_ago = train.loc[:,["date","sales"]].copy()
    one_day_ago["date"] = one_day_ago["date"]+datetime.timedelta(days=1)
    denom = train.loc[:,["date","sales"]].merge(one_day_ago, on = "date", how = "inner")
    denom = (denom["sales_x"] - denom["sales_y"]) * (denom["sales_x"] - denom["sales_y"])
    denom = denom.mean()
    pred = pred.merge(act, left_on = "ds", right_on = "date", how = "inner")
    mole = (pred["yhat"] - act["sales"].reset_index()["sales"])* (pred["yhat"] - act["sales"].reset_index()["sales"])
    mole = mole.mean()
    return mole/denom

d_val = ["d_1913","d_1912","d_1911","d_1910","d_1909","d_1908","d_1907",
        "d_1906", "d_1905", "d_1904", "d_1903", "d_1902", "d_1901", "d_1900",
        "d_1899", "d_1898", "d_1897", "d_1896", "d_1895", "d_1894", "d_1893",
        "d_1892", "d_1891", "d_1890", "d_1889", "d_1888", "d_1887", "d_1886"]

df_val = df.loc[df.variable.isin(d_val)]
w = df_val.groupby(["id"])["sales"].sum()
tot = w.sum()
w = w/tot

if os.path.exists("/content/drive/My Drive/sub.csv"):
    df = df[~df.index.isin(sub.id)]
    ids = df.index.unique()

l = ["F" + str(i+1) for i in range(28)]
l.insert(0,"id")
cols = ["F" + str(i+1) for i in range(28)]

# print("c")

for store_item_id in tqdm(ids):
    df_store_item = df.loc[store_item_id]
    df_train = df_store_item.iloc[1:len(df_store_item)-28]
    df_val = df_store_item.iloc[len(df_store_item)-28:len(df_store_item)]
    model = Prophet()
    model.fit(df_train.loc[:,["date","sales"]].rename(columns = {"date" : "ds", "sales" : "y"}))
    future = model.make_future_dataframe(28)
    forecast = model.predict(future)
    forecast.yhat = np.clip(forecast.yhat,0,None)
    
    res = res.append(pd.DataFrame(data = {"id" : store_item_id,
                                         "weight" : w.loc[store_item_id],
                                         "rmsse" : RMSSE(forecast,df_val,df_train)},index=['i',]))
    WRMSSE += w.loc[store_item_id] * RMSSE(forecast,df_val,df_train)
#     print(i,w.loc[store_item_id],RMSSE(forecast,df_val,df_train))

    model = Prophet()
    model.fit(df_store_item.loc[:,["date","sales"]].rename(columns = {"date" : "ds", "sales" : "y"}))
    future = model.make_future_dataframe(28)
    forecast = model.predict(future)
    forecast.yhat = np.clip(forecast.yhat,0,None)
    
    sub_id = pd.DataFrame(data = {"pred" : forecast.tail(28).yhat,
                           "col" : cols})
    sub_id = sub_id.pivot_table(values=['pred'],columns=['col'], aggfunc='sum').reset_index()
    sub_id["id"] = store_item_id
    sub_id.drop(["index"], axis = 1, inplace = True)
    sub = sub.append(sub_id[l])
    
    sub.to_csv("/content/drive/My Drive/sub.csv", index=False)
    res.to_csv("/content/drive/My Drive/res.csv", index=False)    

submission = submission.loc[:,["id"]].merge(sub, on = "id", how = "left").fillna(0)
submission.to_csv("/content/drive/My Drive/submission.csv", index=False)
print(WRMSSE)
