from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, StratifiedKFold,GroupKFold, KFold

import os
import gc
import time

import warnings
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from tqdm.notebook import tqdm
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ProgbarLogger

from tensorflow.keras.optimizers.schedules import ExponentialDecay

EPOCH = 1200
BATCH_SIZE = 256
WEIGHT_DECAY = 0.0000001

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

path = "/content/drive/My Drive/input/googlebrain/"
df_train = pd.read_csv(path + "train.csv")
df_test = pd.read_csv(path + "test.csv")
df_sub = pd.read_csv(path + "sample_submission.csv")

df_train["row_g"] = df_train.groupby("breath_id").cumcount()
df_test["row_g"] = df_test.groupby("breath_id").cumcount()

idx_ins = df_train.query("u_out == 0").index

# df_train["diff_u_in"] = df_train["u_in"] - df_train.groupby("breath_id")["u_in"].shift(1)
# df_test["diff_u_in"] = df_test["u_in"] - df_test.groupby("breath_id")["u_in"].shift(1)
# df_train["diff_diff_u_in"] = df_train["diff_u_in"] - df_train.groupby("breath_id")["diff_u_in"].shift(1)
# df_test["diff_diff_u_in"] = df_test["diff_u_in"] - df_test.groupby("breath_id")["diff_u_in"].shift(1)

# cumsum
df_train["cumsum_u_in"] = df_train.groupby("breath_id")["u_in"].cumsum()
df_test["cumsum_u_in"] = df_test.groupby("breath_id")["u_in"].cumsum()
# df_train["up_u_in"] = df_train["diff_u_in"].apply(lambda x: 1 if x > 0 else 0)
# df_train["cumsum_up_u_in"] = df_train.groupby("breath_id")["up_u_in"].cumsum()
# df_train["down_u_in"] = df_train["diff_u_in"].apply(lambda x: 1 if x < 0 else 0)
# df_train["cumsum_down_u_in"] = df_train.groupby("breath_id")["down_u_in"].cumsum()
# df_train["zero_u_in"] = df_train["diff_u_in"].apply(lambda x: 1 if x == 0 else 0)
# df_train["cumsum_zero_u_in"] = df_train.groupby("breath_id")["zero_u_in"].cumsum()
# df_test["up_u_in"] = df_test["diff_u_in"].apply(lambda x: 1 if x > 0 else 0)
# df_test["cumsum_up_u_in"] = df_test.groupby("breath_id")["up_u_in"].cumsum()
# df_test["down_u_in"] = df_test["diff_u_in"].apply(lambda x: 1 if x < 0 else 0)
# df_test["cumsum_down_u_in"] = df_test.groupby("breath_id")["down_u_in"].cumsum()
# df_test["zero_u_in"] = df_test["diff_u_in"].apply(lambda x: 1 if x == 0 else 0)
# df_test["cumsum_zero_u_in"] = df_test.groupby("breath_id")["zero_u_in"].cumsum()

# df_train["mean_cumsum_u_in"] = df_train["cumsum_u_in"] / (df_train["row_g"]+1)
# df_test["mean_cumsum_u_in"] = df_test["cumsum_u_in"] / (df_test["row_g"]+1)

# df_train["exp_mean_u_in"] = df_train.groupby('breath_id')['u_in'].expanding().mean().reset_index(level=0,drop=True)
# df_train["exp_std_u_in"] = df_train.groupby('breath_id')['u_in'].expanding().std().reset_index(level=0,drop=True)
# df_train["exp_max_u_in"] = df_train.groupby('breath_id')['u_in'].expanding().max().reset_index(level=0,drop=True)
# df_train["exp_min_u_in"] = df_train.groupby('breath_id')['u_in'].expanding().min().reset_index(level=0,drop=True)
# df_test["exp_mean_u_in"] = df_test.groupby('breath_id')['u_in'].expanding().mean().reset_index(level=0,drop=True)
# df_test["exp_std_u_in"] = df_test.groupby('breath_id')['u_in'].expanding().std().reset_index(level=0,drop=True)
# df_test["exp_max_u_in"] = df_test.groupby('breath_id')['u_in'].expanding().max().reset_index(level=0,drop=True)
# df_test["exp_min_u_in"] = df_test.groupby('breath_id')['u_in'].expanding().min().reset_index(level=0,drop=True)

# df_train['ewm_mean_u_in'] = df_train.groupby('breath_id')['u_in'].apply(lambda x: x.ewm(halflife=10).mean().iloc[-1])
# df_train['ewm_std_u_in'] = df_train.groupby('breath_id')['u_in'].apply(lambda x: x.ewm(halflife=10).std().iloc[-1])
# df_test['ewm_mean_u_in'] = df_test.groupby('breath_id')['u_in'].apply(lambda x: x.ewm(halflife=10).mean().iloc[-1])
# df_test['ewm_std_u_in'] = df_test.groupby('breath_id')['u_in'].apply(lambda x: x.ewm(halflife=10).std().iloc[-1])

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

X_train = df_train.drop(["id","breath_id","pressure","row_g"],axis=1)
y_train = df_train["pressure"].to_numpy().reshape(-1, 80)
X_test = df_test.drop(["id","breath_id","row_g"],axis=1)

RS = RobustScaler()
X_train = RS.fit_transform(X_train)
X_test = RS.transform(X_test)

X_train = X_train.reshape(-1, 80, X_train.shape[-1])
X_test = X_test.reshape(-1, 80, X_train.shape[-1])

from tensorflow.keras.callbacks import Callback
class myLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self,count_mode='steps', stateful_metrics=None):
        super(myLogger, self).__init__(count_mode, stateful_metrics)
        self.show_step = 1
        self.time_start = time.time()
    
    def on_train_begin(self, logs=None):
        pass
#         self.epochs = self.params['epochs']
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%10 == 0:
            time_diff = time.time() - self.time_start
            self.time_start = time.time()
            print(f"Epoch: {epoch+1}  time:  {time_diff:.1f}s  loss:{logs['loss']:.3f} ")
# def dnn_model():
    
#     x_input = keras.layers.Input(shape=(X_train.shape[-2:]))
    
#     x1 = keras.layers.Bidirectional(keras.layers.LSTM(units=768, return_sequences=True))(x_input)
#     x2 = keras.layers.Bidirectional(keras.layers.LSTM(units=512, return_sequences=True))(x1)
#     x3 = keras.layers.Bidirectional(keras.layers.LSTM(units=384, return_sequences=True))(x2)
#     x4 = keras.layers.Bidirectional(keras.layers.LSTM(units=256, return_sequences=True))(x3)
#     x5 = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(x4)
    
#     z2 = keras.layers.Bidirectional(keras.layers.GRU(units=384, return_sequences=True))(x2)
    
#     z31 = keras.layers.Multiply()([x3, z2])
#     z31 = keras.layers.BatchNormalization()(z31)
#     z3 = keras.layers.Bidirectional(keras.layers.GRU(units=256, return_sequences=True))(z31)
    
#     z41 = keras.layers.Multiply()([x4, z3])
#     z41 = keras.layers.BatchNormalization()(z41)
#     z4 = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=True))(z41)
    
#     z51 = keras.layers.Multiply()([x5, z4])
#     z51 = keras.layers.BatchNormalization()(z51)
#     z5 = keras.layers.Bidirectional(keras.layers.GRU(units=64, return_sequences=True))(z51)
    
#     x = keras.layers.Concatenate(axis=2)([x5, z2, z3, z4, z5])
    
#     x = keras.layers.Dense(units=128, activation='selu')(x)
    
#     x_output = keras.layers.Dense(units=1)(x)

#     model = keras.models.Model(inputs=x_input, outputs=x_output, name='DNN_Model')
#     return model

# nsplits = 5
# rand = 46

# folds = StratifiedKFold(n_splits = nsplits, shuffle = True, random_state = rand)
# for i,(tr_idx_id,va_idx_id) in enumerate(folds.split(df_agg,df_agg["mean_pressure"])):
#     tr_id, va_id = df_agg.loc[tr_idx_id,"breath_id"], df_agg.loc[va_idx_id,"breath_id"]
#     # tr_idx = df_train[df_train["breath_id"].isin(tr_id)].index
#     va_idx = df_train[df_train["breath_id"].isin(va_id)].index
#     with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_tr_idx_id_2.csv", 'wb') as f:
#       pickle.dump(tr_idx_id, f)
#     with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_va_idx_id_2.csv", 'wb') as f:
#       pickle.dump(va_idx_id, f)
#     with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_va_idx_2.csv", 'wb') as f:
#       pickle.dump(va_idx, f)

# # %%time
# with tpu_strategy.scope():
#     for i in tqdm([0,1,2,3,4]):
#         print(f"======FOLD {i}======")
#         if i == 0:
#           oof = np.zeros(len(df_train))
#           pred = np.zeros(len(df_test))
#         else:
#           with open(f"/content/drive/My Drive/output/googlebrain/train_oof_lstm02_fold.csv", 'rb') as f:
#             oof = pickle.load(f)
#           with open(f"/content/drive/My Drive/output/googlebrain/pred_lstm02_fold.csv", 'rb') as f:
#             pred = pickle.load(f)

#         with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_tr_idx_id_2.csv", 'rb') as f:
#           tr_idx_id = pickle.load(f)
#         with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_va_idx_id_2.csv", 'rb') as f:
#           va_idx_id = pickle.load(f)
#         with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_va_idx_2.csv", 'rb') as f:
#           va_idx = pickle.load(f)

#         X_tr, X_va = X_train[tr_idx_id],X_train[va_idx_id]
#         y_tr, y_va = y_train[tr_idx_id],y_train[va_idx_id]

#         model = keras.models.Sequential([
#             keras.layers.Input(shape=X_train.shape[-2:]),
#             keras.layers.Bidirectional(keras.layers.LSTM(300, return_sequences=True,kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),recurrent_regularizer=keras.regularizers.l2(WEIGHT_DECAY))),
#             keras.layers.Bidirectional(keras.layers.LSTM(250, return_sequences=True,kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),recurrent_regularizer=keras.regularizers.l2(WEIGHT_DECAY))),
#             keras.layers.Bidirectional(keras.layers.LSTM(150, return_sequences=True,kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),recurrent_regularizer=keras.regularizers.l2(WEIGHT_DECAY))),
#             keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True,kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),recurrent_regularizer=keras.regularizers.l2(WEIGHT_DECAY))),
#             keras.layers.Dense(50, activation='selu'),
#             keras.layers.Dense(1),
#         ])
#         # model = dnn_model()
#         model.compile(optimizer="adam", loss="mae")

#         scheduler = ExponentialDecay(1e-3, int(EPOCH*((len(X_train)*(nsplits-1)/nsplits)/BATCH_SIZE)), 1e-5)
#         lr = LearningRateScheduler(scheduler)
#         mylogger = myLogger()


#         model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr,mylogger], verbose=0)

#         oof[va_idx] = model.predict(X_va).squeeze().reshape(-1, 1).squeeze()
#         pred += model.predict(X_test).squeeze().reshape(-1, 1).squeeze() / nsplits
#         with open(f"/content/drive/My Drive/output/googlebrain/train_oof_lstm02_fold.csv", 'wb') as f:
#           pickle.dump(oof, f)
#         with open(f"/content/drive/My Drive/output/googlebrain/pred_lstm02_fold.csv", 'wb') as f:
#           pickle.dump(pred, f)

# # score
# mae_score = (oof[idx_ins]-df_train.loc[idx_ins,'pressure']).abs().mean()
# print(f"MAE: {mae_score}")

# df_train["pressure_pred"] = oof
# df_train[["id","pressure_pred"]].to_csv("/content/drive/My Drive/output/googlebrain/train_oof_.csv", index=False)
df_sub["pressure"] = pred
df_sub.to_csv("/content/drive/My Drive/output/googlebrain/pred_noCV.csv", index=False)
