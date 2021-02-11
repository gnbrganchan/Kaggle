import pickle
import base64
import random
import time


import numpy as np
import pandas as pd
import sklearn.tree as skt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# Parameters
FUDGE_FACTOR = 0.99
VERBOSE = False
DATA_FILE = '/kaggle/input/sample-training-data/training_data_201223.parquet'
DATA_FILE＿DT100 = '/kaggle/input/generate-training-data-dt/training_data_dt.parquet'
DATA_FILE＿DT200 = '/kaggle/input/generate-training-data-dt200/training_data_dt.parquet'
DATA_FILE＿DT600 = '/kaggle/input/generate-training-data-dt600/training_data_dt.parquet'
DATA_FILE_LB = '/kaggle/input/generate-training-data-lb3/training_data_lb.parquet'

DATA_FILE＿DT_LGB_OTHER50 = '/kaggle/input/santa2020-train-data/training_data_dt_lgb_other50.parquet'

TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = 'payout'

rand = 1024
##

def dataScaler(df):
    df["round_num"] = df["round_num"] / 2000
    df["n_pulls_self"] = df["n_pulls_self"] / 100
    df["n_success_self"] = df["n_success_self"] / 50
    df["n_pulls_opp"] = df["n_pulls_opp"] / 100
    return df
##

data = pd.read_parquet(DATA_FILE_LB)

data = dataScaler(data)
X = data[TRAIN_FEATS]
y = data[TARGET_COL]
##

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

SEED = 46
ps = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc4to32 = nn.Linear(4, 32)
        self.fc32to64 = nn.Linear(32, 64)
        self.fc64to64 = nn.Linear(64, 64)
        self.fc64to128 = nn.Linear(64, 128)
        self.fc4to128 = nn.Linear(4, 128)
        self.fc128to128 = nn.Linear(128, 128)
        self.fc128to64 = nn.Linear(128, 64)
        self.fc64to32 = nn.Linear(64, 32)
        self.fc32to16 = nn.Linear(32, 16)
        self.fc16to1 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc4to128(x))
        x = self.dropout(x)
#         x = F.relu(self.fc128to128(x))
        x = F.relu(self.fc128to64(x))
#         x = self.dropout(x)
        x = F.relu(self.fc64to32(x))
        x = self.dropout(x)
        x = F.relu(self.fc32to16(x))
#         x = self.dropout(x)
        x = self.fc16to1(x)
        return x
              

def make_model(X_train,y_train,X_valid,y_valid):
#     x = Variable(torch.from_numpy(X_train.values).float(), requires_grad=True)
#     y = Variable(torch.from_numpy(y_train.values).float())
    x_va = Variable(torch.from_numpy(X_valid.values).float())
    y_va = Variable(torch.from_numpy(y_valid.values).float())

    bs = 10000
    
    train_set = TensorDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
    valid_set  = TensorDataset(torch.from_numpy(X_valid.values).float(), torch.from_numpy(y_valid.values).float())
    
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=False)
    
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.99 ** epoch)
    criterion = nn.MSELoss()
    
    # set training mode
    net.train()
    
    # start to train
    epoch_loss = []
    epoch_loss_va = []

    for epoch in range(150):
        time_start = time.time()
        loss_ep = 0
        cnt = 0
        for x,y in train_dl:
            # forward
            outputs = net(x)

            # calculate loss
            loss_tr = criterion(outputs, y)

            # update weights
            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()
            
            loss_ep += loss_tr.data.numpy().tolist()
            
            cnt += 1

        # save loss of this epoch
        epoch_loss.append(np.sqrt(loss_ep/cnt))
        scheduler.step()


        with torch.no_grad():
            net.eval()
            outputs = net(x_va)
            loss_va = criterion(outputs, y_va)
            epoch_loss_va.append(np.sqrt(loss_va.data.numpy().tolist()))
        time_end = time.time()
        if (epoch+1) % 1 == 0:
            print(f"epoc:{epoch+1}, Train Loss: {np.sqrt(loss_ep/cnt)}, Valid Loss: {np.sqrt(loss_va.item())}, Elapsed: {int(time_end-time_start)} s")

    return net, epoch_loss, epoch_loss_va
##

rmse = []
i = 0

nsplits = 5
folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)

# pretrainer = pretrain_model(X,y)

for tr_idx,va_idx in folds.split(X,y):
    print(f"Fold {i}")
    X_tr = X.iloc[tr_idx].reset_index().drop("index",axis = 1)
    y_tr = y.iloc[tr_idx].reset_index().drop("index",axis = 1)
    X_va = X.iloc[va_idx].reset_index().drop("index",axis = 1)
    y_va = y.iloc[va_idx].reset_index().drop("index",axis = 1)
    
    model, epoch_loss, epoch_loss_va = make_model(X_tr,y_tr,X_va,y_va)
    model.eval()
#     y_pred = model.predict(X_va.values)
    y_pred = model(Variable(torch.from_numpy(X_va.values).float()))


    rmse.append(np.sqrt(mean_squared_error(y_va, y_pred.data.numpy())))
    i += 1
    
    #save model
#     pickle.dump(model, open(f'model{i}.sav', 'wb'))
    torch.save(model.to('cpu').state_dict(),f'model{i}.sav')
#     model.save(f'model{i}')
#     del model
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(list(range(len(epoch_loss))), epoch_loss,label = "Train Loss")
    ax.plot(list(range(len(epoch_loss_va))), epoch_loss_va, label = "Valid Loss")
    ax.set_xlabel('#epoch')
    ax.set_ylabel('loss')
    fig.show()
##

!tar cvfz main.py.tar.gz main.py model1.sav model2.sav model3.sav model4.sav model5.sav
