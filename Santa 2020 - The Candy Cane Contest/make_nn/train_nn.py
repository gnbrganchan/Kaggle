import argparse
import os
from azureml.core import Run
import numpy as np
import pandas as pd
import subprocess
import time

# Make outputs folder
os.makedirs('outputs', exist_ok=True)

# Set regularization parameter
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=float, dest='num_epoch')
parser.add_argument('--batch_size', type=float, dest='batch_size')
parser.add_argument('--learning_rate', type=float, dest='learning_rate')
parser.add_argument('--drop_rate', type=float, dest='drop_rate')

args = parser.parse_args()
num_epoch = int(args.num_epoch)
bs = int(args.batch_size)
learning_rate = args.learning_rate
drop_rate = args.drop_rate

# Get the experiment run context
run = Run.get_context()

DATA_FILE_LB = 'training_data_lb_1000-1200_20-50.parquet'
TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = ['payout']
data = pd.read_parquet(DATA_FILE_LB,columns=TRAIN_FEATS+TARGET_COL)
data = data.query("n_pulls_self > 0 or n_pulls_opp > 1")

run.log('data',DATA_FILE_LB)

rand = 1024

def dataScaler(df):
    df["round_num"] = df["round_num"] / 2000
    df["n_pulls_self"] = df["n_pulls_self"] / 100
    df["n_success_self"] = df["n_success_self"] / 50
    df["n_pulls_opp"] = df["n_pulls_opp"] / 100
    return df

data = pd.read_parquet(DATA_FILE_LB)

data = dataScaler(data)
X = data[TRAIN_FEATS]
y = data[TARGET_COL]

run.log('data size',len(y))

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
run.log('device',device)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc4to256 = nn.Linear(4, 256)
        self.bn256 = nn.BatchNorm1d(256)
        self.fc256to64 = nn.Linear(256, 64)
        self.bn64 = nn.BatchNorm1d(64)
        self.fc64to16 = nn.Linear(64, 16)
        self.bn16 = nn.BatchNorm1d(16)
        self.fc16to4 = nn.Linear(16, 4)
        self.bn4 = nn.BatchNorm1d(4)
        self.fc4to1 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = F.relu(self.fc4to256(x))
        # x = self.bn256(x)
        x = self.dropout(x)
        x = F.relu(self.fc256to64(x))
        # x = self.bn64(x)
        x = self.dropout(x)
        x = F.relu(self.fc64to16(x))
        # x = self.bn16(x)
        x = self.dropout(x)
        x = F.relu(self.fc16to4(x))
        # x = self.bn4(x)
        x = self.dropout(x)
        x = self.fc4to1(x)
        return x
              

def make_model(X_train,y_train,X_valid,y_valid):
    x_va = Variable(torch.from_numpy(X_valid.values).float())
    y_va = Variable(torch.from_numpy(y_valid.values).float())
    
    train_set = TensorDataset(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
    # valid_set  = TensorDataset(torch.from_numpy(X_valid.values).float(), torch.from_numpy(y_valid.values).float())
    
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    # valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=False, num_workers=os.cpu_count()-1)
    # train_dl = DataLoader(train_set, batch_size=bs, shuffle=True)
    # valid_dl = DataLoader(valid_set, batch_size=bs, shuffle=False)
    
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.99 ** epoch)
    criterion = nn.MSELoss()
    
    # set training mode
    net.train()
    
    # start to train
    epoch_loss = []
    epoch_loss_va = []

    for epoch in range(num_epoch):
        run.log('epoch',epoch+1)
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
            net.train()
        time_end = time.time()
        if (epoch+1) % 1 == 0:
            run.log('train loss',round(np.sqrt(loss_ep/cnt),5))
            run.log('valid loss',round(np.sqrt(loss_va.item()),5))
            run.log('elapsed time',int(time_end-time_start))
            # print(f"epoc:{epoch+1}, Train Loss: {np.sqrt(loss_ep/cnt)}, Valid Loss: {np.sqrt(loss_va.item())}, Elapsed: {int(time_end-time_start)} s")

    return net, epoch_loss, epoch_loss_va

rmse = []
i = 0

nsplits = 5
folds = KFold(n_splits=nsplits, shuffle = True, random_state = rand)

for tr_idx,va_idx in folds.split(X,y):
    # print(f"Fold {i}")
    run.log('fold',i)

    X_tr = X.iloc[tr_idx].reset_index().drop("index",axis = 1)
    y_tr = y.iloc[tr_idx].reset_index().drop("index",axis = 1)
    X_va = X.iloc[va_idx].reset_index().drop("index",axis = 1)
    y_va = y.iloc[va_idx].reset_index().drop("index",axis = 1)
    
    model, epoch_loss, epoch_loss_va = make_model(X_tr,y_tr,X_va,y_va)
    model.eval()
    y_pred = model(Variable(torch.from_numpy(X_va.values).float()))


    rmse.append(np.sqrt(mean_squared_error(y_va, y_pred.data.numpy())))
    run.log('rmse', round(np.sqrt(mean_squared_error(y_va, y_pred.data.numpy())),5))
    i += 1
    
    #save model
    # model = model.to('cpu')
    torch.save(model.state_dict(),f'model{i}.sav')

run.log('mean_rmse', round(np.mean(rmse),5))

subprocess.call(['tar','cvfz', 'outputs/main.py.tar.gz', 'main.py',
 'model1.sav', 'model2.sav', 'model3.sav', 'model4.sav', 'model5.sav'])

