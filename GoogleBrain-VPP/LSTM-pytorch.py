from sklearn.model_selection import StratifiedKFold
import os
import gc
import time

import warnings
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from collections import Counter

warnings.filterwarnings("ignore")
NUM_WORKERS = 4

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import pickle

path = "/content/drive/My Drive/input/googlebrain/"
df_train = pd.read_csv(path + "train.csv")
df_test = pd.read_csv(path + "test.csv")
df_sub = pd.read_csv(path + "sample_submission.csv")

df_train["row_g"] = df_train.groupby("breath_id").cumcount()
row_max = df_train.query("u_out == 0")["row_g"].max()
idx_ins = df_train.query("u_out == 0").index

df_train["diff_u_in"] = df_train["u_in"] - df_train.groupby("breath_id")["u_in"].shift(1)
df_test["diff_u_in"] = df_test["u_in"] - df_test.groupby("breath_id")["u_in"].shift(1)

# cumsum
df_cumsum = df_train.groupby("breath_id")["u_in"].cumsum()
df_train["cumsum_u_in"] = df_cumsum
df_cumsum = df_test.groupby("breath_id")["u_in"].cumsum()
df_test["cumsum_u_in"] = df_cumsum

df_train["up_u_in"] = df_train["diff_u_in"].apply(lambda x: 1 if x > 0 else 0)
df_train["cumsum_up_u_in"] = df_train.groupby("breath_id")["up_u_in"].cumsum()
df_train["down_u_in"] = df_train["diff_u_in"].apply(lambda x: 1 if x < 0 else 0)
df_train["cumsum_down_u_in"] = df_train.groupby("breath_id")["down_u_in"].cumsum()
df_train["zero_u_in"] = df_train["diff_u_in"].apply(lambda x: 1 if x == 0 else 0)
df_train["cumsum_zero_u_in"] = df_train.groupby("breath_id")["zero_u_in"].cumsum()
df_test["up_u_in"] = df_test["diff_u_in"].apply(lambda x: 1 if x > 0 else 0)
df_test["cumsum_up_u_in"] = df_test.groupby("breath_id")["up_u_in"].cumsum()
df_test["down_u_in"] = df_test["diff_u_in"].apply(lambda x: 1 if x < 0 else 0)
df_test["cumsum_down_u_in"] = df_test.groupby("breath_id")["down_u_in"].cumsum()
df_test["zero_u_in"] = df_test["diff_u_in"].apply(lambda x: 1 if x == 0 else 0)
df_test["cumsum_zero_u_in"] = df_test.groupby("breath_id")["zero_u_in"].cumsum()

class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = False

#     # k-fold
#     k = 5
#     selected_folds = [0, 1, 2, 3, 4]
    
    # Model
    selected_model = 'rnn'
    input_dim = 13

    dense_dim = 512
    lstm_dim = 512
    logit_dim = 512
    num_classes = 1
    num_layers = 3
    dropout = 0

    # Training
    loss = "L1Loss"  # not used
    optimizer = "Adam"
    batch_size = 128
    epochs = 400

    lr = 1e-3
    weight_decay = 1e-4
    warmup_prop = 0

    val_bs = 256
    first_epoch_eval = 0
    
class VentilatorDataset(Dataset):
    def __init__(self, df):
        if "pressure" not in df.columns:
            df['pressure'] = 0

        self.df = df.groupby('breath_id').agg(list).reset_index()
        
        self.prepare_data()
                
    def __len__(self):
        return self.df.shape[0]
    
    def prepare_data(self):
        self.pressures = np.array(self.df['pressure'].values.tolist())
        
        rs = np.array(self.df['R'].values.tolist())
        cs = np.array(self.df['C'].values.tolist())
        # u_ins = np.array(self.df['u_in'].values.tolist())
        u_ins_prev2 = np.array(self.df['u_in_prev2'].values.tolist())
        # cumsum_u_ins = np.array(self.df['cumsum_u_in'].values.tolist())
        cumsum_u_ins_prev2 = np.array(self.df['cumsum_u_in_prev2'].values.tolist())
        time_steps = np.array(self.df['time_step'].values.tolist())
        # diff_u_ins = np.array(self.df['diff_u_in'].values.tolist())
        # cumsum_up_u_ins = np.array(self.df['cumsum_up_u_in'].values.tolist())
        # cumsum_down_u_ins = np.array(self.df['cumsum_down_u_in'].values.tolist())
        # cumsum_zero_u_ins = np.array(self.df['cumsum_zero_u_in'].values.tolist())
        # n_up_u_ins = np.array(self.df['n_up_u_in'].values.tolist())
        # n_down_u_ins = np.array(self.df['n_down_u_in'].values.tolist())
        # n_zero_u_ins = np.array(self.df['n_zero_u_in'].values.tolist())
        # lgbm01 = np.array(self.df['lgbm01'].values.tolist())
        # lstm01 = np.array(self.df['lstm01'].values.tolist())
        lstm02 = np.array(self.df['lstm02'].values.tolist())
        # lstm04 = np.array(self.df['lstm04'].values.tolist())
        lstm05 = np.array(self.df['lstm05'].values.tolist())
        # lstm06 = np.array(self.df['lstm06'].values.tolist())
        # gru01 = np.array(self.df['gru01'].values.tolist())
        stack01 = np.array(self.df['stack01'].values.tolist())
        stack02 = np.array(self.df['stack02'].values.tolist())
        stack04 = np.array(self.df['stack04'].values.tolist())
        stack05 = np.array(self.df['stack05'].values.tolist())
        stack06 = np.array(self.df['stack06'].values.tolist())

         
        self.u_outs = np.array(self.df['u_out'].values.tolist())
        
        self.inputs = np.concatenate([
            rs[:, None], 
            cs[:, None], 
            # u_ins[:, None], 
            u_ins_prev2[:, None], 
            # cumsum_u_ins[:, None],
            cumsum_u_ins_prev2[:, None],
            time_steps[:,None],
            # diff_u_ins[:,None],
            # cumsum_up_u_ins[:,None],
            # cumsum_down_u_ins[:,None],
            # cumsum_zero_u_ins[:,None],
            # n_up_u_ins[:,None],
            # n_down_u_ins[:,None],
            # n_zero_u_ins[:,None],
            # lgbm01[:,None],
            # lstm01[:,None],
            lstm02[:,None],
            # lstm04[:,None],
            lstm05[:,None],
            # lstm06[:,None],
            # gru01[:,None],
            stack01[:,None],
            stack02[:,None],
            stack04[:,None],
            stack05[:,None],
            stack06[:,None],

            self.u_outs[:, None]
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }
        
        return data
      
  # class RNNModel(nn.Module):
#     def __init__(
#         self,
#         input_dim=4,
#         lstm_dim=256,
#         dense_dim=256,
#         logit_dim=256,
#         num_classes=1,
#         num_layers=1,
#         dropout=0
#     ):
#         super().__init__()

#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, dense_dim // 2),
#             nn.ReLU(),
#             nn.Linear(dense_dim // 2, dense_dim),
#             nn.ReLU(),
#         )

#         # self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True) # original
#         self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True,num_layers=num_layers,dropout=dropout)


#         self.logits = nn.Sequential(
#             nn.Linear(lstm_dim * 2, logit_dim),
#             nn.ReLU(),
#             nn.Linear(logit_dim, num_classes),
#         )

#     def forward(self, x):
#         features = self.mlp(x)
#         features, _ = self.lstm(features)
#         pred = self.logits(features)
#         return pred

# from common import *
from einops.layers.torch import Rearrange

#----------------------------------------------------------------------------------

def mask_huber_loss(predict, truth, m, delta=0.1):
    loss = F.huber_loss(predict[m], truth[m], delta=delta)
    return loss

def mask_l1_loss(predict, truth, m):
    loss = F.l1_loss(predict[m], truth[m])
    return loss

def mask_smooth_l1_loss(predict, truth, m, beta=0.1):
    loss = F.smooth_l1_loss(predict[m], truth[m], beta=beta)
    return loss


#------------
def rc_to_index(r,c):
    #r_map = { 5: 0, 20: 1, 50: 2}
    #c_map = {10: 0, 20: 1, 50: 2}
    r = (r== 5).float()*1 + (r==20).float()*2 + (r==50).float()*3
    c = (c==10).float()*1 + (c==20).float()*2 + (c==50).float()*3
    r = r.long()
    c = c.long()
    return r,c

class RNNModel(nn.Module):
    def __init__(self,input_dim=10):
        super().__init__()
        self.r_embed = nn.Embedding(4, 2) #, padding_idx=0)
        self.c_embed = nn.Embedding(4, 2) #, padding_idx=0)
        self.seq_embed = nn.Sequential(
            Rearrange('b l d -> b d l'),
            nn.Conv1d(2+input_dim, 32, kernel_size=5, padding=2, stride=1),
            Rearrange('b d l -> b l d'),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.),
        )
        self.lstm1 = nn.LSTM(    32, 400, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM( 2*400, 300, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM( 2*300, 200, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM( 2*200, 100, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(2*100, 50),
            nn.SiLU(),
        )
        self.pressure_in  = nn.Linear(50, 1)
        self.pressure_out = nn.Linear(50, 1)

        #----
        #initialisation : https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization/notebook
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                #print(name,m)
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        batch_size = len(x)

        r,c = rc_to_index(x[:,:,-2],x[:,:,-1])
        r = self.r_embed(r)
        c = self.c_embed(c)
        seq = torch.cat((r, c, x[:,:,:-2]), 2)
        x = self.seq_embed(seq)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.head(x)

        pressure_in  = self.pressure_in(x)#.reshape(batch_size,80)
        pressure_out = self.pressure_out(x)#.reshape(batch_size,80)
        return pressure_in#, pressure_out

      def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def count_parameters(model, all=False):
    """
    Counts the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))
    
def compute_metric(df, preds):
    """
    Metric for the problem, as I understood it.
    """
    
    y = np.array(df['pressure'].values.tolist())
    w = 1 - np.array(df['u_out'].values.tolist())
    
    assert y.shape == preds.shape and w.shape == y.shape, (y.shape, preds.shape, w.shape)
    
    mae = w * np.abs(y - preds)
    mae = mae.sum() / w.sum()
    
    return mae


class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric
    """
    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)

        return mae
  
  def fit(
    model,
    train_dataset,
    val_dataset,
    loss_name="L1Loss",
    optimizer="Adam",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    weight_decay = 0,
    num_classes=1,
    verbose=1,
    first_epoch_eval=0,
    device="cuda"
):
    avg_val_loss = 0.

    # Optimizer
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Loss
    loss_fct = getattr(torch.nn, loss_name)(reduction="none")
    # loss_fct = VentilatorLoss()

    # Scheduler
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        start_time = time.time()

        avg_loss = 0
        for data in train_loader:
            pred = model(data['input'].to(device)).squeeze(-1)

            loss = loss_fct(
                pred,
                data['p'].to(device),
                # data['u_out'].to(device),
            ).mean()
            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        model.eval()
        mae, avg_val_loss = 0, 0
        preds = []

        with torch.no_grad():
            for data in val_loader:
                pred = model(data['input'].to(device)).squeeze(-1)

                loss = loss_fct(
                    pred.detach(), 
                    data['p'].to(device),
                    # data['u_out'].to(device),
                ).mean()
                avg_val_loss += loss.item() / len(val_loader)

                preds.append(pred.detach().cpu().numpy())
        
        preds = np.concatenate(preds, 0)
        mae = compute_metric(val_dataset.df, preds)

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )

            if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
                print(f"val_loss={avg_val_loss:.3f}\tmae={mae:.3f}")
            else:
                print("")

    del (val_loader, train_loader, loss, data, pred)
    gc.collect()
    torch.cuda.empty_cache()

    return preds
  
def predict(
    model,
    dataset,
    batch_size=64,
    device="cuda"
):
    """
    Usual torch predict function. Supports sigmoid and softmax activations.
    Args:
        model (torch model): Model to predict with.
        dataset (PathologyDataset): Dataset to predict on.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    
    preds = []
    with torch.no_grad():
        for data in loader:
            pred = model(data['input'].to(device)).squeeze(-1)
            preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(preds, 0)
    return preds
 
def train(config, df_train, df_val, df_test,  fold):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        df_test (pandas dataframe): Test metadata.
        fold (int): Selected fold.

    Returns:
        np array: Study validation predictions.
    """

    seed_everything(config.seed)

    model = RNNModel(
        input_dim=config.input_dim,
        # lstm_dim=config.lstm_dim,
        # dense_dim=config.dense_dim,
        # logit_dim=config.logit_dim,
        # num_classes=config.num_classes,
        # num_layers=config.num_layers,
        # dropout=config.dropout
    ).to(config.device)
    model.zero_grad()

    train_dataset = VentilatorDataset(df_train)
    val_dataset = VentilatorDataset(df_val)
    test_dataset = VentilatorDataset(df_test)

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training breathes")
    print(f"    -> {len(val_dataset)} validation breathes")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        loss_name=config.loss,
        optimizer=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )
    
    pred_test = predict(
        model, 
        test_dataset, 
        batch_size=config.val_bs, 
        device=config.device
    )

    if config.save_weights:
        save_model_weights(
            model,
            f"{config.selected_model}_{fold}.pt",
            cp_folder="",
        )

    del (model, train_dataset, val_dataset, test_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    return pred_val,pred_test
  
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
# X_train = df_train.drop(["id","breath_id","pressure"],axis=1)
y_train = df_train["pressure"]
# X_test = df_test.drop(["id","breath_id"],axis=1)

nsplits = 5
rand = 46

# folds = StratifiedKFold(n_splits = nsplits, shuffle = True, random_state = rand)
# for i,(tr_idx_id,va_idx_id) in enumerate(folds.split(df_agg,df_agg["mean_pressure"])):
#     tr_id, va_id = df_agg.loc[tr_idx_id,"breath_id"], df_agg.loc[va_idx_id,"breath_id"]
#     tr_idx = df_train[df_train["breath_id"].isin(tr_id)].index
#     va_idx = df_train[df_train["breath_id"].isin(va_id)].index
#     with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_tr_idx_1.csv", 'wb') as f:
#       pickle.dump(tr_idx, f)
#     with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_va_idx_1.csv", 'wb') as f:
#       pickle.dump(va_idx, f)

# # R、Cごとモデル作成
# nsplits = 5
# rand = 46

# for R in tqdm([50]):#5,20,50
#     if R == 5:
#       oof = np.zeros(len(df_train))
#       pred = np.zeros(len(df_test))
#     else:
#       with open(f"/content/drive/My Drive/output/googlebrain/train_oof_lstm01_fold.csv", 'rb') as f:
#         oof = pickle.load(f)
#       with open(f"/content/drive/My Drive/output/googlebrain/pred_lstm01_fold.csv", 'rb') as f:
#         pred = pickle.load(f)

#     for C in tqdm([10,20,50]):
#         folds = StratifiedKFold(n_splits = nsplits, shuffle = True, random_state = rand)
#         rc_idx_train = df_agg.query("R == @R and C == @C").index
#         rc_idx_test = df_test.query("R == @R and C == @C").index
#         for i,(tr_idx_id,va_idx_id) in enumerate(folds.split(df_agg.loc[rc_idx_train],df_agg.loc[rc_idx_train,"mean_pressure"])):
#             print(f"======FOLD {i}======")
#             tr_idx_id = df_agg.loc[rc_idx_train].iloc[tr_idx_id].index
#             va_idx_id = df_agg.loc[rc_idx_train].iloc[va_idx_id].index

#             tr_id, va_id = df_agg.loc[tr_idx_id,"breath_id"], df_agg.loc[va_idx_id,"breath_id"]
#             tr_idx = df_train[df_train["breath_id"].isin(tr_id)].index
#             va_idx = df_train[df_train["breath_id"].isin(va_id)].index

#             df_tr = df_train.iloc[tr_idx].copy().reset_index(drop=True)
#             df_va = df_train.iloc[va_idx].copy().reset_index(drop=True)

#             pred_val,pred_test = train(Config, df_tr, df_va, df_test.loc[rc_idx_test], i)
            
#             oof[va_idx] = pred_val.flatten()
#             pred[rc_idx_test] += pred_test.flatten() / nsplits
#     with open(f"/content/drive/My Drive/output/googlebrain/train_oof_lstm01_fold.csv", 'wb') as f:
#       pickle.dump(oof, f)
#     with open(f"/content/drive/My Drive/output/googlebrain/pred_lstm01_fold.csv", 'wb') as f:
#       pickle.dump(pred, f)
#     # df_train["pressure_pred"] = oof
#     # df_train[["id","R","C","pressure","pressure_pred"]].to_csv(f"/content/drive/My Drive/output/googlebrain/train_oof_lstm01_R{R}.csv", index=False)

# # folds = StratifiedKFold(n_splits = nsplits, shuffle = True, random_state = rand)
# # for i,(tr_idx_id,va_idx_id) in enumerate(folds.split(df_agg,df_agg["mean_pressure"])):
# #     print(f"======FOLD {i}======")
# #     tr_id, va_id = df_agg.loc[tr_idx_id,"breath_id"], df_agg.loc[va_idx_id,"breath_id"]
# #     tr_idx = df_train[df_train["breath_id"].isin(tr_id)].index
# #     va_idx = df_train[df_train["breath_id"].isin(va_id)].index

# #     df_tr = df_train.iloc[tr_idx].copy().reset_index(drop=True)
# #     df_va = df_train.iloc[va_idx].copy().reset_index(drop=True)

# #     pred_val = train(Config, df_tr, df_va, i)
    
# #     oof[va_idx] = pred_val.flatten()
# #     # pred += pred_test.flatten() / nsplits
    
# # score
# mae_score = (oof[idx_ins]-y_train[idx_ins]).abs().mean()
# print(f"MAE: {mae_score}")


for i in tqdm([0,1,2,3,4]):
    print(f"======FOLD {i}======")
    if i == 0:
      oof = np.zeros(len(df_train))
      pred = np.zeros(len(df_test))
    else:
      with open(f"/content/drive/My Drive/output/googlebrain/train_oof_lstm01_fold.csv", 'rb') as f:
        oof = pickle.load(f)
      with open(f"/content/drive/My Drive/output/googlebrain/pred_lstm01_fold.csv", 'rb') as f:
        pred = pickle.load(f)

    with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_tr_idx_1.csv", 'rb') as f:
      tr_idx = pickle.load(f)
    with open(f"/content/drive/My Drive/output/googlebrain/CV{i}_va_idx_1.csv", 'rb') as f:
      va_idx = pickle.load(f)

    df_tr = df_train.iloc[tr_idx].copy().reset_index(drop=True)
    df_va = df_train.iloc[va_idx].copy().reset_index(drop=True)

    pred_val,pred_test = train(Config, df_tr, df_va, df_test, i)
    
    oof[va_idx] = pred_val.flatten()
    pred += pred_test.flatten() / nsplits
    with open(f"/content/drive/My Drive/output/googlebrain/train_oof_lstm01_fold.csv", 'wb') as f:
      pickle.dump(oof, f)
    with open(f"/content/drive/My Drive/output/googlebrain/pred_lstm01_fold.csv", 'wb') as f:
      pickle.dump(pred, f)

    
# score
mae_score = (oof[idx_ins]-y_train[idx_ins]).abs().mean()
print(f"MAE: {mae_score}")

df_train["pressure_pred"] = oof
df_train[["id","pressure_pred"]].to_csv("/content/drive/My Drive/output/googlebrain/train_oof_stack10.csv", index=False)
df_sub["pressure"] = pred
df_sub.to_csv("/content/drive/My Drive/output/googlebrain/pred_stack10.csv", index=False)
