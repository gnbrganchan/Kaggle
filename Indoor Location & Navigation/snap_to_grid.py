# Helper Functions
import pandas as pd
import numpy as np
import os
import multiprocessing
import itertools
from scipy.interpolate import interp2d

import matplotlib.pyplot as plt # visualization
plt.rcParams.update({'font.size': 14})
import seaborn as sns # visualization

from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import math
import json

def split_col(df):
    df = pd.concat([
        df['site_path_timestamp'].str.split('_', expand=True).rename(columns={0:'site',1:'path_id',2:'timestamp'}),
        df], axis=1).copy()
    df["timestamp"] = df["timestamp"].astype(float)
    return df

floor_map = {"B2":-2, "B1":-1, "F1":0, "F2": 1, "F3":2,
             "F4":3, "F5":4, "F6":5, "F7":6,"F8":7,"F9":8,
             "1F":0, "2F":1, "3F":2, "4F":3, "5F":4, "6F":5,
             "7F":6, "8F": 7, "9F":8}

def sub_process(sub, train_waypoints,mode="test"):
    train_waypoints['isTrainWaypoint'] = True
    if mode == "test": sub = split_col(sub[['site_path_timestamp','floor','x','y']]).copy()
    sub = sub.merge(train_waypoints[['site','floor']].drop_duplicates(), how='left')
    sub = sub.merge(
        train_waypoints[['x','y','site','floor','isTrainWaypoint']].drop_duplicates(),
        how='left',on=['site','x','y','floor'])
    sub['isTrainWaypoint'] = sub['isTrainWaypoint'].fillna(False)
    return sub.copy()

def add_xy(df):
    df['xy'] = [(x, y) for x,y in zip(df['x'], df['y'])]
    return df

def mycdist(p,points):
    return [((poi[0]-p[0])**2+(poi[1]-p[1])**2)**0.5 for poi in points]

train_waypoints = add_xy(train_waypoints)

th_max = 15
th_min = 0

if not os.path.exists("train"): os.mkdir("train")
  
  # d_path -- 0:x, 1:y, 2:xy, 3:delta_prev 4,5: waypoint
# locs_path 0:x, 1:y, 2:xy
a = 1
b = 4.0

def s2g(path_id,d_path,locs_path):
    len_path = len(d_path)
    if len(locs_path) > 1000:
        set_wp = set()
        for idx_path in range(len_path):
            dist = mycdist(d_path[idx_path,2],locs_path[:,2])
            set_wp = set_wp | {i for i in range(len(dist)) if dist[i] <= 30}
        locs_path = locs_path[list(set_wp),:]

    n_wp = len(locs_path)
    dp_path = [[i] for i in range(n_wp)]

    for idx_path in range(len_path):
        if idx_path == 0:
            dp_score = a*mycdist(d_path[idx_path,2],locs_path[:,2])
            continue
        score_old = dp_score.copy()
        path_old = dp_path.copy()
        for idx_tr in range(n_wp):
            dist = mycdist(d_path[idx_path,3],
                  [(locs_path[idx_tr,0]-x,locs_path[idx_tr,1]-y) for x,y in zip(locs_path[:,0],locs_path[:,1])]
                 )
            cost = [x+b*y for x,y in zip(score_old,dist)]
            dp_score[idx_tr] = min(cost) + a*mycdist(d_path[idx_path,2],[locs_path[idx_tr,2]])[0]
            dp_path[idx_tr] = path_old[np.argmin(cost)] + [idx_tr]
    d_path = np.insert(d_path,d_path.shape[1],locs_path[dp_path[np.argmin(dp_score)],2],axis=1)
    return d_path

  
# test data
sub = pd.read_csv("../input/cost-minimization-ens01/submission.csv")
sub = sub_process(sub, train_waypoints)
df_rel = pd.concat([pd.read_parquet(f"../input/mart31/test/{site}.parquet") for site in sub.site.unique()])

sub = pd.merge(sub,df_rel[["path_id","timestamp","rel_x","rel_y"]],how="left",on=["path_id","timestamp"])
sub = add_xy(sub)

ds = []
for (site, myfloor), d in tqdm(sub.groupby(['site','floor'])):
    true_floor_locs = train_waypoints.loc[(train_waypoints['floor'] == myfloor) & (train_waypoints['site'] == site)].reset_index(drop=True)
    if len(true_floor_locs) == 0:
        print(f'Skipping {site} {myfloor}')
        continue
    d = d.sort_values(["path_id","timestamp"]).reset_index(drop=True)
    dsprev = d.groupby("path_id").shift(1)
    dsnext = d.groupby("path_id").shift(-1)

    d['delta_prev'] = [(x,y) for x,y in zip(d['rel_x'] - dsprev['rel_x'], d['rel_y'] - dsprev['rel_y'])]
    list_path = d.path_id.unique()
    def s2gp(path_id):
        return s2g(path_id,d.loc[d["path_id"]==path_id,["x","y","xy","delta_prev","site_path_timestamp","floor"]].values,true_floor_locs[["x","y","xy"]].values)
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        dfs = pool.imap_unordered(s2gp, list_path)
        dfs = list(dfs)
    dfs = np.concatenate(dfs)
    dfs = pd.DataFrame(dfs)
    dfs = dfs.set_axis(["x","y","xy","delta_prev",'site_path_timestamp','floor',"matched_point"], axis='columns')
    dfs['x_mp'] = dfs['matched_point'].apply(lambda x: x[0])
    dfs['y_mp'] = dfs['matched_point'].apply(lambda x: x[1])
    ds.append(dfs)

sub = pd.concat(ds)

# Calculate the distances
sub['dist'] = ((sub.x-sub.x_mp)**2 + (sub.y-sub.y_mp)**2)**0.5

sub_pp = sub.copy()
sub_pp["x"] = sub_pp["x_mp"].where((sub_pp["dist"] <= th_max) & (sub_pp["dist"] >= th_min), sub_pp["x"])
sub_pp["y"] = sub_pp["y_mp"].where((sub_pp["dist"] <= th_max) & (sub_pp["dist"] >= th_min), sub_pp["y"])
