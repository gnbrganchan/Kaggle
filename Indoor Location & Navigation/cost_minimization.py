!git clone --depth 1 https://github.com/location-competition/indoor-location-competition-20 indoor_location_competition_20
!rm -rf indoor_location_competition_20/data

import multiprocessing
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.sparse
from tqdm.notebook import tqdm
import os
import glob

from indoor_location_competition_20.io_f import read_data_file
import indoor_location_competition_20.compute_f as compute_f

if not os.path.exists("train"): os.mkdir("train")

INPUT_PATH = '../input/indoor-location-navigation'

def correct_path_tr(args):
    path, path_df = args
    
    T_ref  = path_df['timestamp'].values
    xy_hat = path_df[['x', 'y']].values
    
    filename = glob.glob(INPUT_PATH + f"/**/{path}.txt", recursive = True)[0]
    example = read_data_file(filename)
    rel_positions = compute_rel_positions(example.acce_uncali, example.ahrs)

    if T_ref[-1] > rel_positions[-1, 0]:
        rel_positions = [np.array([[0, 0, 0]]), rel_positions, np.array([[T_ref[-1], 0, 0]])]
    else:
        rel_positions = [np.array([[0, 0, 0]]), rel_positions]
    rel_positions = np.concatenate(rel_positions)
    
    T_rel = rel_positions[:, 0]
    delta_xy_hat = np.diff(scipy.interpolate.interp1d(T_rel, np.cumsum(rel_positions[:, 1:3], axis=0), axis=0)(T_ref), axis=0)
    delta_xy_hat2 = np.diff(scipy.interpolate.interp1d(T_rel, np.cumsum(rel_positions[:, 1:3], axis=0), axis=0)(T_ref), axis=0, n=2)

    N = xy_hat.shape[0]
    delta_t = np.diff(T_ref)
    delta_t2 = np.diff(T_ref,n=2)
    alpha = (4.8)**(-2) * np.ones(N) # 8.1
    beta  = (0.3 + 0.3 * 1e-3 * delta_t)**(-2) # 0.3

#     print(alpha)
#     print(beta)
    A = scipy.sparse.spdiags(alpha, [0], N, N)
    B = scipy.sparse.spdiags( beta, [0], N-1, N-1)
    D = scipy.sparse.spdiags(np.stack([-np.ones(N), np.ones(N)]), [0, 1], N-1, N)


    Q = A + (D.T @ B @ D)
    c = (A @ xy_hat) + (D.T @ (B @ delta_xy_hat))
    xy_star = scipy.sparse.linalg.spsolve(Q, c)

    return pd.DataFrame({
        'path_id' : path_df['path_id'],
        'timestamp': path_df['timestamp'],
        'floor' : path_df['floor'],
        'x' : xy_star[:, 0],
        'y' : xy_star[:, 1],
        'waypoint_x':path_df["waypoint_x"],
        'waypoint_y':path_df["waypoint_y"]
    })
    
def correct_path(args):
    path, path_df = args
    
    T_ref  = path_df['timestamp'].values
    xy_hat = path_df[['x', 'y']].values
    xy_leak = path_df[['x_leak', 'y_leak']].values
    
    example = read_data_file(f'{INPUT_PATH}/test/{path}.txt')
    rel_positions = compute_rel_positions(example.acce_uncali, example.ahrs)
    if T_ref[-1] > rel_positions[-1, 0]:
        rel_positions = [np.array([[0, 0, 0]]), rel_positions, np.array([[T_ref[-1], 0, 0]])]
    else:
        rel_positions = [np.array([[0, 0, 0]]), rel_positions]
    rel_positions = np.concatenate(rel_positions)
    
    T_rel = rel_positions[:, 0]
    delta_xy_hat = np.diff(scipy.interpolate.interp1d(T_rel, np.cumsum(rel_positions[:, 1:3], axis=0), axis=0)(T_ref), axis=0)
    delta_xy_hat2 = np.diff(scipy.interpolate.interp1d(T_rel, np.cumsum(rel_positions[:, 1:3], axis=0), axis=0)(T_ref), axis=0, n=2)

    N = xy_hat.shape[0]
    delta_t = np.diff(T_ref)
    delta_t2 = np.diff(T_ref,n=2)
    alpha = (4.8)**(-2) * np.ones(N) # 8.1
    beta  = (0.3 + 0.3 * 1e-3 * delta_t)**(-2) # 0.3
    gamma  = np.zeros(N)
    gamma[path_df["leak"] == 1] = (1.2)**(-2)

    A = scipy.sparse.spdiags(alpha, [0], N, N)
    B = scipy.sparse.spdiags( beta, [0], N-1, N-1)
    C = scipy.sparse.spdiags(gamma, [0], N, N)
    D = scipy.sparse.spdiags(np.stack([-np.ones(N), np.ones(N)]), [0, 1], N-1, N)


    Q = A + (D.T @ B @ D) + C
    c = (A @ xy_hat) + (D.T @ (B @ delta_xy_hat)) + (C @ xy_leak)
    xy_star = scipy.sparse.linalg.spsolve(Q, c)
 
    return pd.DataFrame({
        'site_path_timestamp' : path_df['site_path_timestamp'],
        'floor' : path_df['floor'],
        'x' : xy_star[:, 0],
        'y' : xy_star[:, 1],
    })


# train data
df_sample = pd.read_csv("../input/predict-location-post02/submission.csv")
df_sample["site_id"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[0])
df_sample["path_id"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[1])
df_sample["timestamp"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[2]).astype(float)

list_site = df_sample["site_id"].unique()
EST_before = 0.0
EST = 0.0
for site_id in tqdm(list_site):
    sub = pd.read_csv(f'../input/predict-location-post02/trainp/{site_id}.csv')
    score_before = np.sqrt((sub.waypoint_x-sub.x)**2 + (sub.waypoint_y-sub.y)**2).mean()
    EST_before += score_before*len(df_sample.query("site_id==@site_id"))/len(df_sample)
    sub['site'] = site_id
    sub['path'] = sub["path_id"]

    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        dfs = pool.imap_unordered(correct_path_tr, sub.groupby('path'))
        dfs = tqdm(dfs)
        dfs = list(dfs)
    sub = pd.concat(dfs)#.sort_values('site_path_timestamp')
    sub.to_csv(f'train/{site_id}.csv', index=False)
    score = np.sqrt((sub.waypoint_x-sub.x)**2 + (sub.waypoint_y-sub.y)**2).mean()
    EST += score*len(df_sample.query("site_id==@site_id"))/len(df_sample)
    print(f"{site_id} CV:{score_before} -> {score}")
#     break
print(f"EST:{EST_before} -> {EST}")

# test data
df_sub = pd.read_csv('../input/predict-location-post02/submission.csv')
df_leak = pd.read_csv("../input/martpath/test_all.csv")

tmp = df_sub['site_path_timestamp'].apply(lambda s : pd.Series(s.split('_')))
df_sub['site'] = tmp[0]
df_sub['path_id'] = tmp[1]
df_sub['timestamp'] = tmp[2].astype(float)

list_path = df_sub["path_id"].unique()
for path_id in tqdm(list_path):
    df_sub_path = df_sub.query("path_id == @path_id")
    start_idx = df_sub.loc[df_sub["path_id"] == path_id].index.min()
    end_idx = df_sub.loc[df_sub["path_id"] == path_id].index.max()
    start_x_leak = df_leak.query("path_id == @path_id")["start_waypoint_x"].iloc[0]
    start_y_leak = df_leak.query("path_id == @path_id")["start_waypoint_y"].iloc[0]
    end_x_leak = df_leak.query("path_id == @path_id")["end_waypoint_x"].iloc[0]
    end_y_leak = df_leak.query("path_id == @path_id")["end_waypoint_y"].iloc[0]
    if not np.isnan(start_x_leak):
        df_sub.at[start_idx,"x_leak"] = start_x_leak
        df_sub.at[start_idx,"y_leak"] = start_y_leak
        df_sub.at[start_idx,"leak"] = 1
    if not np.isnan(end_x_leak):
        df_sub.at[end_idx,"x_leak"] = end_x_leak
        df_sub.at[end_idx,"y_leak"] = end_y_leak
        df_sub.at[end_idx,"leak"] = 1
df_sub = df_sub.fillna(0)

processes = multiprocessing.cpu_count()
with multiprocessing.Pool(processes=processes) as pool:
    dfs = pool.imap_unordered(correct_path, df_sub.groupby('path_id'))
    dfs = tqdm(dfs)
    dfs = list(dfs)
sub = pd.concat(dfs).sort_values('site_path_timestamp')
sub[["site_path_timestamp","floor","x","y"]].to_csv('submission.csv', index=False)
